#!/usr/bin/env python3
"""LITH v3 controlled telemetry benchmark.

Changes relative to v2:
- count-valued features are discrete integers;
- UDP ratio is reconstructed from an integer UDP-packet count and total packet count;
- unique-destination count is constrained not to exceed packet count;
- packet-size variability is explicitly packet-size standard deviation (bytes);
- adds a simple Max-Z statistical baseline;
- adds matched paired tests with Holm correction and Cohen's dz;
- adds attack-strength x benign-variability sensitivity analysis;
- retains benign-only temporal train/validation/test partitions and attack-only test injection;
- uses a disjoint benign training-partition holdout for autoencoder early stopping, reserving the full benign validation partition for calibration.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_rel
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "packet_count",
    "mean_packet_size_bytes",
    "packet_size_std_bytes",
    "unique_destination_ips",
    "udp_ratio",
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "process_count",
]
COUNT_IDXS = np.array([0, 3, 7])
DEVICES = ["sensor", "camera", "actuator"]
ATTACK_TO_DEVICE = {"ddos": "sensor", "exfil": "camera", "recon": "actuator"}
SEEDS = list(range(42, 52))
TRAIN_PER_DEVICE = 2000
AE_EARLY_STOP_PER_DEVICE = 1500
Q = 0.99
SENSITIVITY_STRENGTHS = [0.25, 0.50, 0.75, 1.00]
SENSITIVITY_VARIABILITY = [1.00, 1.25, 1.50]


def clip(a, lo, hi):
    return np.clip(a, lo, hi)


def as_int_counts(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, COUNT_IDXS] = np.rint(X[:, COUNT_IDXS])
    X[:, 0] = np.maximum(X[:, 0], 1)
    X[:, 3] = np.maximum(X[:, 3], 1)
    X[:, 7] = np.maximum(X[:, 7], 1)
    return X


def enforce_packet_consistency(X):
    """Enforce simple packet-level physical consistency.

    - packet_count is an integer >= 1;
    - unique_destination_ips is an integer in [1, packet_count];
    - udp_ratio is represented by an integer UDP packet count divided by
      packet_count, so every ratio is physically attainable.
    """
    X = as_int_counts(X)

    packet_count = X[:, 0].astype(int)

    # A behavioral observation cannot contain more distinct destinations than packets.
    unique_dest = np.rint(X[:, 3]).astype(int)
    unique_dest = np.clip(unique_dest, 1, packet_count)
    X[:, 3] = unique_dest

    # Convert the latent continuous UDP fraction to the nearest feasible
    # integer UDP packet count, then reconstruct the exact ratio.
    latent_ratio = np.clip(X[:, 4], 0.0, 1.0)
    udp_packets = np.rint(latent_ratio * packet_count).astype(int)
    udp_packets = np.clip(udp_packets, 0, packet_count)
    X[:, 4] = udp_packets / packet_count

    return X


def normal(n, dev, rng, phase=0.0, variability=1.0):
    """Generate benign feature vectors for one device class.

    variability multiplies benign stochastic variation and sinusoidal drift amplitude.
    Count-valued features are returned as integer-valued floats for model compatibility.
    """
    tt = np.linspace(0, 1, n, endpoint=False)
    drift_amp = 0.025 * variability
    drift = 1 + drift_amp * np.sin(2 * np.pi * (tt + phase))

    if dev == "sensor":
        vals = [
            rng.poisson(np.maximum(0.1, 6 * drift)) + 1,
            rng.normal(145, 22 * variability, n) * drift,
            np.abs(rng.normal(28, 10 * variability, n)),
            1 + rng.binomial(2, min(0.25, 0.06 * variability), n),
            clip(rng.normal(0.08, 0.05 * variability, n), 0, 1),
            clip(rng.normal(4, 1.1 * variability, n) * drift, 0.5, 20),
            clip(rng.normal(18, 2.5 * variability, n) * drift, 5, 50),
            clip(np.rint(rng.normal(8, 1.1 * variability, n)), 3, 16),
        ]
    elif dev == "camera":
        vals = [
            np.rint(clip(rng.normal(950, 115 * variability, n) * drift, 350, 1750)),
            clip(rng.normal(1080, 120 * variability, n) * drift, 450, 1550),
            np.abs(rng.normal(180, 45 * variability, n)),
            1 + rng.binomial(3, min(0.35, 0.12 * variability), n),
            clip(rng.normal(0.84, 0.08 * variability, n), 0.35, 1),
            clip(rng.normal(31, 4.5 * variability, n) * drift, 8, 60),
            clip(rng.normal(42, 5 * variability, n) * drift, 15, 75),
            clip(np.rint(rng.normal(18, 2 * variability, n)), 8, 34),
        ]
    else:
        vals = [
            rng.poisson(np.maximum(0.1, 10 * drift)) + 1,
            clip(rng.normal(190, 32 * variability, n) * drift, 60, 430),
            np.abs(rng.normal(40, 14 * variability, n)),
            1 + rng.binomial(2, min(0.30, 0.08 * variability), n),
            clip(rng.normal(0.12, 0.07 * variability, n), 0, 1),
            clip(rng.normal(5.5, 1.4 * variability, n) * drift, 0.5, 22),
            clip(rng.normal(20, 3 * variability, n) * drift, 5, 50),
            clip(np.rint(rng.normal(10, 1.3 * variability, n)), 4, 20),
        ]
    return enforce_packet_consistency(np.column_stack(vals))


def _full_attack_from_base(base, attack_name, rng, variability=1.0):
    """Construct the full-strength attack target from a benign-like base."""
    X = base.copy()
    n = len(X)
    if attack_name == "ddos":
        sev = rng.lognormal(1.55, 0.35, n)
        X[:, 0] *= (2.8 + sev)
        X[:, 1] = clip(rng.normal(260, 90 * variability, n), 70, 750)
        X[:, 2] = clip(rng.normal(95, 40 * variability, n), 10, 280)
        X[:, 3] = clip(np.rint(rng.normal(3.5, 1.8 * variability, n)), 1, 14)
        X[:, 4] = clip(rng.normal(0.88, 0.10 * variability, n), 0.35, 1)
        X[:, 5] = clip(X[:, 5] + rng.normal(5, 2.2 * variability, n), 1, 40)
        X[:, 6] = clip(X[:, 6] + rng.normal(3, 2 * variability, n), 5, 60)
    elif attack_name == "exfil":
        X[:, 0] *= rng.normal(1.16, 0.11 * variability, n)
        X[:, 1] *= rng.normal(0.90, 0.08 * variability, n)
        X[:, 2] *= rng.normal(1.28, 0.20 * variability, n)
        X[:, 3] += rng.choice([1, 1, 1, 2], n)
        X[:, 4] = clip(X[:, 4] - np.abs(rng.normal(0.20, 0.09 * variability, n)), 0.15, 1)
        X[:, 5] = clip(X[:, 5] + rng.normal(4.5, 3 * variability, n), 8, 80)
        X[:, 6] = clip(X[:, 6] + rng.normal(4, 2.5 * variability, n), 12, 90)
        X[:, 7] = clip(X[:, 7] + rng.binomial(2, 0.5, n), 8, 38)
    else:
        X[:, 0] *= rng.normal(3.4, 0.75 * variability, n)
        X[:, 1] = clip(rng.normal(105, 30 * variability, n), 35, 260)
        X[:, 2] = clip(rng.normal(65, 22 * variability, n), 8, 180)
        X[:, 3] = clip(np.rint(rng.normal(10, 4 * variability, n)), 3, 32)
        X[:, 4] = clip(rng.normal(0.05, 0.04 * variability, n), 0, 0.30)
        X[:, 5] = clip(X[:, 5] + rng.normal(3, 2 * variability, n), 1, 35)
        X[:, 6] = clip(X[:, 6] + rng.normal(1.5, 1.5 * variability, n), 5, 55)
    return enforce_packet_consistency(X)


def attack(n, attack_name, rng, strength=1.0, variability=1.0):
    """Generate one attack class at a controlled deviation strength.

    The same benign-like base is blended toward a full-strength attack target.
    strength=0 would reproduce the base; strength=1 uses the full injected attack.
    """
    d = ATTACK_TO_DEVICE[attack_name]
    phase = {"ddos": 0.15, "exfil": 0.25, "recon": 0.35}[attack_name]
    base = normal(n, d, rng, phase=phase, variability=variability)
    target = _full_attack_from_base(base, attack_name, rng, variability=variability)
    X = base + float(strength) * (target - base)
    X[:, 4] = clip(X[:, 4], 0, 1)
    return enforce_packet_consistency(X)


def generate(seed, variability=1.0, attack_strength=1.0):
    rng = np.random.default_rng(seed)
    counts = {"sensor": 16667, "camera": 16667, "actuator": 16666}
    dct = {}
    for i, (d, n) in enumerate(counts.items()):
        X = normal(n, d, rng, phase=i * 0.17, variability=variability)
        a = int(0.6 * n)
        b = int(0.8 * n)
        dct[d] = {"train": X[:a], "val": X[a:b], "test": X[b:]}
    ac = {"ddos": 1667, "exfil": 1667, "recon": 1666}
    attacks = {a: attack(n, a, rng, strength=attack_strength, variability=variability) for a, n in ac.items()}
    return dct, attacks


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 6), nn.ReLU(),
            nn.Linear(6, 4), nn.ReLU(),
            nn.Linear(4, 3), nn.ReLU(),
            nn.Linear(3, 4), nn.ReLU(),
            nn.Linear(4, 6), nn.ReLU(),
            nn.Linear(6, 8),
        )

    def forward(self, x):
        return self.net(x)


def train_ae(Xtr, X_es, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    m = AE()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=256, shuffle=True, generator=gen)
    x_es = torch.tensor(X_es, dtype=torch.float32)
    best = 1e99
    wait = 0
    best_state = None
    epochs_run = 0
    for ep in range(25):
        epochs_run = ep + 1
        m.train()
        for (xb,) in dl:
            opt.zero_grad()
            y = m(xb)
            l = loss(y, xb)
            l.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = loss(m(x_es), x_es).item()
        if vl < best - 1e-4:
            best = vl
            wait = 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 4:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, epochs_run


def ae_score(m, X):
    m.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        return ((x - m(x)) ** 2).mean(1).numpy()


def raw_scores(name, model, X):
    if name == "IF":
        return -model.score_samples(X)
    if name == "OCSVM":
        return -model.decision_function(X)
    if name == "AE":
        return ae_score(model, X)
    if name == "Max-Z":
        return np.max(np.abs(X), axis=1)
    raise ValueError(name)


def empirical_percentile(val_sorted, s):
    return np.searchsorted(val_sorted, s, side="right") / len(val_sorted)


def metrics(y, s, pred):
    pp, rr, ff, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "precision": pp,
        "recall": rr,
        "f1": ff,
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "fpr": fp / (fp + tn),
    }


def summarize(df, groups):
    ms = ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]
    out = []
    for keys, g in df.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(groups, keys))
        n = len(g)
        for m in ms:
            mean = g[m].mean()
            sd = g[m].std(ddof=1)
            ci = t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
            r[m + "_mean"] = mean
            r[m + "_sd"] = sd
            r[m + "_ci95"] = ci
        out.append(r)
    return pd.DataFrame(out)


def fit_bundle(name, data, seed, cols=np.arange(8)):
    bundle = {}
    rng = np.random.default_rng(seed + 777)
    ae_epochs = []
    for di, d in enumerate(DEVICES):
        sc = StandardScaler().fit(data[d]["train"][:, cols])
        tr = sc.transform(data[d]["train"][:, cols])
        va = sc.transform(data[d]["val"][:, cols])
        idx = rng.choice(len(tr), TRAIN_PER_DEVICE, replace=False)
        tr_fit = tr[idx]
        if name == "IF":
            m = IsolationForest(
                n_estimators=100,
                max_samples=256,
                max_features=1.0,
                contamination="auto",
                random_state=seed + di,
                n_jobs=-1,
            ).fit(tr_fit)
        elif name == "OCSVM":
            m = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale", tol=0.001, cache_size=1000).fit(tr_fit)
        elif name == "AE":
            # AE architecture assumes all eight features; ablation uses IF only.
            # Early stopping uses a benign holdout drawn from the training
            # partition after excluding the observations used for model fitting.
            # The full benign validation partition is reserved exclusively for
            # anomaly-score calibration.
            remaining = np.setdiff1d(
                np.arange(len(tr)),
                idx,
                assume_unique=False,
            )
            if len(remaining) < AE_EARLY_STOP_PER_DEVICE:
                raise ValueError(
                    f"Not enough disjoint benign training observations for AE early stopping "
                    f"in device class {d}: need {AE_EARLY_STOP_PER_DEVICE}, "
                    f"found {len(remaining)}."
                )
            es_idx = rng.choice(
                remaining,
                AE_EARLY_STOP_PER_DEVICE,
                replace=False,
            )
            es = tr[es_idx]
            m, epochs = train_ae(tr_fit, es, seed + di)
            ae_epochs.append((seed, d, epochs))
        elif name == "Max-Z":
            m = None
        else:
            raise ValueError(name)
        vs = raw_scores(name, m, va)
        bundle[d] = (sc, m, np.sort(vs))
    return bundle, ae_epochs


def score_bundle(name, bundle, data, attacks, cols=np.arange(8)):
    scores, labels, atypes, preds = [], [], [], []
    for d in DEVICES:
        sc, m, vsort = bundle[d]
        X = sc.transform(data[d]["test"][:, cols])
        rs = raw_scores(name, m, X)
        ps = empirical_percentile(vsort, rs)
        scores.extend(ps)
        labels.extend([0] * len(ps))
        atypes.extend(["benign"] * len(ps))
        preds.extend(ps > Q)
    for a, Xraw in attacks.items():
        d = ATTACK_TO_DEVICE[a]
        sc, m, vsort = bundle[d]
        X = sc.transform(Xraw[:, cols])
        rs = raw_scores(name, m, X)
        ps = empirical_percentile(vsort, rs)
        scores.extend(ps)
        labels.extend([1] * len(ps))
        atypes.extend([a] * len(ps))
        preds.extend(ps > Q)
    return np.array(scores), np.array(labels), np.array(atypes), np.array(preds, dtype=int)


def bench_bundle(name, bundle, data, cols=np.arange(8), repeats=50):
    size = 0
    td = OUT / "_tmp"
    td.mkdir(exist_ok=True)
    for d, (sc, m, vsort) in bundle.items():
        p = td / f"{name}_{d}.bin"
        if name == "AE":
            torch.save(m.state_dict(), p)
        elif name == "Max-Z":
            # no fitted detector beyond StandardScaler; report detector size as zero
            continue
        else:
            joblib.dump(m, p, compress=0)
        size += p.stat().st_size
        p.unlink()
    groups = {d: data[d]["test"][:1000, cols] for d in DEVICES}

    def runonce():
        for d, Xraw in groups.items():
            sc, m, vsort = bundle[d]
            X = sc.transform(Xraw)
            raw_scores(name, m, X)

    runonce()
    ts = []
    for _ in range(repeats):
        st = time.perf_counter()
        runonce()
        ts.append(time.perf_counter() - st)
    med = float(np.median(ts))
    return {
        "model": name,
        "bundle_size_kb": size / 1024,
        "median_inference_ms_per_1000": med / 3000 * 1e6,
    }


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def paired_tests(overall):
    comparisons = [("OCSVM", "IF"), ("OCSVM", "AE"), ("OCSVM", "Max-Z")]
    rows = []
    pvals = []
    for a, b in comparisons:
        xa = overall[overall.model == a].sort_values("seed").f1.values
        xb = overall[overall.model == b].sort_values("seed").f1.values
        diff = xa - xb
        tt, pv = ttest_rel(xa, xb)
        n = len(diff)
        md = diff.mean()
        sd = diff.std(ddof=1)
        ci = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        row = {
            "comparison": f"{a} vs {b}",
            "mean_f1_difference": md,
            "ci95_low": md - ci,
            "ci95_high": md + ci,
            "t": tt,
            "df": n - 1,
            "p_raw": pv,
            "cohens_dz": md / sd if sd > 0 else np.nan,
        }
        rows.append(row)
        pvals.append(pv)
    adj = holm_adjust(np.array(pvals, dtype=float))
    for r, p in zip(rows, adj):
        r["p_holm"] = p
    return pd.DataFrame(rows)


def sensitivity_analysis():
    rows = []
    # Refit IF only once per seed/variability level, then score matched attack-strength variants.
    for variability in SENSITIVITY_VARIABILITY:
        for seed in SEEDS:
            data, _ = generate(seed, variability=variability, attack_strength=1.0)
            bundle, _ = fit_bundle("IF", data, seed)
            for strength in SENSITIVITY_STRENGTHS:
                # same benign data; attack randomness is deterministic for a given seed/variability/strength call
                # generate only attacks through a separate reproducible RNG stream.
                rng = np.random.default_rng(seed + int(round(variability * 1000)))
                ac = {"ddos": 1667, "exfil": 1667, "recon": 1666}
                attacks = {
                    a: attack(n, a, rng, strength=strength, variability=variability)
                    for a, n in ac.items()
                }
                s, y, at, p = score_bundle("IF", bundle, data, attacks)
                r = metrics(y, s, p)
                r.update(seed=seed, variability_factor=variability, attack_strength=strength)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_sensitivity_runs.csv", index=False)
    summary = summarize(df, ["variability_factor", "attack_strength"])
    summary.to_csv(OUT / "summary_sensitivity.csv", index=False)
    return df, summary


def main():
    overall = []
    attackrows = []
    abrows = []
    ae_epoch_rows = []
    rep = None

    for seed in SEEDS:
        print("seed", seed, flush=True)
        data, attacks = generate(seed, variability=1.0, attack_strength=1.0)
        bundles = {}
        for name in ["IF", "OCSVM", "AE", "Max-Z"]:
            b, ep = fit_bundle(name, data, seed)
            bundles[name] = b
            for s, d, e in ep:
                ae_epoch_rows.append({"seed": s, "device": d, "epochs_run": e})

        for name, b in bundles.items():
            s, y, at, p = score_bundle(name, b, data, attacks)
            r = metrics(y, s, p)
            r.update(seed=seed, model=name)
            overall.append(r)

        # Isolation Forest attack-specific analysis against the same benign test population.
        s, y, at, p = score_bundle("IF", bundles["IF"], data, attacks)
        benign = at == "benign"
        for a in ["ddos", "exfil", "recon"]:
            mask = benign | (at == a)
            r = metrics(y[mask], s[mask], p[mask])
            r.update(seed=seed, attack=a)
            attackrows.append(r)

        # Isolation Forest feature ablation.
        for label, cols in [
            ("Network-only", np.arange(5)),
            ("System-only", np.arange(5, 8)),
            ("Combined", np.arange(8)),
        ]:
            b, _ = fit_bundle("IF", data, seed, cols)
            s, y, at, p = score_bundle("IF", b, data, attacks, cols)
            r = metrics(y, s, p)
            r.update(seed=seed, feature_set=label)
            abrows.append(r)

        if seed == 42:
            rep = (data, attacks, bundles)

    overall = pd.DataFrame(overall)
    attacksdf = pd.DataFrame(attackrows)
    ab = pd.DataFrame(abrows)
    ae_epochs = pd.DataFrame(ae_epoch_rows)

    overall.to_csv(OUT / "results_overall_runs.csv", index=False)
    attacksdf.to_csv(OUT / "results_attack_runs.csv", index=False)
    ab.to_csv(OUT / "results_ablation_runs.csv", index=False)
    ae_epochs.to_csv(OUT / "ae_epochs.csv", index=False)

    so = summarize(overall, ["model"])
    sa = summarize(attacksdf, ["attack"])
    sab = summarize(ab, ["feature_set"])
    so.to_csv(OUT / "summary_overall.csv", index=False)
    sa.to_csv(OUT / "summary_attack.csv", index=False)
    sab.to_csv(OUT / "summary_ablation.csv", index=False)

    tests = paired_tests(overall)
    tests.to_csv(OUT / "paired_tests.csv", index=False)

    sensitivity_runs, sensitivity_summary = sensitivity_analysis()

    # Representative analytics-side resource benchmark.
    data, attacks, bundles = rep
    rb = [bench_bundle(n, bundles[n], data) for n in ["IF", "OCSVM", "AE", "Max-Z"]]
    pd.DataFrame(rb).to_csv(OUT / "resource_benchmark.csv", index=False)

    # Seed-42 audit dataset.
    rows = []
    for d in DEVICES:
        for split in ["train", "val", "test"]:
            for x in data[d][split]:
                rows.append((*x, d, "benign", split))
    for a, X in attacks.items():
        for x in X:
            rows.append((*x, ATTACK_TO_DEVICE[a], a, "test"))
    audit = pd.DataFrame(rows, columns=FEATURES + ["device_class", "label", "split"])
    audit.to_csv(OUT / "LITH_synthetic_seed42.csv", index=False)

    # Audit discrete-count and packet-level physical-consistency invariants.
    discrete_audit = {
        FEATURES[i]: bool(np.allclose(audit[FEATURES[i]].to_numpy(), np.rint(audit[FEATURES[i]].to_numpy())))
        for i in COUNT_IDXS
    }
    pc = audit["packet_count"].to_numpy()
    ud = audit["unique_destination_ips"].to_numpy()
    ur = audit["udp_ratio"].to_numpy()
    udp_counts = ur * pc

    physical_audit = {
        **discrete_audit,
        "packet_count_at_least_1": bool(np.all(pc >= 1)),
        "unique_destination_ips_le_packet_count": bool(np.all(ud <= pc)),
        "udp_ratio_in_0_1": bool(np.all((ur >= 0) & (ur <= 1))),
        "udp_ratio_implies_integer_udp_packet_count": bool(np.allclose(udp_counts, np.rint(udp_counts), atol=1e-10)),
    }
    (OUT / "discrete_feature_audit.json").write_text(json.dumps(physical_audit, indent=2))

    # Actual hunt lead from representative seed-42 DDoS observation.
    sc, m, vsort = bundles["IF"]["sensor"]
    Xs = sc.transform(attacks["ddos"])
    rs = raw_scores("IF", m, Xs)
    ps = empirical_percentile(vsort, rs)
    # Choose the observation with the largest raw anomaly score for a concrete high-priority lead.
    j = int(np.argmax(rs))
    z = Xs[j]
    lead = {
        "seed": 42,
        "device_class": "sensor",
        "attack_scenario": "ddos",
        "anomaly_percentile": float(ps[j]),
        "decision_threshold_percentile": Q,
        "raw_anomaly_score": float(rs[j]),
        "changed_features": [FEATURES[i] for i in np.argsort(np.abs(z))[::-1][:4]],
        "raw_values": dict(zip(FEATURES, map(float, attacks["ddos"][j]))),
        "device_profile_z_scores": dict(zip(FEATURES, map(float, z))),
    }
    (OUT / "hunt_lead_seed42.json").write_text(json.dumps(lead, indent=2))

    # Environment metadata.
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem = np.nan
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = float(line.split()[1]) / 1024 / 1024
                break
    except Exception:
        pass
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpus_visible": os.cpu_count(),
        "memory_gb_visible": mem,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "torch": torch.__version__,
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))

    with open(OUT / "EXPERIMENT_SUMMARY.txt", "w") as f:
        f.write("LITH v3 controlled telemetry experiment\n=====================================\n\n")
        f.write("10 independent Monte Carlo runs (seeds 42-51); 50,000 benign + 5,000 attack observations per baseline run.\n")
        f.write("Count-valued features are discrete; packet-size variability is the packet-size standard deviation in bytes.\n")
        f.write("Device-specific profiles; benign-only temporal 60/20/20 train/validation/test split.\n")
        f.write("2,000 benign training observations/device/model; StandardScaler fitted per device profile.\n")
        f.write("Decision threshold: 99th percentile of benign validation anomaly scores per device profile.\n")
        f.write("Sensitivity grid: attack strength 0.25/0.50/0.75/1.00 x benign variability 1.00/1.25/1.50.\n\n")
        f.write("OVERALL\n" + so.to_string(index=False))
        f.write("\n\nATTACK\n" + sa.to_string(index=False))
        f.write("\n\nABLATION\n" + sab.to_string(index=False))
        f.write("\n\nPAIRED TESTS\n" + tests.to_string(index=False))
        f.write("\n\nSENSITIVITY\n" + sensitivity_summary.to_string(index=False))
        f.write("\n\nRESOURCE\n" + pd.DataFrame(rb).to_string(index=False))
        f.write("\n\nPHYSICAL CONSISTENCY AUDIT\n" + json.dumps(physical_audit, indent=2))
        f.write("\n\nLEAD\n" + json.dumps(lead, indent=2))
        f.write("\n\nENV\n" + json.dumps(env, indent=2))

    print("\nOVERALL\n", so.to_string(index=False))
    print("\nATTACK\n", sa.to_string(index=False))
    print("\nABLATION\n", sab.to_string(index=False))
    print("\nPAIRED TESTS\n", tests.to_string(index=False))
    print("\nSENSITIVITY\n", sensitivity_summary.to_string(index=False))
    print("\nRESOURCE\n", pd.DataFrame(rb).to_string(index=False))
    print("\nPHYSICAL CONSISTENCY AUDIT\n", json.dumps(physical_audit, indent=2))
    print("\nLEAD\n", json.dumps(lead, indent=2))


if __name__ == "__main__":
    main()
