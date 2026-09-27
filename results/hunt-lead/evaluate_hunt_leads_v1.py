"""
Supplementary hunt-lead evaluation for LITH.

Evaluates:
1) Precision@k and Recall@k for k = 50, 100, 500 per 10,000 observations.
   Ranking is evaluated within each device-class-specific queue because raw
   detector scores are only used as tie-breakers within a device class and
   are not comparable across classes. Counts are then pooled across the
   three class-specific queues for each Monte Carlo seed.
2) Generator-consistency fidelity of the four reported changed indicators.
   Changed indicators are the four largest absolute device-class-standardized
   feature deviations. Fidelity is the fraction of those four indicators
   belonging to the feature set intentionally modified by the generator.

This script imports the corrected v4 LITH experiment implementation.
"""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

CODE_PATH = Path("/mnt/data/run_lith_experiment_v4_udp_consistent.py")
OUT = Path("/mnt/data/lith_hunt_eval_outputs")
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("lith_v4", CODE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PERTURBED = {
    "ddos": set(range(7)),      # features 0-6; process_count unchanged
    "exfil": set(range(8)),     # all eight features modified
    "recon": set(range(7)),     # features 0-6; process_count unchanged
}
DEVICE_TO_ATTACK = {v: k for k, v in m.ATTACK_TO_DEVICE.items()}

queue_rows = []
fidelity_rows = []

for seed in m.SEEDS:
    data, attacks = m.generate(seed, variability=1.0, attack_strength=1.0)
    bundle, _ = m.fit_bundle("IF", data, seed)

    for device in m.DEVICES:
        attack_name = DEVICE_TO_ATTACK[device]
        scaler, model, val_sorted = bundle[device]

        Xb = scaler.transform(data[device]["test"])
        Xa = scaler.transform(attacks[attack_name])

        rb = m.raw_scores("IF", model, Xb)
        ra = m.raw_scores("IF", model, Xa)
        pb = m.empirical_percentile(val_sorted, rb)
        pa = m.empirical_percentile(val_sorted, ra)

        raw = np.concatenate([rb, ra])
        pct = np.concatenate([pb, pa])
        y = np.concatenate([
            np.zeros(len(pb), dtype=int),
            np.ones(len(pa), dtype=int),
        ])

        # Device-class-specific ranking:
        # calibrated percentile first, raw anomaly score second.
        order = np.lexsort((-raw, -pct))
        n = len(y)
        attack_total = int(y.sum())

        for k_norm in [50, 100, 500]:
            k_actual = max(1, int(round(k_norm * n / 10000)))
            selected = order[:k_actual]
            tp = int(y[selected].sum())

            queue_rows.append({
                "seed": seed,
                "device_class": device,
                "attack": attack_name,
                "queue_size": n,
                "k_per_10000": k_norm,
                "k_actual": k_actual,
                "tp_at_k": tp,
                "attack_total": attack_total,
                "precision_at_k": tp / k_actual,
                "recall_at_k": tp / attack_total,
            })

        # Changed indicators: fixed top-4 absolute standardized deviations.
        # Evaluate only true-positive leads at the baseline p > 0.99 threshold.
        for idx in np.where(pa > m.Q)[0]:
            z = Xa[idx]
            top4_idx = np.argsort(np.abs(z))[::-1][:4]
            top4 = set(top4_idx.tolist())
            overlap = len(top4 & PERTURBED[attack_name])

            fidelity_rows.append({
                "seed": seed,
                "device_class": device,
                "attack": attack_name,
                "indicator_fidelity": overlap / 4.0,
                "all_four_generator_consistent": int(overlap == 4),
                "top4_features": "|".join(m.FEATURES[i] for i in top4_idx),
            })

queue = pd.DataFrame(queue_rows)
fidelity = pd.DataFrame(fidelity_rows)

seed_queue = (
    queue.groupby(["seed", "k_per_10000"], as_index=False)
    .agg(
        tp_at_k=("tp_at_k", "sum"),
        selected=("k_actual", "sum"),
        attack_total=("attack_total", "sum"),
    )
)
seed_queue["precision_at_k"] = seed_queue["tp_at_k"] / seed_queue["selected"]
seed_queue["recall_at_k"] = seed_queue["tp_at_k"] / seed_queue["attack_total"]

queue_summary = (
    seed_queue.groupby("k_per_10000", as_index=False)
    .agg(
        precision_mean=("precision_at_k", "mean"),
        precision_sd=("precision_at_k", "std"),
        recall_mean=("recall_at_k", "mean"),
        recall_sd=("recall_at_k", "std"),
        selected_mean=("selected", "mean"),
    )
)

fidelity_by_attack = (
    fidelity.groupby("attack", as_index=False)
    .agg(
        true_positive_leads=("indicator_fidelity", "size"),
        mean_indicator_fidelity=("indicator_fidelity", "mean"),
        full_top4_fidelity_rate=("all_four_generator_consistent", "mean"),
    )
)

overall = pd.DataFrame([{
    "true_positive_leads": len(fidelity),
    "mean_indicator_fidelity": fidelity["indicator_fidelity"].mean(),
    "full_top4_fidelity_rate": fidelity["all_four_generator_consistent"].mean(),
}])

queue.to_csv(OUT / "hunt_queue_metrics_by_class_seed.csv", index=False)
seed_queue.to_csv(OUT / "hunt_queue_metrics_by_seed.csv", index=False)
queue_summary.to_csv(OUT / "hunt_queue_metrics_summary.csv", index=False)
fidelity.to_csv(OUT / "explanation_fidelity_leads.csv", index=False)
fidelity_by_attack.to_csv(OUT / "explanation_fidelity_by_attack.csv", index=False)
overall.to_csv(OUT / "explanation_fidelity_overall.csv", index=False)

print(queue_summary)
print(fidelity_by_attack)
print(overall)
