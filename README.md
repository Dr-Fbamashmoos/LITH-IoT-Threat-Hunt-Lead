# LITH: Lightweight IoT Threat Hunter

This repository contains the code, controlled telemetry generator, representative data, and experimental outputs supporting the manuscript:

**LITH: A Lightweight Behavioral Anomaly-to-Hunt-Lead Framework for Resource-Constrained IoT Sensor Ecosystems**

Authors:
- Fatmah Bamashmoos
- Enas Khairullah

## Overview

LITH is an anomaly-to-hunt-lead framework for IoT sensor ecosystems. It separates automated anomaly detection from downstream analyst or SIEM-supported threat investigation.

The repository supports reproduction of the controlled synthetic experiments reported in the manuscript.

## Repository Contents

- `code/` — experimental implementation and telemetry generator
- `data/seed42/` — representative synthetic dataset
- `results/` — detector, sensitivity, statistical, and hunt-lead evaluation outputs
- `figures/` — selected manuscript figures
- `supplementary/` — complete supplementary package
- `requirements.txt` — software dependencies

## Experimental Setup

The baseline experiment uses:

- 10 Monte Carlo runs (seeds 42–51)
- 50,000 benign observations per run
- 5,000 test-only attack observations per run
- device-class-specific benign-only model fitting
- 60/20/20 temporal train/validation/test split
- validation-based empirical percentile calibration

Evaluated detectors:

- Isolation Forest
- One-Class SVM
- shallow autoencoder
- Max-Z statistical baseline

## Reproducibility

The complete synthetic benchmark can be regenerated using the supplied generator and documented random seeds.

Example:

```bash
python code/run_lith_experiment_v4_udp_consistent.py
