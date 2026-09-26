# LITH: Lightweight IoT Threat Hunter

This repository contains the code, representative synthetic dataset, experimental outputs, figures, and reproducibility materials supporting the manuscript:

**LITH: A Lightweight Behavioral Anomaly-to-Hunt-Lead Framework for IoT Sensor Ecosystems**

Authors: Fatmah Bamashmoos and Enas Khairullah.

## Reproducibility

The controlled benchmark uses benign-only fitting and validation calibration across 10 Monte Carlo seeds (42–51). Attack observations are excluded from scaling, model fitting, early stopping, and threshold calibration.

The main experiment can be reproduced using:

`code/run_lith_experiment_v4_udp_consistent.py`

A representative seed-42 synthetic dataset and supporting result archives are included.

## Repository Contents

- `code/` — experimental implementation
- `data/` — representative synthetic data
- `results/` — baseline, statistical, sensitivity, hunt-lead, and computational-footprint outputs
- `figures/` — selected manuscript figures

## Status

This repository supports a manuscript currently under peer review. The current release corresponds to the revised experimental analysis.

## License

MIT License.
