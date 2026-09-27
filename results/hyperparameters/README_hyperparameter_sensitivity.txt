LITH hyperparameter sensitivity check
=======================================
Baseline synthetic benchmark: corrected v4 generator and protocol.
Seeds: 42-51.
Training budget: 2,000 benign observations per device class.
Calibration: full benign validation partition.
Decision threshold: p(s) > 0.99.

Isolation Forest:
- n_estimators varied over {50, 100, 200}.
- max_samples=256, max_features=1.0, contamination='auto' unchanged.
- No attack label was used for fitting or calibration.

Autoencoder:
- architecture varied only at bottleneck: 8-6-4-b-4-6-8, b in {2,3,4}.
- Adam 0.001, batch 256, maximum 25 epochs, patience 4 unchanged.
- The same disjoint 1,500-observation benign training holdout was used for early stopping.
- No attack label was used for fitting, early stopping, or calibration.

The sensitivity grid is a post-hoc robustness check requested by the reviewer.
It was not used to re-select the baseline hyperparameters using attack labels.
