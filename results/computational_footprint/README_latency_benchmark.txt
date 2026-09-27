LITH controlled analytics-side latency benchmark
================================================
Representative seed: 42
Threads: 1 for numerical/model inference
Batch size: 1,000 observations per device class
Device classes: 3
Total observations per timed repetition: 3,000
Warm-up repetitions: 10
Timed repetitions: 100
Statistic reported: median wall-clock latency
Timing scope: StandardScaler transform plus raw anomaly-score computation.
Empirical percentile lookup/calibration is excluded.
Latency is normalized to milliseconds per 1,000 observations.
Serialized detector sizes exclude scalers and validation-score arrays.
Max-Z has no fitted detector state beyond the excluded StandardScaler.
