# Sleep Health Demo

Input dataset:
- [sleep_health_dataset.csv](../data/sleep_health_dataset.csv)

Generated report:
- [sleep_health_dataset_demo_report.html](../real_runs/sleep_health_dataset_demo_report.html)

Command used:

```bash
python -m explot data/sleep_health_dataset.csv -o real_runs/sleep_health_dataset_demo_report.html --fast
```

Current headline results:
- Quality score: `87/100`
- Hopkins statistic: `0.6887`
- Intrinsic dimensionality: `15`
- Autoencoder bottleneck: `11`
- Autoencoder reconstruction MSE: `0.3108`
- Best target probe: `exercise_day` with `LogisticRegression`, F1 macro `0.9684`

Interpretation:
- The dataset is structurally healthy enough for downstream analysis.
- It has moderate global structure, but not strong natural clustering.
- The autoencoder finds a compact nonlinear representation without collapsing the data.
- `exercise_day` is highly predictable and should be reviewed for proxy relationships before being treated as a substantive modeling success.
