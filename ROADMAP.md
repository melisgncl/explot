# Explot Roadmap

Current status: pipeline feature-complete across three phases.
This document tracks what remains before explot can be considered portfolio-ready and handoff-safe.

---

## What Was Built (historical)

### The five silent-wrong-answer problems (fixed)

Before Phase 1 the pipeline had five bugs that produced clean-looking reports with incorrect results:

1. Categorical features silently dropped — the feature matrix was numeric-only.
2. NaN handling was passive `dropna()` — no imputation, no row-loss warning.
3. SVM trained on unscaled features — `SVC(rbf)` without `StandardScaler` was meaningless.
4. SHIP verdict ignored absolute performance — a 0.51 model could receive SHIP.
5. 8k row cap was invisible — 1M-row datasets were silently truncated.

All five are fixed. See git log for implementation details.

### Phase 1 — Correct results on clean tabular CSVs ✅

- `PreprocessingStage`: OrdinalEncoder (≤20 unique), frequency encoding (21–100), drop + finding (>100)
- Imputation: numeric median, categorical mode; HIGH finding when >20% imputed
- Scaling inside model pipelines: LR, Ridge, SVM_RBF wrapped in `Pipeline([scaler, model])`; trees unwrapped
- Track C: raw preprocessed features as third supervised track alongside PCA (A) and DVAE (B)
- Verdict performance floor: INVESTIGATE when best model is within 0.03 of baseline
- Verdict row-cap disclosure: INVESTIGATE + `[Sampling]` note when training data was capped

### Phase 2 — Competitive results on real-world data ✅

- Class imbalance: `class_weight='balanced'` auto-applied when `imbalance_ratio >= 5`; finding surfaces the change
- SHAP: `TreeExplainer` for RF/XGBoost/LightGBM, `LinearExplainer` for Ridge/LR, graceful fallback for SVM
- Configurable thresholds: `max_fit_rows`, `leakage_delta_threshold`, `leakage_score_floor`, `verdict_lift_floor` in `config/*.yaml`
- Row-cap badge in HTML hero section; `*(sampled N of M rows)*` in markdown best-models table
- High-cardinality findings: MEDIUM for frequency-encoded columns, HIGH for dropped free-text columns

### Phase 3 — Full product ✅ (partial)

- Batch mode: `explot run ./data/*.csv --output ./reports/` with `index.md` summary
- Run comparison: `explot compare report_v1.json report_v2.json` — diff verdicts, scores, trust flags
- Survival analysis: KM curves, Cox PH model, hazard ratios, log-rank p-values; survival tab in HTML
- Model export: best model saved as `{stem}_model.joblib`; code snippet in HTML report
- Interactive HTML: Plotly scatter (UMAP preferred, PCA fallback), Plotly feature importance bar chart with SHAP/impurity toggle, interactive KM step chart

---

## Open Items

### O1 — Commit all uncommitted work ✅

All Phase 2–3 changes committed. Working tree is clean.

### O2 — Declare optional dependencies in `pyproject.toml` ✅

Optional extras `[ml]`, `[umap]`, `[survival]`, `[autoencoder]` declared.
`pip install "explot[ml,umap,survival]"` produces a fully-featured install.

### O3 — Fix Adult Income F1 target ✅

Resolved: test matrix and assertion now consistently use F1 ≥ 0.75 at 8k cap.
Theoretical ceiling at full 48k rows (~0.82) noted in docstring, not enforced in CI.

### O4 — Validate survival stage on a real dataset ✅

Rossi Recidivism added to `test_real_datasets.py` (library-sourced via `lifelines.datasets`,
always available in CI). Four tests: detection, KM curve, Cox significant covariate, C-index > 0.6.

### O5 — Fill the test matrix gaps ✅ (partial)

Breast Cancer Wisconsin added to `test_real_datasets.py` (library-sourced via
`sklearn.datasets.load_breast_cancer`). Two tests: score >= 0.95, verdict = SHIP.

The remaining five datasets (House Prices, TCGA RNA-seq, TCGA BRCA clinical, MIMIC-III,
Stack Overflow Survey) remain as future validation targets — noted in the test matrix.

### O6 — Suppress the `pd.to_datetime` UserWarning ✅

`format="mixed"` applied in `profiling/stage.py`. Warning no longer fires in the test suite.

### O7 — Confusion matrix click-through (explicitly deferred)

The one unchecked Phase 3e item: "Confusion matrix: click a cell to see which row indices were
misclassified." This is a nice-to-have. Defer until after O1–O4 are done.

---

## Dataset Test Matrix

| Dataset | Rows | What it tests | CI status | Pass criterion |
|---|---|---|---|---|
| Titanic | 891 | Categorical encoding, NaN imputation | ✅ in CI (`fetch_openml`) | `Sex` top-3 importance, F1 ≥ 0.79 |
| UCI Adult Income | 48k | High-cardinality categoricals, row cap | ✅ in CI (`fetch_openml`) | F1 ≥ 0.75 at 8k cap |
| Credit Card Fraud | 284k | Extreme imbalance (0.17% fraud) | ✅ in CI (`data/creditcard_fraud.csv`) | Verdict not SHIP, balanced weights fire |
| Telco Customer Churn | 7k | Moderate imbalance + categoricals | ✅ in CI (`data/telco_churn.csv`) | `Contract` in top-5 importance |
| NYC Taxi Jan 2024 | 1.3M | Row cap disclosure, temporal leakage | ✅ in CI (`data/nyc_taxi_yellow_2024_01.parquet`) | Sampling note in verdict, temporal audit runs |
| Breast Cancer Wisconsin | 569 | High-correlation features, leakage detection | ✅ in CI (`sklearn.datasets`) | Score ≥ 0.95, `single_feature_leakage` fires |
| Rossi Recidivism | 433 | Survival analysis | ✅ in CI (`lifelines.datasets`) | KM renders, Cox C-index ≥ 0.6, ≥ 1 significant covariate |
| House Prices | 1.4k | Regression, many NaN, many categoricals | ❌ no data | RMSE competitive with LightGBM baseline |
| TCGA RNA-seq subset | 1k × 20k | High-dimensional bioinformatics | ❌ no data | Log-norm detected, top genes surface |
| TCGA BRCA clinical | 1k | Survival + clinical covariates | ❌ no data | KM curve renders, Cox HR output |
| MIMIC-III derived | 50k | Group + temporal leakage together | ❌ no data | Both `patient_id` + `charttime` audits fire |
| Stack Overflow Survey | 90k | High-cardinality strings, massive NaN | ❌ no data | High-cardinality findings fire |

---

## Definition of Done

**Portfolio-ready ✅ (O1–O6 complete):**
- All uncommitted work committed and tagged.
- Optional dependencies declared; `pip install "explot[ml,umap,survival]"` produces a fully-featured install.
- Adult Income F1 target consistent between test matrix and assertion.
- Survival validated on Rossi Recidivism (library-sourced, always in CI).
- Breast Cancer Wisconsin in CI as clean-numeric regression test.
- No format-inference warnings in the test suite.
- Every ✅ row in the test matrix has a test that actually runs in CI.

**Full bioinformatics coverage (future):**
- TCGA RNA-seq, TCGA BRCA clinical, MIMIC-III, Stack Overflow Survey added to `data/` and CI.
- High-dimensional mode (20k features) does not OOM at default settings.
- Confusion matrix click-through implemented (O7).
