# Explot

[![CI](https://github.com/melisgncl/explot/actions/workflows/ci.yml/badge.svg)](https://github.com/melisgncl/explot/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-180%2B-brightgreen)](https://github.com/melisgncl/explot/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo](https://melisgncl.github.io/explot/)** · [Credit Card Fraud Report](https://melisgncl.github.io/explot/reports/creditcard_fraud_report.html) · [Telco Churn Report](https://melisgncl.github.io/explot/reports/telco_churn_report.html) · [scRNA-seq Simulated Report](https://melisgncl.github.io/explot/reports/scrna_simulated_report.html)

![Explot report preview](docs/images/explot-demo-preview.svg)

**You have a dataset and don't know where to start.** Which model fits? Which features matter? Is there even structure worth modeling — and can you trust the result? Explot is a first-pass triage that answers those questions from a single CSV, before you commit to an analysis plan.

Profilers (ydata-profiling, sweetviz) *describe* your data and stop. AutoML tools (PyCaret, FLAML) jump straight to *training* — and will happily hand you a leaked 0.99 model. Explot sits in the gap between them: it orients you (is there structure? which features matter? which model fits?) **and** gates the recommendation with trust flags, so a strong score is a signal you can act on instead of a trap.

```
profile  →  probe  →  trust
describe    which model      can I believe
the data    & features fit   the score?
```

## Why I Built This

Working with biological and tabular datasets, I kept hitting the same wall: a new dataset lands, and before any real analysis you have to spend hours answering the same setup questions — is there structure, which features carry signal, which model is even appropriate, and is that suspiciously good score real or leakage? Explot automates that first pass into one command, so the hours go into the actual analysis instead of the orientation.

## How Explot Compares

| Feature | ydata-profiling | sweetviz | Explot |
|---|---|---|---|
| Column profiling | ✅ | ✅ | ✅ |
| Correlation analysis | ✅ | ✅ | ✅ |
| ML model selection | ❌ | ❌ | ✅ |
| Leakage / trust flags | ❌ | ❌ | ✅ |
| Survival analysis | ❌ | ❌ | ✅ |
| Omics normalization detection | ❌ | ❌ | ✅ |
| SHAP feature importance | ❌ | ❌ | ✅ |
| Self-contained HTML | ❌ | ✅ | ✅ (requires Plotly CDN for charts) |

## Quick Start

```bash
pip install -e .

# Basic run
explot data/sleep_health_dataset.csv -o report.html

# Fast mode (sampling for speed)
explot data/sleep_health_dataset.csv -o report.html --fast

# JSON output (programmatic use)
explot data/sleep_health_dataset.csv -o results.json --json --fast
```

See [Installation Options](#installation-options) for extras (SHAP, survival, DVAE).

## Try It Without Installing

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melisgncl/explot/blob/main/demo.ipynb)

Or install directly from GitHub (no cloning needed):

```bash
pip install "git+https://github.com/melisgncl/explot.git[ml,survival]"
explot mydata.csv -o report.html --fast
```

> **Demo data:** `data/telco_churn.csv` is included. For the credit card fraud demo,
> download from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> and place it at `data/creditcard_fraud.csv`.

## Use With Your Own Data

Explot is designed so someone can point it at a tabular file and get a report back.

Supported inputs:
- `.csv`
- `.tsv`
- Excel files such as `.xlsx`
- Parquet files such as `.parquet`

Typical workflow:

```bash
# 1. Clone the repo and enter it
git clone https://github.com/melisgncl/explot.git
cd explot

# 2. Install the package
pip install -e .

# 3. Run it on your own dataset
explot path/to/your_data.csv -o my_report.html
```

Fast mode is useful for a quick first pass on larger datasets:

```bash
explot path/to/your_data.csv -o my_report.html --fast
```

If someone wants machine-readable output instead of HTML:

```bash
explot path/to/your_data.csv -o my_results.json --json --fast
```

What they do next:
- Open the generated HTML report in a browser
- Start in the `Overview` and `Findings` tabs
- Check trust flags before trusting strong model scores
- Use the report as a first-pass analysis, not a final scientific conclusion

## What the Report Tells You

Explot runs 9 stages, each building on the last:

| Stage | What it does |
|-------|-------------|
| **Profiling** | Column types, missing values, suspicious columns, quality score (0-100) |
| **Exploration** | Redundant feature pairs, cluster tendency (Hopkins), missingness patterns, outliers |
| **Preprocessing** | Median imputation, ordinal/frequency encoding, drops ID/timestamp columns |
| **Dimensionality** | PCA variance decomposition, intrinsic dimensionality estimate, scree plot |
| **DVAE** | Nonlinear compression via denoising VAE, compared against PCA. *Only runs on high-dimensional data (≥12 features) — on low-dimensional tables PCA + a tree model capture the same structure, so it's skipped automatically. Its real home is omics-scale data.* |
| **Unsupervised** | KMeans sweep, DBSCAN auto-tuning, Isolation Forest anomaly detection, multi-signal consensus |
| **Model Selection** | Auto-detects classification and regression targets, runs cross-validated probes with SHAP importance, recommends the best model |
| **Survival** | Kaplan-Meier curves and Cox PH model when time + event columns are detected (requires `lifelines`) |
| **Findings** | Cross-stage synthesis with confidence levels, class imbalance warnings, and suggested next steps |

The output is an HTML file with tabbed navigation. Interactive charts use the Plotly CDN (requires internet to view charts); the rest of the report opens in any browser offline.

## Trust Flags

The model selection stage doesn't just report scores — it warns you when scores shouldn't be trusted.

| Flag | What it means |
|------|--------------|
| `single_feature_leakage` | One feature alone predicts the target nearly as well as the full model |
| `exact_copy_feature` | A feature is an exact copy of the target column |
| `proxy_like_feature` | A feature deterministically maps to the target (feature→target direction) |
| `high_correlation_proxy` | A feature has |r| > 0.995 with the target |
| `near_perfect_score` | F1 > 0.95 (classification) or R2 > 0.99 (regression) — investigate before celebrating |
| `suspicious_feature_name` | Feature name shares tokens with target name |
| `severe_class_imbalance` | Target class ratio exceeds 10:1 — metrics may be misleading |
| `class_imbalance` | Target class ratio exceeds 5:1 |

## How to Read the Results

**3-minute read:**
1. Open the **Overview** tab — check quality score and top 3 findings
2. Jump to **Findings** — HIGH-confidence items are actionable, LOW is noise
3. Check **Model Selection** — look at trust flags first, scores second

**Deep read:**
- **Profiling** — verify suspicious columns were caught (especially ID columns)
- **Exploration** — redundant pairs tell you what to drop; Hopkins < 0.5 means clustering won't help
- **Dimensionality** — intrinsic dim estimate reveals how many features actually matter
- **Model Selection** — PCA features vs DVAE latent features comparison shows whether nonlinear structure exists; check class imbalance warnings for skewed targets

## Example Output

**Credit card fraud** (284k rows, 578:1 imbalance — trust flags are the story):
```
HIGH: Target 'Class' is highly predictable: RandomForest F1=0.86.
      Trust flags: near_perfect_score, severe_class_imbalance
HIGH: Severe class imbalance (578:1). Balanced weights applied automatically.
      F1 may be misleading — check precision/recall separately.
HIGH: SHAP top features: V14, V10, V12 (PCA-transformed anonymized features).

Suggested next steps:
- Do not trust F1 at face value: 578:1 imbalance means Accuracy will be near-perfect trivially
- Evaluate using AUPRC (precision-recall AUC), not ROC AUC
```

**Sleep health** (leakage detection demo):
```
HIGH: Target 'exercise_day' is highly predictable: LogisticRegression F1=0.94.
      Trust flags: possible_leakage, single_feature_leakage
MEDIUM: Moderate cluster tendency (Hopkins=0.48).
```

## Architecture

```
CSV / TSV / Excel / Parquet
         |
    load_table()
         |
   PipelineState
         |
    +----+----+----+----+----+----+----+----+----+
    | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
    |Prof|Expl|Pre |Dim |DVAE|Uns |Sup |Surv|Find|
    +----+----+----+----+----+----+----+----+----+
         |
   ReportGenerator  or  --json export
         |
   report.html      or  results.json
```

Each stage reads from `PipelineState.results` and writes back to it. If a stage fails, the pipeline continues — downstream stages degrade gracefully.

## Installation Options

```bash
# Core (no PyTorch; DVAE and survival stages skipped gracefully)
pip install -e .

# With LightGBM + SHAP feature importance
pip install -e ".[ml]"

# With survival analysis (Kaplan-Meier, Cox PH)
pip install -e ".[survival]"

# With DVAE autoencoder (requires PyTorch)
pip install -e ".[autoencoder]"

# Full stack (ml + survival + autoencoder)
pip install -e ".[ml,survival,autoencoder]"

# Development (all extras + pytest, ruff)
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.11, pandas, scikit-learn, scipy, numpy, jinja2, pyyaml.

## Project Structure

```
explot/
  cli.py              # Entry point
  orchestrator.py     # Stage runner
  state.py            # PipelineState dataclass
  export.py           # JSON serialization
  stages/
    manifest.yaml     # Stage order and dependencies
    profiling/        # Stage 1
    exploration/      # Stage 2
    preprocessing/    # Stage 3
    dimensionality/   # Stage 4
    autoencoder/      # Stage 5 (optional, needs torch)
    unsupervised/     # Stage 6
    supervised/       # Stage 7
    survival/         # Stage 8 (optional, needs lifelines)
    findings/         # Stage 9
  report/
    generator.py      # HTML report builder
simulator/            # Synthetic data generators for testing
tests/                # 180+ tests against planted ground truth
```

## What This Is Not

- Not AutoML — it doesn't tune hyperparameters or deploy models
- Not a replacement for domain expertise — it's a first pass, not a final answer
- Not optimized for large scale — designed for exploratory use; sampling kicks in above ~8k rows to keep runtimes reasonable
- Interpretations are heuristic-based and labeled as such

## License

MIT
