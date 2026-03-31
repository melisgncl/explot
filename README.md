# Explot

Automated first-pass analyst for tabular data — one CSV in, one trust-aware report out.

Most profiling tools stop at column summaries. Explot goes further: it profiles your data, checks for structure, runs unsupervised and supervised probes, **recommends which ML model fits**, and flags potential leakage — all in a single command.

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

Optional: install with DVAE autoencoder support (requires PyTorch):
```bash
pip install -e ".[autoencoder]"
```

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
git clone <your-repo-url>
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

Explot runs 7 stages, each building on the last:

| Stage | What it does |
|-------|-------------|
| **Profiling** | Column types, missing values, suspicious columns, quality score (0-100) |
| **Exploration** | Redundant feature pairs, cluster tendency (Hopkins), missingness patterns, outliers |
| **Dimensionality** | PCA variance decomposition, intrinsic dimensionality estimate, scree plot |
| **DVAE** | Nonlinear compression via denoising VAE, reconstruction error, latent space view |
| **Unsupervised** | KMeans sweep, DBSCAN auto-tuning, Isolation Forest anomaly detection |
| **Model Selection** | Auto-detects targets, runs cross-validated probes, compares models, recommends the best |
| **Findings** | Cross-stage synthesis with confidence levels and suggested next steps |

The output is a self-contained HTML file with tabbed navigation — no CDN dependencies, opens in any browser.

## Trust Flags

The model selection stage doesn't just report scores — it warns you when scores shouldn't be trusted.

| Flag | What it means |
|------|--------------|
| `single_feature_leakage` | One feature alone predicts the target nearly as well as the full model |
| `exact_copy_feature` | A feature is an exact copy of the target column |
| `proxy_like_feature` | A feature deterministically maps to the target (1-to-1) |
| `high_correlation_proxy` | A feature has |r| > 0.995 with the target |
| `near_perfect_score` | F1 > 0.95 — investigate before celebrating |
| `suspicious_feature_name` | Feature name shares tokens with target name |

## How to Read the Results

**3-minute read:**
1. Open the **Overview** tab — check quality score and top 3 findings
2. Jump to **Findings** — HIGH-confidence items are actionable, LOW is noise
3. Check **Model Selection** — look at trust flags first, scores second

**Deep read:**
- **Profiling** — verify suspicious columns were caught (especially ID columns)
- **Exploration** — redundant pairs tell you what to drop; Hopkins < 0.5 means clustering won't help
- **Dimensionality** — intrinsic dim estimate reveals how many features actually matter
- **Model Selection** — Track A (PCA features) vs Track B (DVAE latent) comparison shows whether nonlinear structure exists

## Example Output

From the sleep health dataset (`data/sleep_health_dataset.csv`):

```
HIGH: Dataset quality score is 87/100 (good).
HIGH: Target 'exercise_day' is highly predictable: LogisticRegression F1=0.94.
      Trust flags: possible_leakage, single_feature_leakage
MEDIUM: Moderate cluster tendency (Hopkins=0.48).
LOW: Weak cluster structure: k=2, silhouette=0.19.

Suggested next steps:
- Review leakage risk for exercise_day before deployment
- Consider removing redundant features before downstream modeling
```

## Architecture

```
CSV / TSV / Excel / Parquet
         |
    load_table()
         |
   PipelineState
         |
    +----+----+----+----+----+----+----+
    | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
    |Prof|Expl|Dim |DVAE|Uns |Sup |Find|
    +----+----+----+----+----+----+----+
         |
   ReportGenerator  or  --json export
         |
   report.html      or  results.json
```

Each stage reads from `PipelineState.results` and writes back to it. If a stage fails, the pipeline continues — downstream stages degrade gracefully.

## Installation Options

```bash
# Core (no PyTorch, DVAE stage skipped gracefully)
pip install -e .

# With DVAE autoencoder
pip install -e ".[autoencoder]"

# Development (includes pytest, ruff, torch)
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
    dimensionality/   # Stage 3
    autoencoder/      # Stage 4 (optional, needs torch)
    unsupervised/     # Stage 5
    supervised/       # Stage 6
    findings/         # Stage 7
  report/
    generator.py      # HTML report builder
simulator/            # Synthetic data generators for testing
tests/                # 58+ tests against planted ground truth
```

## What This Is Not

- Not AutoML — it doesn't tune hyperparameters or deploy models
- Not a replacement for domain expertise — it's a first pass, not a final answer
- Not optimized for large scale — works well up to ~10k rows, sampling kicks in above that
- Interpretations are heuristic-based and labeled as such

## License

MIT
