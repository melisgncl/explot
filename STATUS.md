## Explot Status

Current phase: real-data hardening and runtime scaling.

Implemented and tested:
- Profiling
- Exploration
- Dimensionality
- DVAE compression
- Unsupervised probes
- Supervised probes
- Findings
- Multi-tab HTML report

Current runtime behavior:
- Small and medium datasets run end to end.
- Large datasets now use deterministic sampling in unsupervised and supervised stages.
- Reports now include PCA/SVD visuals, DVAE output, model trust notes, Track A vs Track B comparison, confusion matrices, and precision/recall/F1 summaries.
- Full test suite is green.

Current control files:
- `config/default.yaml`: enabled stages and budget defaults
- `explot/stages/manifest.yaml`: pipeline order
- `RISKS.md`: scope and interpretation guardrails
- `STATUS.md`: current implementation status

Next focus:
- Add stronger proxy/leakage heuristics and clearer trust badges
- Expand exploration and simulator coverage toward the original design
- Compare DVAE anomaly signals more directly against Isolation Forest and Stage 2 outliers
