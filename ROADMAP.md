# Explot Roadmap To The Original Plan

This roadmap is the shortest sensible path from the current implementation to the original v3 project plan.

## Current State

- Core staged pipeline is implemented.
- Real-data reports generate successfully.
- Track A vs Track B supervised probes now exist.
- Reports include confusion matrices, precision/recall/F1, trust notes, and sampling notes.
- A real DVAE stage is now in place, but its downstream anomaly and comparison logic can still be expanded.

## Highest-Value Next Steps

1. Strengthen leakage and proxy detection so near-perfect targets are labeled more cautiously.
2. Finish Stage 2 exploration features from the original spec:
  group comparisons, near-duplicate detection, richer outlier logic, and scatter views.
3. Expand simulator coverage so every major stage is tested against planted ground truth.
4. Improve findings provenance and confidence rules.
5. Compare DVAE-derived anomaly signals more explicitly against the other stages.

## Phase Plan

### Phase 1: Trust And Evaluation

- Add stronger proxy checks:
  exact-copy detection, near-deterministic mappings, suspiciously perfect-score warnings.
- Surface trust levels in the report:
  `good signal`, `proxy risk`, `leakage risk`, `uncertain`.
- Add per-class summaries where useful for multiclass targets.

### Phase 2: DVAE Refinement

- Extend the current PyTorch DVAE stage with:
  richer tuning controls, optional validation tracking, and direct anomaly comparison outputs.
- Compare DVAE anomalies against Isolation Forest and Stage 2 outliers.

### Phase 3: Exploration Completion

- Add low-cardinality group comparisons.
- Add near-duplicate column detection separate from redundant pairs.
- Add richer outlier explanations and top-variable scatter views.
- Tighten missingness interpretation toward the original spec.

### Phase 4: Simulator And Verification Expansion

- Fill out the missing simulator families and adversarial cases.
- Tie interpretation checks more tightly to simulator metadata.
- Keep the simulator-first verification chain as the standard for new work.

### Phase 5: Product Polish

- Refine the report copy and trust language.
- Add one polished case study per representative dataset type.
- Consider comparison/export features after the trust story is solid.

## Product Guardrail

Explot is strongest when it acts as a trustworthy first-pass analyst for tabular and bioinformatics-style data.

It gets weaker if it tries to be a universal AutoML platform. The project should prioritize:

- interpretability
- trust
- structure discovery
- useful next-step guidance

over a larger model zoo or flashy features.
