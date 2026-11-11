# Final holdout v4 — analysis erratum (post-execution)

**Frozen inference / runner hash (unchanged):** `c15d37e54fe38d63`

**Immutable execution artifacts** (`pilot_results.json`, `final_receipt.json`, `pilot_metrics.csv`, `pre_evaluation_freeze_v4.json`, protocol registry) were **not** modified in this correction pass.

## What was wrong

1. **Bootstrap:** `scripts/analyze_final_holdout_v4.py` (pre-correction SHA-256 prefix `e00d4da51c0b965e`) resampled only **10** times, used modulo cluster IDs, and kept a seed row at most once per draw (`seed in picked` set logic), **discarding bootstrap multiplicity**.
2. **Comparisons:** Uncertainty summarized the bounded missing RMSE mean only; **paired** bounded-minus-baseline differences were not bootstrapped.
3. **Coverage tables (external audit):** Some narratives conflated denominators (e.g. 70/69 vs **80** boundary-truth-in-box rows and **79/81** LP-data coverage on bounded-valid cells).
4. **OD NNLS:** Frozen v4 records `od_nnls_total_error` using a **buggy** baseline implementation (`nan_to_num` count center, unmasked incidence rows). **Withdraw** any comparative ranking against OD NNLS from reporting.

## What was fixed (derived only)

- Seeded cluster bootstrap: `resampling_seed=424242`, **10** independent seed clusters, **10 000** replicates, multiplicity-preserving pooling.
- Machine-readable `denominator_reconciliation` with `sum_assertions`.
- Explicit paired diffs and full-matrix vs shared-valid denominators.
- `analysis_receipt.json` with old/new script hashes and raw SHA-256 witnesses.

Archived pre-correction outputs: `results/research_final_v4/archive/pre_posthoc_correction_20261111T173220Z/`.

## OD NNLS — minimum future fix (not applied in v4)

In `run_one_scenario`, replace unmasked OD NNLS with `A[mask], c[mask]` (and consistent count handling). Until then, treat `od_nnls_total_error` as **invalid for benchmarks**; raw JSON values remain historical audit only.

## `wrong_interval` display

Protocol name retained. Generative **truth remains inside** the stated boundary box (interval stress narrows bounds toward truth). Do not describe as “misspecified empty set.”

## Historical v3 file integrity

Retrospective content hashes on archived v3 monoliths **cannot** re-prove per-file immutability without contemporary witnesses; do not claim missing archival evidence exists or fabricate execution timelines.

## Scientific takeaway

On shared-valid cells (**n=81**), deployable bounded missing RMSE **exceeds** masked-mean and forward-fill baselines — a **negative accuracy** result relative to simple imputers. No SOTA or superiority claim is supported.

## Manuscript packaging pointer

`results/research/build_verification.json` records the pre-correction analysis script prefix `e00d4da51c0b965e` at integration time. Manuscript numerics and tables use the corrected analysis chain documented in `results/research_final_v4/analysis_receipt.json` and `research_manuscript/packaging_analysis_receipt.json` (do not retro-edit the historical build receipt).
