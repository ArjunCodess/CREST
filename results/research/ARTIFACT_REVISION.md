# Research pilot artifacts — revision notice

**Status (2026-11-11 boundary pass):** Development and measured pilots regenerated per `results/research/build_verification.json`. v3 final holdout **`retired_failed`** on ledger and manifest exposure record `configs/research/final_holdout_v3_exposure_record.json` (reported full 10×10 matrix admission by implementing agent; independently observed subset empty). Frozen failed attempt preserved (`research_code_hash` **07a399f8e9550149**, `results/research_final_v3/final_attempt_failure.json`). Generative public API domain validation: `docs/research_simulator_domain.md`.

**Analysis correction (2026-11-11, derived artifacts only):** Post-hoc fix to `scripts/analyze_final_holdout_v4.py` (bootstrap multiplicity, reconciliation tables, OD NNLS erratum). Frozen v4 pilot JSON / receipt / registry byte-unchanged. Detail: `docs/research_analysis_erratum.md`, archive `results/research_final_v4/archive/pre_posthoc_correction_20261111T173220Z/`.

**Generative amendment (this pass only):**
- Strict typed domain at `generate_scenario` / protocol JSON boundary (no fractional `int()` truncation; reject bool/string/nonfinite counts).
- Feasible supply-capped chain integrator replaces independent flux draw + post-integration capacity assert.
- Estimator / LP / baseline modules unchanged (see `build_verification.json` `estimator_lp_content_sha256_prefix`).
- No v3 rerun, no v4 reserve, no `--override-final`.

**Supersedes (historical, not erased):**
- Legacy independent-flux generator (v3 final failure root cause under `n_links=7`).
- Shared single-`u` boundary model (constant corridor vehicle sum).
- Gaussian storage noise with hard `2.5σ` tubes used as a silent baseline (joint misspecification; seed 55 truth-tube violation ~0.88 veh).
- OD aggregate LPs without declared `path_flow_cap` witnesses (`unbounded` mis-labeled as rank-deficiency infeasibility).
- Measured pilot artifacts predating independent inlet/outlet `R[:,0], R[:,1]`.

**Within-assumption baseline contract:**
- `obs_noise_model=uniform_bounded`, `obs_tolerance=obs_noise_scale`.
- Count intervals and path caps constructed so generative truth is a feasible witness on dev seeds `11..55`.
- Named stresses: `gaussian_misspecification`, `inconsistent_od_counts`, `intentional_boundary_inconsistency`.

**Holdout:**
- v1 seeds 101–505 retired (prior exposure).
- v2 seeds 701, 802, 903, 1004, 1105 **retired** (subset 701, 802 exposed via `run_one_scenario`).
- v3 seeds in `final_holdout_v3_seed_manifest.json` **archived** after single failed final attempt; `FINAL_HOLDOUT_ACTIVE_SEEDS` empty.

Regenerate development synthetic pilot:

```powershell
cd crest
python -m src.research.runner --mode development --config configs/research/dev_pilot.json --output results/research --force
```

Measured pilot:

```powershell
python -m src.research.measured_runner --config configs/research/measured_pilot.json --output results/research_measured
```
