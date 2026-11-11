# Research pilot results (synthetic V1)

**Scope:** Independent synthetic RNG scenarios on chain networks only. Not real TrafficFlowBench generalization, not competition leaderboard evidence, not conference acceptance.

**Code hash (frozen failed v3 final):** `07a399f8e9550149`. **Post-generative-repair baseline (dev/measured, pre-v4 activation):** `6959a7802af6991e`. **Pre-v4-execution activation hash:** `c15d37e54fe38d63` (v4 protocol/manifest/constants only; estimator/LP module digests unchanged vs `results/research/build_verification.json`).

## Commands

```powershell
cd crest
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q
python -m src.research.runner --mode development --config configs/research/dev_pilot.json --output results/research --force
python -m src.research.measured_runner --config configs/research/measured_pilot.json --output results/research_measured
python scripts/fetch_research_data.py --verify-only
python scripts/record_pre_evaluation_freeze_v4.py --pytest-count <N>
python -m src.research.runner --mode final --config configs/research/final_protocol_v4.json --output results/research_final_v4 --registry .crest_research_protocol_registry.json
python scripts/analyze_final_holdout_v4.py
```

Wheel smoke is included in `tests/test_research_wheel.py` (no network; `--no-deps` install).

## Development pilot (complete)

| Item | Value |
|------|--------|
| Matrix | 10 stress variants × 5 seeds (11, 22, 33, 44, 55) |
| Rows | 50 |
| Primary metric (declared) | `bounded_state_rmse_missing` (conditional on valid bounded point estimate) |
| Artifacts | `results/research/pilot_results.json`, `pilot_metrics.csv`, `pilot_state_rmse.png` |

Baselines in matrix: masked mean, forward fill, interval aggregate midpoint, OD NNLS, OD LP bounds, oracle exact boundary (diagnostic only). Strong baselines / GNN / SOTA comparisons are **not** claimed.

## Measured development pilot (complete)

NGSIM I-80 slice and I-24 sample used for **development feasibility only** (65 s cadence cohort; OD truth not measured; I-24 units blocked per protocol). See `docs/research_measured.md`.

## Final holdout v3 (failed — single attempt, archived)

| Item | Value |
|------|--------|
| Protocol | `final_holdout_v3` |
| Fingerprint | `5da3b1d664faf594` |
| Outcome | **Failed** before metrics (generative capacity); **not rerun** |
| Artifacts preserved | `configs/research/final_protocol.json`, `pre_evaluation_freeze.json`, `results/research_final_v3/final_attempt_failure.json` |

## Final holdout v4 (amended reserve — single completed attempt)

| Item | Value |
|------|--------|
| Protocol | `final_holdout_v4` (amended after generative repair; not original v3 preregistration) |
| Fingerprint | `0040a9f70180ada0` |
| Seeds | 10 (`configs/research/final_holdout_v4_seed_manifest.json`, written before execution) |
| Matrix | 10 stress variants × 10 seeds = **100 scenarios completed** |
| Pre-freeze | `results/research/pre_evaluation_freeze_v4.json` (does not overwrite v3 freeze) |
| Attempt | Registry `started_utc` `2026-11-11T11:48:42Z` → `finished_utc` `2026-11-11T11:49:45Z` |
| Receipt | `results/research_final_v4/final_receipt.json` (`status: completed`, no override) |
| Analysis | `results/research_final_v4/final_analysis.json`, `scripts/analyze_final_holdout_v4.py` (hash in analysis receipt) |
| Primary metric | `bounded_state_rmse_missing` |

**Matrix-level outcomes (honest denominators):**

- 100/100 scenarios generated and scored; 0 generative capacity failures under repaired integrator.
- 81/100 rows with deployable bounded point estimate (`optimal`, primal ≤ 1e-5).
- 19/100 rows infeasible polytope (by design under `truth_outside_interval`, `intentional_boundary_inconsistency`, and one `gaussian_misspecification` cell).
- `truth_outside_interval`: **0/10** valid bounded point estimates (primary metric denominator 0).
- `intentional_boundary_inconsistency`: **2/10** valid bounded point estimates.

**Conditional paired comparisons** (shared valid subset only, **n=81**; recomputed from frozen JSON): masked-mean missing RMSE **6.6591** vs bounded deployable **32.4108** vs forward fill **20.6045** vs interval midpoint **66.1359** (paired bounded−masked mean **25.7517**). Oracle missing RMSE = 0.0 is **diagnostic only**, not a deployable win. **Negative result:** bounded deployable does not beat masked-mean on this metric (no SOTA claim).

**Uncertainty (corrected cluster bootstrap, post-hoc):** `resampling_seed=424242`, **10** independent seed clusters, **10 000** replicates with multiplicity preserved → sample primary missing RMSE **32.4108**, bootstrap mean of replicate means ≈ **32.38**, percentile 95% CI ≈ **[27.85, 37.88]** (precision-limited; see `docs/research_analysis_erratum.md` and prior archived analysis in `results/research_final_v4/archive/`).

**Denominators (do not conflate):**

- Matrix rows: **100**; bounded point-valid: **81**; infeasible polytope: **19** (= 10 `truth_outside_interval` + 8 `intentional_boundary_inconsistency` + 1 `gaussian_misspecification`).
- Rows with generative boundary truth inside stated interval: **80** (= 8 variants × 10 seeds, excluding intentional + truth-outside stresses).
- `boundary_interval_covers_truth_net_vehicles`: **80/100** eligible booleans.
- `lp_data_covers_truth_net_vehicles`: **79/81** on bounded-valid rows (LP metric absent when infeasible).
- Feasible bounded optimum **≠** truth-in-polytope; LP feasibility can admit alternatives that exclude the true trajectory.

**OD NNLS:** `od_nnls_total_error` in frozen JSON is **not** a valid comparative benchmark (masking bug in frozen runner). See erratum; raw values remain historical only.

## Retired holdouts

- **v1:** seeds 101–505, retired after review exposure in development mode.
- **v2:** seeds 701, 802, 903, 1004, 1105; subset 701/802 executed despite retirement.
- **v3:** failed single attempt; exposure record on disk; **no v3 execution in this pass**.

## Limitations

- v4 is an **amended** protocol after an archived v3 generative failure; it is not a blind repetition of the original preregistered v3 reserve.
- Registry ledger is a cooperative same-workspace guard, not a cross-host security boundary.
- Infeasible bounded LPs are negative identification outcomes, not successful predictions.
- Measured pilot remains development-only; no OD ground truth on NGSIM slice.

## Verification

Integration receipt: `results/research/build_verification.json` (updated after v4 execution pass).
