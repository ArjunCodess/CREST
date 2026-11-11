# Research formulation: partial identification and boundary conservation

This document describes the V1 research stack under `src/research/`. It is separate from the competition training and submission pipeline in `src/pipeline.py`.

## Scope and limits

- **Static count window:** path flows are inferred from link counts and priors at a single departure interval per scenario. No dynamic OD assignment or time-varying path shares are claimed.
- **Synthetic truth only:** the pilot simulator generates known boundary and path truth on small chain networks. It does not fabricate real TrafficFlowBench rows.
- **Conservation is discrete balance:** storage updates follow `N[t+1] = N[t] + dt (B z[t] + R u[t])` with vehicles and vph. Chain networks use **two nonnegative boundary channels**: inlet `u[:,0]` on `R[0,0]=1` and outlet magnitude `u[:,1]` on `R[-1,1]=-1`. LP aggregates report cumulative inflow, outflow, and net `dt*(u_in-u_out)` separately (not a sum of all positive channels). The retired single-column `R` fixture that forced constant corridor vehicle sum is explicit as `_chain_operators_legacy_shared_inlet_outlet` only.
- **Generative truth** uses start-of-step send/receive flux caps (see `docs/research_simulator_domain.md`); not a full CTM/LWR solve and `q = k v` is not used as independent physics evidence.
- **Bounds are conditional:** interval and LP outputs are valid only under stated intervals, priors, and the discrete model class.

## Module 1: partial identification (`partial_identification.py`)

Given nonnegative path flows `f` and incidence `A` (link loads `A f`):

- Observed links impose **interval counts** `l ≤ A f ≤ u` (equalities when `l = u`).
- Optional per-path caps tighten the feasible set.
- **Row-space diagnostics:** SVD rank of the observed incidence rows, with scale-aware tolerance, distinguishes structural under-determination from tight intervals.
- **Aggregate bounds:** for linear functional `w' f`, HI GHS `linprog` returns min/max or explicit `infeasible` / `unbounded` / `numerical_failure` without substituting midpoints.

Nonnegativity can identify aggregates (e.g. total flow on a shared link) even when nullity is positive.

## Module 2: boundary conservation (`boundary_conservation.py`)

- **Deployable default (hard polytope):** `baselines.bounded_boundary_estimate` calls `fit_boundary_conservation` with `w_obs=0` and `use_hard_observation_constraints=True`. Masked storage enters as **linear inequalities** (same `assemble_conservation_polytope` as LP extrema), not as a squared-loss objective. Boundary flows are pulled toward `u_prior` via `w_prior` only inside the feasible set.
- **Optional soft fit:** set `w_obs>0` to add a diagnostic observation misfit term; hard constraints remain unless `use_hard_observation_constraints=False` (not used in the pilot matrix).
- **LP extrema:** min/max of linear boundary aggregates over that polytope (box, capacities, observation tubes). Distinct from the point-estimate QP layer.
- **Stationarity field:** box-projected gradient norm — a smoothness diagnostic, not a KKT certificate with active linear inequalities.
- **Failure-aware summaries:** conditional means use rows with `bounded_status=optimal` and small primal violation; unconditional failure rates are always reported separately.

**Generative contract (development baseline):** storage observation noise is **uniform** on `[-obs_noise_scale, obs_noise_scale]` with `obs_tolerance = obs_noise_scale` (shared tube). We do **not** claim nominal joint-Gaussian coverage. The stress variant `gaussian_misspecification` keeps Gaussian draws with hard `2.5σ`-style tubes to document joint misspecification honestly.

When intervals exclude generative truth, coverage flags fail without repair.

## Baselines (`baselines.py`, `experiments.py`)

| Method | Role |
|--------|------|
| Masked mean / forward fill | Unconstrained imputers |
| Bounded boundary fit | Deployable estimator |
| Oracle exact boundary | Diagnostic only (uses generative truth) |
| Interval box on `u` | Honest wide baseline |
| OD NNLS + LP aggregate bounds | Links partial-identification stack |

Metrics are per truth: state RMSE, boundary flow RMSE, true discrete conservation residual, bound coverage/width, OD total error, runtime. No competition composite proxy.

## Protocol (`protocol.py`, `configs/research/`)

- **Development:** `configs/research/dev_pilot.json`, seeds `11–55`, default CLI mode.
- **Final:** `configs/research/final_protocol.json`, held-out seeds `101–505`; `--mode final` writes `results/research/final_receipt.json` and refuses repeat fingerprints unless `--override-final` with reason.
- Real-data adapters need an explicit embargo note before treating calendar weeks as fresh evaluation windows.

## Reproduction

```powershell
cd crest
python -m pytest tests/test_research_*.py
python -m src.research.runner --config configs/research/dev_pilot.json --output results/research
```
