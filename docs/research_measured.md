# Measured-Data Development Pilot (CREST)

Development-only path from public trajectory feeds to macro link observations compatible with the boundary conservation estimator. **Not** for final protocol / holdout reporting.

## Scope and limitations

- **NGSIM I-80 slice**: 25,000-row SODA export (`$limit=25000`); **edge-only** trim (sparse first/last timestamp at slice boundaries). Interior density peaks are **not** dropped. See `coverage_flags.drop_rationale` / `retained_time_range_ms`.
- **Count identity (sampled tracks)**: per cell–window bookkeeping identity with births/deaths (`storage_count_identity`); **not** proof of real-world mass conservation.
- **Edie** outputs use **metric** corridor fields: `edie_q_vph` (all-lane), `edie_k_vpkm`, `edie_v_kmh` (+ optional `edie_k_vpmi` / `edie_v_mph`). \(q = k \cdot v\) with those units is **definitional**, not FD validation.
- **Inference**: storage observations use **T+1** boundary snapshots; flux row `k` applies to `[t_k, t_{k+1})`. Crossing flux and boundary demand are **all-lane corridor** vph (every lane at a spatial cut). Declared **per-lane** saturation in config is converted to `all_lane_capacity_vph = mainline_lane_count × per_lane_capacity_vph` for the development **physical_envelope** scenario. A separate **prefix_empirical** scenario calibrates `z_upper` from the first `calibration_prefix_steps` windows only (can under-cap later peaks — not eval leakage). `flow_assumption_scenarios` records an LP feasibility witness per scenario before SLSQP; infeasible LPs skip the optimizer (`skipped_lp_infeasible`).
- **Boundary model**: independent inlet/outlet decision variables (`n_bound=2`); prefix calibration sets separate intervals from `flux_in[:,0]` and `flux_out[:,-1]`. `empirical_boundary_flux_totals` and `boundary_aggregate_lp_*` report inflow, outflow, and net vehicles. Prior artifacts that used a single shared `u` (constant corridor sum) are superseded.
- **Hard fits may remain infeasible** when observed snapshots are inconsistent with the discrete chain even after physically scoped caps; see `failure_category`, `feasibility_lp_witness_*`, and `lp_infeasibility_diagnosis`. Do not treat temporal carry-in/out as hidden physical sources.
- **Development pilot (3 cells × 4 snapshots, 3 flux intervals)**: `physical_envelope` uses declared all-lane cap (15000 vph for six lanes). With `calibration_prefix_steps` below the window count, **prefix_empirical** falls back to the same physical floor — scenarios may be **identical** (see `flow_assumption_scenarios`).
- **Debug metrics** (`obs_tolerance_veh`, conditional RMSE when LP is feasible): development-only; not independent measured validation.
- **OD / route truth**: not evaluated on these linear corridors.
- **I-24 sample**: probed when present; pixel-only schemas are skipped with an explicit reason.

## Adapter contract

Implemented in [`src/research/trajectory_adapter.py`](../src/research/trajectory_adapter.py):

| Output | Meaning |
|--------|---------|
| Longitudinal position | meters (`local_y` feet → m for NGSIM). I-24 open sample: **not loaded** (unverified `world_position` basis). |
| Time | milliseconds since epoch (NGSIM) or frame-derived ms (I-24) |
| `MacroCellAggregate` | Edie totals, snapshot start/end counts, crossing flux in/out, track accounting |
| `AdapterProvenance` | raw SHA-256, `adapter_source_hash`, coverage flags |

## Running the pilot

```powershell
cd crest
python -m src.research.measured_runner --config configs/research/measured_pilot.json --output results/research_measured
```

Raw data are **not** imported via `pip install`; paths are passed explicitly. Tests use tiny synthetic fixtures and optionally skip when `data/research_external/` is absent.

## Artifacts

| File | Contents |
|------|----------|
| `results/research_measured/measured_pilot_results.json` | Provenance, calibration status, inference attempts, failure rates, residuals |
| `results/research_measured/macro_cell_aggregates.csv` | Per cell-space-time Edie and flux metrics |

## Configuration

[`configs/research/measured_pilot.json`](../configs/research/measured_pilot.json): spatial cells, `dt_ms`, `mainline_lane_count`, `per_lane_capacity_vph`, `flow_flux_scope`, mask seed, calibration prefix length, SHA-256 for NGSIM.

## Tests

```powershell
python -m pytest tests/test_research_trajectory_adapter.py tests/test_research_measured_integration.py tests/test_research_data_sources.py -q
```
