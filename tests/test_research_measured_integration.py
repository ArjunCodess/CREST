"""Adapter → boundary conservation integration (synthetic fixtures only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.research.baselines import masked_mean_imputer
from src.research.boundary_conservation import fit_boundary_conservation
from src.research.measured_runner import (
    _aggregate_table,
    _csv_value,
    _json_safe,
    _write_outputs,
    all_lane_vph_from_per_lane,
    calibrate_boundary_bounds,
    calibrate_internal_flow_bounds,
    physical_internal_flow_bounds,
    resolve_flow_capacity_metadata,
)
from src.research.synthetic_sim import _chain_operators
from src.research.trajectory_adapter import NormalizedTrajectory, aggregate_cell_edie, macro_grid_from_trajectories


def _crossing_traj(
    *,
    vehicle_key: str,
    x0_m: float,
    v_mps: float,
    t0_ms: int,
    t1_ms: int,
    dt_ms: int = 100,
) -> NormalizedTrajectory:
    n = max(2, int((t1_ms - t0_ms) / dt_ms) + 1)
    t = np.linspace(t0_ms, t1_ms, n, dtype=np.int64)
    x = x0_m + v_mps * (t - t0_ms) / 1000.0
    v = np.full(n, v_mps)
    return NormalizedTrajectory(vehicle_key=vehicle_key, t_ms=t, x_m=x, v_mps=v)


def test_measured_grid_to_estimator_shapes_compatible() -> None:
    """Known crossings produce storage/flux arrays compatible with chain operators."""
    tr = _crossing_traj(x0_m=0.0, v_mps=12.0, t0_ms=0, t1_ms=10000, vehicle_key="syn:1")
    aggregates, meta = macro_grid_from_trajectories(
        [tr],
        n_spatial_cells=2,
        dt_ms=5000,
        x_min_m=0.0,
        x_max_m=200.0,
    )
    assert meta["n_time_windows"] >= 2
    storage, flux_in, flux_out, _cross = _aggregate_table(aggregates, 2)
    Tm1, L = storage.shape[0] - 1, storage.shape[1]
    assert L == 2
    assert storage.shape[0] >= 2
    B, R = _chain_operators(L)
    dt_hours = 5000 / 1000.0 / 3600.0
    n0 = np.nan_to_num(storage[0], nan=0.0)
    u_lo, u_hi, _ = calibrate_boundary_bounds(flux_in, flux_out, calib_steps=1, fallback_lower=0, fallback_upper=5000)
    u_prior = 0.5 * (u_lo + u_hi)
    mask = np.isfinite(storage)
    est = fit_boundary_conservation(
        B,
        R,
        dt_hours,
        n0,
        mask,
        np.nan_to_num(storage, nan=0.0),
        u_prior,
        u_lo,
        u_hi,
        w_obs=0.0,
        z_upper=5000.0,
        obs_tolerance=2.0,
        use_hard_observation_constraints=True,
    )
    assert est.storage.shape == storage.shape
    assert est.boundary_flow.shape[0] == max(Tm1, 0)


def test_track_accounting_carry_in_not_spurious_birth() -> None:
    """Vehicle present at window open is carry-in, not internal appearance."""
    tr = _crossing_traj(x0_m=50.0, v_mps=0.0, t0_ms=0, t1_ms=5000, vehicle_key="syn:parked")
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=5000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert cell.track_accounting.time_window_carry_in == 1
    assert cell.track_accounting.internal_appearances == 0


def test_upstream_entry_not_counted_internal() -> None:
    v = 20.0
    tr = _crossing_traj(x0_m=-30.0, v_mps=v, t0_ms=0, t1_ms=5000, vehicle_key="syn:enter")
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=5000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert cell.track_accounting.internal_appearances == 0
    assert cell.crossing_flux_in_vph > 0


def test_masked_baseline_same_mask_offline() -> None:
    storage = np.array([[2.0, 1.0], [3.0, 2.0], [3.0, 1.0]])
    mask = np.ones_like(storage, dtype=bool)
    mask[1, 0] = False
    imputed = masked_mean_imputer(np.where(mask, storage, np.nan), mask)
    assert np.isfinite(imputed[1, 0])


def test_all_lane_cap_converts_from_per_lane_metadata() -> None:
    meta = resolve_flow_capacity_metadata(
        {"mainline_lane_count": 6, "per_lane_capacity_vph": 2500.0, "x_min_m": 0.0, "x_max_m": 100.0}
    )
    assert meta["flow_flux_scope"] == "all_lanes_corridor_cross_section"
    assert meta["all_lane_capacity_vph"] == all_lane_vph_from_per_lane(2500.0, 6)
    assert meta["all_lane_capacity_vph"] == 15000.0


def test_prefix_calibration_floor_is_all_lane_not_per_lane_singleton() -> None:
    """Prefix scheme must not silently treat 2500 vph as corridor total when lanes>1."""
    flux_out = np.zeros((4, 3))
    flux_in = np.zeros((4, 3))
    _, _, diag = calibrate_internal_flow_bounds(
        flux_out,
        flux_in,
        calib_steps=2,
        all_lane_fallback_vph=15000.0,
    )
    assert diag["applied_all_lane_floor_vph"] == 15000.0
    z, _status, _diag2 = calibrate_internal_flow_bounds(
        flux_out,
        flux_in,
        calib_steps=2,
        all_lane_fallback_vph=15000.0,
    )
    assert float(np.max(z)) == 15000.0


def test_physical_envelope_internal_bounds_match_lane_product() -> None:
    z, status = physical_internal_flow_bounds(3, 2, all_lane_capacity_vph=12000.0)
    assert status == "declared_physical_envelope_all_lanes"
    assert np.all(z == 12000.0)


def test_empty_edie_json_null_csv_blank_vs_occupied_stationary(tmp_path: Path) -> None:
    """Empty cells serialize as JSON null / CSV blank; occupied stationary has k>0 and v=0."""
    empty = aggregate_cell_edie(
        [],
        t_start_ms=0,
        t_end_ms=10_000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert np.isnan(empty.edie_q_vph)
    assert np.isnan(empty.edie_k_vpkm)
    assert np.isnan(empty.edie_v_kmh)

    t = np.array([0, 1000, 2000], dtype=np.int64)
    x = np.array([50.0, 50.0, 50.0])
    v = np.zeros(3)
    parked = NormalizedTrajectory("syn:parked", t, x, v)
    occupied = aggregate_cell_edie(
        [parked],
        t_start_ms=0,
        t_end_ms=10_000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert occupied.edie_k_vpkm > 0
    assert occupied.edie_v_kmh == 0.0
    assert occupied.edie_q_vph == 0.0

    for field in ("edie_q_vph", "edie_k_vpkm", "edie_v_kmh"):
        assert _json_safe(getattr(empty, field)) is None
        assert _csv_value(getattr(empty, field)) == ""
    for field in ("edie_q_vph", "edie_k_vpkm", "edie_v_kmh"):
        assert _json_safe(getattr(occupied, field)) is not None
        assert _csv_value(getattr(occupied, field)) != ""

    _write_outputs({"serialization_probe": True}, [empty, occupied], tmp_path)
    doc = json.loads((tmp_path / "measured_pilot_results.json").read_text(encoding="utf-8"))
    assert doc["serialization_probe"] is True
    with (tmp_path / "macro_cell_aggregates.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["edie_q_vph"] == ""
    assert float(rows[1]["edie_k_vpkm"]) > 0
    assert float(rows[1]["edie_v_kmh"]) == 0.0
