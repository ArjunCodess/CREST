"""Regression tests for measured trajectory adapter (synthetic fixtures; no required downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.trajectory_adapter import (
    NormalizedTrajectory,
    UnsupportedCoordinateSystem,
    aggregate_cell_edie,
    clip_segment_to_cell,
    crossing_time_at_x,
    load_ngsim_i80_csv,
    remove_incomplete_edge_frames,
    segment_crosses_x,
    load_i24_motion_json,
    exclusive_vehicle_cell_assignment,
    assign_non_overlapping_spatial_cells,
    storage_identity_audit,
    _ngsim_speed_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NGSIM = REPO_ROOT / "data" / "research_external" / "ngsim_i80_slice.csv"
I24 = REPO_ROOT / "data" / "research_external" / "i24_motion_sample_3.json"


def _traj_const_speed(*, x0_m: float, v_mps: float, t0_ms: int, n_pts: int, dt_ms: int) -> NormalizedTrajectory:
    t = np.arange(n_pts, dtype=np.int64) * dt_ms + t0_ms
    x = x0_m + v_mps * (t - t0_ms) / 1000.0
    v = np.full(n_pts, v_mps)
    return NormalizedTrajectory(vehicle_key="syn:1", t_ms=t, x_m=x, v_mps=v)


def test_clip_segment_uniform_motion_exact_edie() -> None:
    """One vehicle at 10 m/s crossing a 100m x 10s cell: Edie q and snapshot counts."""
    v = 10.0
    tr = _traj_const_speed(x0_m=0.0, v_mps=v, t0_ms=0, n_pts=101, dt_ms=100)
    # Exit at t=10s must fall strictly inside the window (half-open [t_start, t_end)).
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=10500,
        x_start_m=0.0,
        x_end_m=100.0,
        dt_sample_ms=100,
    )
    assert cell.snapshot_count_start == 1
    dt_hours = 10.5 / 3600.0
    assert cell.crossing_flux_out_vph == pytest.approx(1.0 / dt_hours, rel=0.05)
    assert cell.edie_travel_time_s == pytest.approx(10.0, rel=0.05)
    assert cell.edie_distance_m == pytest.approx(100.0, rel=0.05)


def test_crossing_time_and_units() -> None:
    t0, x0, t1, x1 = 0.0, 0.0, 1000.0, 10.0
    assert segment_crosses_x(t0, x0, t1, x1, 5.0)
    ct = crossing_time_at_x(t0, x0, t1, x1, 5.0)
    assert ct == pytest.approx(500.0)
    clip = clip_segment_to_cell(t0, x0, t1, x1, t_start_ms=0, t_end_ms=2000, x_start_m=0, x_end_m=20)
    assert clip is not None


def test_truncated_edge_frame_removal_sparse_first_only() -> None:
    df = pd.DataFrame(
        {
            "global_time": [100, 100, 100, 200, 200],
            "vehicle_id": [1, 2, 3, 1, 2],
            "local_y": [10.0, 20, 30, 11, 21],
            "v_vel": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out, flags = remove_incomplete_edge_frames(df)
    assert flags["dropped_rows"] >= 0
    assert len(out) <= len(df)


def test_interior_density_peak_not_dropped() -> None:
    """Legitimate interior occupancy peak must not be removed (regression on argmax heuristic)."""
    times = [1000] * 2 + [2000] * 50 + [3000] * 3
    df = pd.DataFrame(
        {
            "global_time": times,
            "vehicle_id": list(range(len(times))),
            "local_y": np.linspace(1, 100, len(times)),
            "v_vel": 1.0,
        }
    )
    out, flags = remove_incomplete_edge_frames(df)
    assert 2000 in set(out["global_time"])
    assert flags["per_frame_count_summary"]["max_count_ms"] == 2000
    assert 2000 not in flags["removed_edge_times_ms"]


def test_partial_last_frame_retained_when_not_sparse() -> None:
    base = [1000] * 40 + [1100] * 5
    df = pd.DataFrame(
        {"global_time": base, "vehicle_id": range(len(base)), "local_y": 1.0, "v_vel": 1.0}
    )
    out, flags = remove_incomplete_edge_frames(df)
    assert 1100 in set(out["global_time"])


def test_edie_metric_units_independent_of_qkv_tautology() -> None:
    """10 m/s × 10 s in 100 m × 10 s cell → known k, v, q in metric units."""
    tr = _traj_const_speed(x0_m=0.0, v_mps=10.0, t0_ms=0, n_pts=101, dt_ms=100)
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=10000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert cell.edie_k_vpkm == pytest.approx(10.0, rel=0.02)
    assert cell.edie_v_kmh == pytest.approx(36.0, rel=0.02)
    assert cell.edie_q_vph == pytest.approx(360.0, rel=0.02)
    assert cell.edie_k_vpmi == pytest.approx(10.0 / 1.609344, rel=0.02)
    assert cell.edie_v_mph == pytest.approx(36.0 / 1.609344, rel=0.02)


def test_missing_speed_is_nan_stationary_is_zero() -> None:
    v_nan, _, _ = _ngsim_speed_row(float("nan"), None)
    assert np.isnan(v_nan)
    v_zero, _, _ = _ngsim_speed_row(0.0, None)
    assert v_zero == 0.0


def test_analytic_conservation_one_vehicle_crosses() -> None:
    """FD storage delta matches crossing flux for uniform motion through one cell."""
    tr = _traj_const_speed(x0_m=-50.0, v_mps=10.0, t0_ms=0, n_pts=120, dt_ms=100)
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=10000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    dt_s = 10.0
    net_cross_veh = (cell.crossing_flux_in_vph - cell.crossing_flux_out_vph) * (dt_s / 3600.0)
    assert cell.fd_storage_delta == pytest.approx(net_cross_veh, abs=0.6)


def test_time_window_carry_in_second_window() -> None:
    tr = _traj_const_speed(x0_m=50.0, v_mps=0.0, t0_ms=0, n_pts=201, dt_ms=100)
    first = aggregate_cell_edie([tr], t_start_ms=0, t_end_ms=10000, x_start_m=0.0, x_end_m=100.0)
    second = aggregate_cell_edie([tr], t_start_ms=10000, t_end_ms=20000, x_start_m=0.0, x_end_m=100.0)
    assert second.track_accounting.time_window_carry_in == 1
    assert second.track_accounting.internal_appearances == 0
    assert first.track_accounting.internal_appearances == 0


def test_fragmentation_counted() -> None:
    t = np.array([0, 100, 200, 800, 900], dtype=np.int64)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.zeros(5)
    tr = NormalizedTrajectory("syn:gap", t, x, v)
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=1000,
        x_start_m=-1.0,
        x_end_m=10.0,
        max_gap_ms=250,
    )
    assert cell.track_accounting.fragmentation_events >= 1


def test_exclusive_cell_assignment_no_overlap() -> None:
    cells = assign_non_overlapping_spatial_cells(0.0, 100.0, 2)
    tr1 = NormalizedTrajectory("syn:a", np.array([0, 1000]), np.array([10.0, 10.0]), np.array([0.0, 0.0]))
    tr2 = NormalizedTrajectory("syn:b", np.array([0, 1000]), np.array([60.0, 60.0]), np.array([0.0, 0.0]))
    assign = exclusive_vehicle_cell_assignment([tr1, tr2], cells, t_mid_ms=500)
    assert assign["syn:a"] == 0
    assert assign["syn:b"] == 1


@pytest.mark.skipif(not NGSIM.exists(), reason="NGSIM slice not on disk")
def test_ngsim_load_smoke_no_temporal_leakage_in_windows() -> None:
    trajectories, prov = load_ngsim_i80_csv(NGSIM)
    assert len(trajectories) > 0
    t_min = min(tr.t_ms[0] for tr in trajectories)
    t_max = max(tr.t_ms[-1] for tr in trajectories)
    assert prov.coverage_flags["duration_s"] < 120.0
    assert t_max > t_min


@pytest.mark.skipif(not I24.exists(), reason="I-24 sample not on disk")
def test_i24_world_position_fail_closed_unverified_units() -> None:
    with pytest.raises(UnsupportedCoordinateSystem):
        load_i24_motion_json(I24)


def test_storage_identity_exact_uniform_crossing() -> None:
    """ΔN equals crossing net when no interior births or deaths."""
    tr = _traj_const_speed(x0_m=-20.0, v_mps=10.0, t0_ms=0, n_pts=80, dt_ms=100)
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=5000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert cell.storage_identity_residual_veh == 0
    assert cell.fd_storage_delta == cell.crossing_count_in - cell.crossing_count_out


def test_storage_identity_interior_birth() -> None:
    """Track begins inside cell without upstream crossing; identity stays closed."""
    t = np.array([2500, 2600, 2700], dtype=np.int64)
    x = np.array([50.0, 50.5, 51.0])
    tr = NormalizedTrajectory("syn:spawn", t, x, np.zeros(3))
    cell = aggregate_cell_edie([tr], t_start_ms=0, t_end_ms=5000, x_start_m=0.0, x_end_m=100.0)
    assert cell.track_accounting.dataset_boundary_births == 1
    assert cell.storage_identity_residual_veh == 0


def test_consecutive_windows_share_snapshot_boundary() -> None:
    tr = _traj_const_speed(x0_m=50.0, v_mps=0.0, t0_ms=0, n_pts=101, dt_ms=100)
    w0 = aggregate_cell_edie([tr], t_start_ms=0, t_end_ms=5000, x_start_m=0.0, x_end_m=100.0)
    w1 = aggregate_cell_edie([tr], t_start_ms=5000, t_end_ms=10000, x_start_m=0.0, x_end_m=100.0)
    assert w0.snapshot_count_end == w1.snapshot_count_start
    audit = storage_identity_audit([w0, w1])
    assert audit["max_abs_residual_veh"] == 0


def test_nonconstant_flow_two_vehicles() -> None:
    tr1 = _traj_const_speed(x0_m=-10.0, v_mps=10.0, t0_ms=0, n_pts=60, dt_ms=100)
    tr2 = _traj_const_speed(x0_m=-10.0, v_mps=10.0, t0_ms=2000, n_pts=60, dt_ms=100)
    tr2.vehicle_key = "syn:2"
    cell = aggregate_cell_edie(
        [tr1, tr2],
        t_start_ms=0,
        t_end_ms=10000,
        x_start_m=0.0,
        x_end_m=100.0,
    )
    assert cell.storage_identity_residual_veh == 0
    assert cell.crossing_count_in == 2


def test_tiny_synthetic_csv_fixture_without_download(tmp_path: Path) -> None:
    """Adapter path on minimal in-memory CSV in pytest tmp_path."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "global_time": 1000 + i * 100,
                "vehicle_id": 1,
                "local_y": 50.0 + 0.1 * i,
                "v_vel": 33.0,
                "lane_id": 3,
            }
        )
    csv_path = tmp_path / "ngsim_tiny.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    tr, _ = load_ngsim_i80_csv(csv_path, expected_sha256=None)
    assert len(tr) == 1
    assert tr[0].t_ms[-1] - tr[0].t_ms[0] == 1900
