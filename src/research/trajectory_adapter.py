"""Measured trajectory normalization, Edie macro aggregation, and track accounting.

All external units are converted once at load (feet→meters, ft/s→m/s, ms time).
Aggregation uses clipped trajectory segments in space–time cells; snapshot counts
and boundary crossing flux are reported separately from Edie (k, q, v).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

FT_TO_M = 0.3048
MPS_TO_KMH = 3.6
MI_TO_KM = 1.609344
K_PER_M_TO_VPKM = 1000.0

# NGSIM micro sample interval (documented 10 Hz).
NGSIM_DT_MS = 100
FLOW_SCOPE_ALL_LANES = "all_lanes_corridor_cross_section"

SPEED_NEGATIVE_POLICY: Literal["nan_derivative", "zero_clip_reported"] = "nan_derivative"
MIN_TRAJECTORY_POINTS = 3
MAX_GAP_MS_DEFAULT = 250  # fragmentation if Δt exceeds this within one vehicle id


@dataclass(frozen=True)
class AdapterProvenance:
    source_id: str
    raw_sha256: str
    adapter_source_hash: str
    units: dict[str, str]
    coverage_flags: dict[str, Any]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedTrajectory:
    vehicle_key: str
    t_ms: np.ndarray
    x_m: np.ndarray
    v_mps: np.ndarray
    lane_id: int | None = None
    negative_speed_count: int = 0
    inferred_speed_count: int = 0
    gap_count: int = 0

    def duration_ms(self) -> int:
        if self.t_ms.size < 2:
            return 0
        return int(self.t_ms[-1] - self.t_ms[0])


@dataclass
class TrackAccounting:
    """Per cell–time-window observed discrepancies; sums over grid are not unique-vehicle counts."""

    track_starts_in_cell: int = 0
    track_ends_in_cell: int = 0
    internal_appearances: int = 0  # spawned inside without upstream boundary crossing in window
    internal_disappearances: int = 0
    time_window_carry_in: int = 0  # in cell at window open (not a birth event)
    time_window_carry_out: int = 0  # still in cell at window close (not a death event)
    fragmentation_events: int = 0
    dataset_boundary_births: int = 0
    dataset_boundary_deaths: int = 0
    negative_speed_samples: int = 0
    stationary_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroCellAggregate:
    t_start_ms: int
    t_end_ms: int
    x_start_m: float
    x_end_m: float
    edie_travel_time_s: float
    edie_distance_m: float
    edie_q_vph: float
    edie_k_vpkm: float
    edie_v_kmh: float
    edie_k_vpmi: float | None
    edie_v_mph: float | None
    snapshot_count_start: int
    snapshot_count_end: int
    snapshot_count_mean: float
    fd_storage_delta: int
    crossing_flux_in_vph: float
    crossing_flux_out_vph: float
    crossing_count_in: int
    crossing_count_out: int
    storage_identity_residual_veh: int
    n_trajectory_points_in_cell: int
    track_accounting: TrackAccounting

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["track_accounting"] = self.track_accounting.to_dict()
        return d


def adapter_module_hash() -> str:
    path = Path(__file__)
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clip_alpha_for_interval(v0: float, v1: float, lo: float, hi: float) -> tuple[float, float] | None:
    """Return [a_lo, a_hi] in [0,1] where v(a)=v0+a*(v1-v0) lies in [lo, hi), or None."""
    if abs(v1 - v0) < 1e-15:
        if lo <= v0 < hi:
            return 0.0, 1.0
        return None
    a_enter = (lo - v0) / (v1 - v0)
    a_exit = (hi - v0) / (v1 - v0)
    a_lo = max(0.0, min(a_enter, a_exit))
    a_hi = min(1.0, max(a_enter, a_exit))
    if a_hi - a_lo <= 1e-12:
        return None
    return a_lo, a_hi


def clip_segment_to_cell(
    t0_ms: float,
    x0_m: float,
    t1_ms: float,
    x1_m: float,
    *,
    t_start_ms: float,
    t_end_ms: float,
    x_start_m: float,
    x_end_m: float,
) -> tuple[float, float] | None:
    """Intersection of segment with [x_start,x_end)×[t_start,t_end) as alpha interval."""
    if t1_ms <= t0_ms:
        return None
    a_t = _clip_alpha_for_interval(t0_ms, t1_ms, t_start_ms, t_end_ms)
    a_x = _clip_alpha_for_interval(x0_m, x1_m, x_start_m, x_end_m)
    if a_t is None or a_x is None:
        return None
    a_lo = max(a_t[0], a_x[0])
    a_hi = min(a_t[1], a_x[1])
    if a_hi - a_lo <= 1e-12:
        return None
    return a_lo, a_hi


def segment_crosses_x(t0_ms: float, x0_m: float, t1_ms: float, x1_m: float, x_line_m: float) -> bool:
    if t1_ms <= t0_ms:
        return False
    return (x0_m < x_line_m <= x1_m) or (x0_m > x_line_m >= x1_m)


def crossing_time_at_x(t0_ms: float, x0_m: float, t1_ms: float, x1_m: float, x_line_m: float) -> float | None:
    if not segment_crosses_x(t0_ms, x0_m, t1_ms, x1_m, x_line_m):
        return None
    if abs(x1_m - x0_m) < 1e-12:
        return None
    frac = (x_line_m - x0_m) / (x1_m - x0_m)
    if frac < 0.0 or frac > 1.0:
        return None
    return t0_ms + frac * (t1_ms - t0_ms)


def position_at_time(tr: NormalizedTrajectory, t_ms: float) -> float | None:
    if tr.t_ms.size == 0:
        return None
    if t_ms < tr.t_ms[0] or t_ms > tr.t_ms[-1]:
        return None
    return float(np.interp(t_ms, tr.t_ms, tr.x_m))


def in_spatial_cell(x_m: float, x_start_m: float, x_end_m: float) -> bool:
    return x_start_m <= x_m < x_end_m


def count_vehicles_in_cell(
    trajectories: Sequence[NormalizedTrajectory],
    *,
    t_ms: float,
    x_start_m: float,
    x_end_m: float,
) -> int:
    """Snapshot occupancy: vehicles whose interpolated position lies in [x_start, x_end) at t_ms."""
    n = 0
    for tr in trajectories:
        x_at = position_at_time(tr, t_ms)
        if x_at is not None and in_spatial_cell(x_at, x_start_m, x_end_m):
            n += 1
    return n


def count_upstream_crossings(
    trajectories: Sequence[NormalizedTrajectory],
    *,
    x_line_m: float,
    t_start_ms: float,
    t_end_ms: float,
    positive_longitudinal_only: bool = True,
) -> int:
    n = 0
    for tr in trajectories:
        for i in range(tr.t_ms.size - 1):
            t0, x0 = float(tr.t_ms[i]), float(tr.x_m[i])
            t1, x1 = float(tr.t_ms[i + 1]), float(tr.x_m[i + 1])
            if positive_longitudinal_only and x1 <= x0:
                continue
            if segment_crosses_x(t0, x0, t1, x1, x_line_m) and x1 > x0:
                ct = crossing_time_at_x(t0, x0, t1, x1, x_line_m)
                if ct is not None and t_start_ms <= ct < t_end_ms:
                    n += 1
    return n


def aggregate_cell_edie(
  trajectories: Sequence[NormalizedTrajectory],
  *,
  t_start_ms: int,
  t_end_ms: int,
  x_start_m: float,
  x_end_m: float,
  dt_sample_ms: int = NGSIM_DT_MS,
  max_gap_ms: int = MAX_GAP_MS_DEFAULT,
  count_dataset_edges: bool = True,
) -> MacroCellAggregate:
    """Edie q,k,v from clipped segments; separate snapshot and crossing flux metrics."""
    dx_m = x_end_m - x_start_m
    dt_s = (t_end_ms - t_start_ms) / 1000.0
    area_m_s = dx_m * dt_s
    acc = TrackAccounting()

    total_time_s = 0.0
    total_dist_m = 0.0
    n_pts = 0

    crossing_times_in: list[float] = []
    crossing_times_out: list[float] = []

    for tr in trajectories:
        if tr.t_ms.size < 2:
            continue

        gaps = np.diff(tr.t_ms) > max_gap_ms
        acc.fragmentation_events += int(np.sum(gaps))

        for i in range(tr.t_ms.size - 1):
            t0, x0 = float(tr.t_ms[i]), float(tr.x_m[i])
            t1, x1 = float(tr.t_ms[i + 1]), float(tr.x_m[i + 1])
            clipped = clip_segment_to_cell(
                t0, x0, t1, x1,
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
                x_start_m=x_start_m,
                x_end_m=x_end_m,
            )
            if clipped is not None:
                a_lo, a_hi = clipped
                seg_dt_s = (t1 - t0) / 1000.0
                total_time_s += seg_dt_s * (a_hi - a_lo)
                total_dist_m += abs(x1 - x0) * (a_hi - a_lo)
                n_pts += 1

            if segment_crosses_x(t0, x0, t1, x1, x_start_m):
                ct = crossing_time_at_x(t0, x0, t1, x1, x_start_m)
                if ct is not None and t_start_ms <= ct < t_end_ms and x1 > x0:
                    crossing_times_in.append(ct)
            if segment_crosses_x(t0, x0, t1, x1, x_end_m):
                ct = crossing_time_at_x(t0, x0, t1, x1, x_end_m)
                if ct is not None and t_start_ms <= ct < t_end_ms and x1 > x0:
                    crossing_times_out.append(ct)

        # Track accounting heuristics (observed, not ground-truth conservation).
        acc.negative_speed_samples += tr.negative_speed_count
        finite_v = np.isfinite(tr.v_mps)
        stat = int(np.sum(finite_v & (tr.v_mps <= 0.05)))
        acc.stationary_samples += stat

        inside_mask = (tr.x_m >= x_start_m) & (tr.x_m < x_end_m) & (tr.t_ms >= t_start_ms) & (tr.t_ms < t_end_ms)
        if not np.any(inside_mask):
            continue

        first_idx = int(np.argmax(inside_mask))
        last_idx = int(len(inside_mask) - 1 - np.argmax(inside_mask[::-1]))

        x_at_open = position_at_time(tr, float(t_start_ms))
        carry_in = (
            x_at_open is not None
            and in_spatial_cell(x_at_open, x_start_m, x_end_m)
        )
        entered_via_upstream = _upstream_entry_in_window(
            tr,
            first_idx,
            x_start_m=x_start_m,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
        )

        if carry_in:
            acc.time_window_carry_in += 1
        elif first_idx == 0 and in_spatial_cell(float(tr.x_m[0]), x_start_m, x_end_m):
            if not entered_via_upstream:
                acc.dataset_boundary_births += 1
        elif not entered_via_upstream:
            acc.internal_appearances += 1

        x_at_close = position_at_time(tr, float(t_end_ms - 1))
        carry_out = (
            x_at_close is not None
            and in_spatial_cell(x_at_close, x_start_m, x_end_m)
        )
        exited_via_downstream = _downstream_exit_in_window(
            tr,
            last_idx,
            x_end_m=x_end_m,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
        )

        if carry_out and not exited_via_downstream:
            acc.time_window_carry_out += 1
        elif last_idx == tr.t_ms.size - 1 and in_spatial_cell(float(tr.x_m[-1]), x_start_m, x_end_m):
            if not exited_via_downstream:
                acc.dataset_boundary_deaths += 1
        elif not exited_via_downstream and in_spatial_cell(float(tr.x_m[last_idx]), x_start_m, x_end_m):
            acc.internal_disappearances += 1

    if area_m_s <= 0 or total_time_s <= 1e-12:
        edie_q = edie_k = float("nan")
        edie_v_kmh = float("nan")
        edie_k_vpmi = None
        edie_v_mph = None
    else:
        # Edie on metric cell: q [vph], k [veh/km], v [km/h]; all-lane corridor cross-section.
        edie_q = (total_dist_m / area_m_s) * 3600.0
        edie_k = (total_time_s / area_m_s) * K_PER_M_TO_VPKM
        edie_v_kmh = (total_dist_m / total_time_s) * MPS_TO_KMH
        edie_k_vpmi = edie_k / MI_TO_KM
        edie_v_mph = edie_v_kmh / MI_TO_KM

    snap_start = count_vehicles_in_cell(
        trajectories, t_ms=float(t_start_ms), x_start_m=x_start_m, x_end_m=x_end_m
    )
    snap_end = count_vehicles_in_cell(
        trajectories, t_ms=float(t_end_ms), x_start_m=x_start_m, x_end_m=x_end_m
    )
    cross_in_n = count_upstream_crossings(
        trajectories,
        x_line_m=x_start_m,
        t_start_ms=float(t_start_ms),
        t_end_ms=float(t_end_ms),
    )
    cross_out_n = count_upstream_crossings(
        trajectories,
        x_line_m=x_end_m,
        t_start_ms=float(t_start_ms),
        t_end_ms=float(t_end_ms),
    )

    births = acc.internal_appearances + acc.dataset_boundary_births
    deaths = acc.internal_disappearances + acc.dataset_boundary_deaths
    identity_lhs = snap_end - snap_start
    identity_rhs = cross_in_n - cross_out_n + births - deaths
    storage_identity_residual = int(identity_lhs - identity_rhs)

    dt_hours = dt_s / 3600.0
    cross_in_vph = cross_in_n / dt_hours if dt_hours > 0 else 0.0
    cross_out_vph = cross_out_n / dt_hours if dt_hours > 0 else 0.0

    return MacroCellAggregate(
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        x_start_m=x_start_m,
        x_end_m=x_end_m,
        edie_travel_time_s=total_time_s,
        edie_distance_m=total_dist_m,
        edie_q_vph=float(edie_q),
        edie_k_vpkm=float(edie_k),
        edie_v_kmh=float(edie_v_kmh),
        edie_k_vpmi=float(edie_k_vpmi) if edie_k_vpmi is not None else None,
        edie_v_mph=float(edie_v_mph) if edie_v_mph is not None else None,
        snapshot_count_start=snap_start,
        snapshot_count_end=snap_end,
        snapshot_count_mean=0.5 * (snap_start + snap_end),
        fd_storage_delta=snap_end - snap_start,
        crossing_flux_in_vph=float(cross_in_vph),
        crossing_flux_out_vph=float(cross_out_vph),
        crossing_count_in=cross_in_n,
        crossing_count_out=cross_out_n,
        storage_identity_residual_veh=storage_identity_residual,
        n_trajectory_points_in_cell=n_pts,
        track_accounting=acc,
    )


def remove_incomplete_edge_frames(
    df: pd.DataFrame,
    *,
    time_col: str = "global_time",
    vehicle_col: str = "vehicle_id",
    sparse_fraction_of_median: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Conservative EDGE-only trim: optional sparse first/last timestamps; never interior argmax drops.

    SODA row-cap exports can truncate mid-recording; we do not infer completeness from interior
    per-frame vehicle counts. Interior density peaks are retained.
    """
    flags: dict[str, Any] = {
        "removed_edge_times_ms": [],
        "edge_trim_policy": "conservative_first_last_sparse_only",
    }
    if df.empty:
        return df, flags

    raw_t_min = int(df[time_col].min())
    raw_t_max = int(df[time_col].max())
    flags["raw_time_range_ms"] = [raw_t_min, raw_t_max]

    t_counts = df.groupby(time_col).size().sort_index()
    t_min = int(t_counts.index[0])
    t_last = int(t_counts.index[-1])
    median_n = float(np.median(t_counts.to_numpy()))
    frame_counts = t_counts.to_numpy(dtype=np.int64)
    flags["per_frame_count_summary"] = {
        "n_frames": int(len(t_counts)),
        "min": int(frame_counts.min()) if frame_counts.size else 0,
        "max": int(frame_counts.max()) if frame_counts.size else 0,
        "median": float(np.median(frame_counts)) if frame_counts.size else 0.0,
        "first_ms": t_min,
        "last_ms": t_last,
        "first_n": int(t_counts.iloc[0]),
        "last_n": int(t_counts.iloc[-1]),
        "max_count_ms": int(t_counts.idxmax()),
        "max_count_n": int(t_counts.max()),
    }

    drop_rationale: list[dict[str, Any]] = []
    drop_times: list[int] = []
    sparse_threshold = max(1, sparse_fraction_of_median * median_n)

    if median_n >= 5 and t_counts.iloc[0] <= sparse_threshold:
        drop_times.append(t_min)
        drop_rationale.append(
            {
                "global_time_ms": t_min,
                "reason": "sparse_first_timestamp",
                "vehicle_count": int(t_counts.iloc[0]),
                "median_count": median_n,
            }
        )
    if len(t_counts) > 1 and median_n >= 5 and t_counts.iloc[-1] <= sparse_threshold:
        drop_times.append(t_last)
        drop_rationale.append(
            {
                "global_time_ms": t_last,
                "reason": "sparse_last_timestamp",
                "vehicle_count": int(t_counts.iloc[-1]),
                "median_count": median_n,
            }
        )

    drop_times = sorted(set(drop_times))
    out = df[~df[time_col].isin(drop_times)].copy()
    flags["removed_edge_times_ms"] = drop_times
    flags["drop_rationale"] = drop_rationale
    flags["frame_count_median"] = median_n
    flags["retained_rows"] = int(len(out))
    flags["dropped_rows"] = int(len(df) - len(out))
    if len(out):
        flags["retained_time_range_ms"] = [int(out[time_col].min()), int(out[time_col].max())]
    else:
        flags["retained_time_range_ms"] = [raw_t_min, raw_t_max]
    flags["frame_completeness_guarantee"] = (
        "none — edge trim is count-heuristic on first/last timestamps only; interior frames kept verbatim"
    )
    return out, flags


def _upstream_entry_in_window(
    tr: NormalizedTrajectory,
    first_idx: int,
    *,
    x_start_m: float,
    t_start_ms: int,
    t_end_ms: int,
) -> bool:
    """True if vehicle entered cell through upstream face within [t_start, t_end)."""
    if first_idx > 0:
        t0, x0 = float(tr.t_ms[first_idx - 1]), float(tr.x_m[first_idx - 1])
        t1, x1 = float(tr.t_ms[first_idx]), float(tr.x_m[first_idx])
        if segment_crosses_x(t0, x0, t1, x1, x_start_m) and x1 > x0:
            ct = crossing_time_at_x(t0, x0, t1, x1, x_start_m)
            if ct is not None and t_start_ms <= ct < t_end_ms:
                return True
    return False


def _downstream_exit_in_window(
    tr: NormalizedTrajectory,
    last_idx: int,
    *,
    x_end_m: float,
    t_start_ms: int,
    t_end_ms: int,
) -> bool:
    if last_idx + 1 < tr.t_ms.size:
        t0, x0 = float(tr.t_ms[last_idx]), float(tr.x_m[last_idx])
        t1, x1 = float(tr.t_ms[last_idx + 1]), float(tr.x_m[last_idx + 1])
        if segment_crosses_x(t0, x0, t1, x1, x_end_m) and x1 > x0:
            ct = crossing_time_at_x(t0, x0, t1, x1, x_end_m)
            if ct is not None and t_start_ms <= ct < t_end_ms:
                return True
    return False


def _ngsim_speed_row(v_raw: float, v_infer: float | None) -> tuple[float, int, int]:
    """Map NGSIM ft/s sample to m/s. NaN = missing/undefined speed (not standstill)."""
    neg = 0
    infer = 0
    if not np.isfinite(v_raw):
        if v_infer is not None and np.isfinite(v_infer):
            infer += 1
            return float(v_infer), neg, infer
        return float("nan"), neg, infer
    if v_raw < 0:
        neg += 1
        if SPEED_NEGATIVE_POLICY == "zero_clip_reported":
            v_raw = 0.0
        else:
            v_raw = np.nan
    if v_infer is not None and np.isfinite(v_infer) and (not np.isfinite(v_raw)):
        infer += 1
        v_raw = v_infer
    elif v_infer is not None and np.isfinite(v_infer) and np.isfinite(v_raw) and v_raw <= 0:
        infer += 1
        v_raw = v_infer
    return float(v_raw) if np.isfinite(v_raw) else float("nan"), neg, infer


def load_ngsim_i80_csv(
    path: Path,
    *,
    expected_sha256: str | None = None,
    x_col: str = "local_y",
    lane_col: str = "lane_id",
) -> tuple[list[NormalizedTrajectory], AdapterProvenance]:
    path = Path(path)
    raw_sha = _sha256_file(path)
    if expected_sha256 and raw_sha != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {raw_sha} != {expected_sha256}")

    df = pd.read_csv(path)
    df, edge_flags = remove_incomplete_edge_frames(df)

    trajectories: list[NormalizedTrajectory] = []
    policy = {
        "speed_negative_policy": SPEED_NEGATIVE_POLICY,
        "min_trajectory_points": MIN_TRAJECTORY_POINTS,
        "max_gap_ms": MAX_GAP_MS_DEFAULT,
        "longitudinal_column": x_col,
        "longitudinal_units_in": "feet",
    }

    for vid, grp in df.groupby("vehicle_id", sort=True):
        grp = grp.sort_values("global_time")
        if len(grp) < MIN_TRAJECTORY_POINTS:
            continue
        t_ms = grp["global_time"].to_numpy(dtype=np.int64)
        x_m = grp[x_col].to_numpy(dtype=np.float64) * FT_TO_M
        v_raw = grp["v_vel"].to_numpy(dtype=np.float64) * FT_TO_M
        v_mps = np.full(v_raw.shape, np.nan, dtype=np.float64)
        neg_total = 0
        inf_total = 0
        for i, v in enumerate(v_raw):
            v_inf = None
            if i > 0 and t_ms[i] > t_ms[i - 1]:
                v_inf = (x_m[i] - x_m[i - 1]) / ((t_ms[i] - t_ms[i - 1]) / 1000.0)
            vv, neg, inf = _ngsim_speed_row(v, v_inf)
            v_mps[i] = vv
            neg_total += neg
            inf_total += inf
        lane = int(grp[lane_col].iloc[0]) if lane_col in grp else None
        trajectories.append(
            NormalizedTrajectory(
                vehicle_key=f"ngsim:{int(vid)}",
                t_ms=t_ms,
                x_m=x_m,
                v_mps=v_mps,
                lane_id=lane,
                negative_speed_count=neg_total,
                inferred_speed_count=inf_total,
                gap_count=int(np.sum(np.diff(t_ms) > MAX_GAP_MS_DEFAULT)),
            )
        )

    t_range = (int(df["global_time"].min()), int(df["global_time"].max())) if len(df) else (0, 0)
    coverage = {
        "source_rows": int(len(df)),
        "n_trajectories": len(trajectories),
        "time_range_ms": list(t_range),
        "duration_s": (t_range[1] - t_range[0]) / 1000.0 if t_range[1] > t_range[0] else 0.0,
        "slice_limit_rows": 25000,
        "truncation_note": "SODA $limit=25000 may truncate mid-recording; conservative edge-only trim only.",
        **edge_flags,
    }

    prov = AdapterProvenance(
        source_id="ngsim_i80_slice",
        raw_sha256=raw_sha,
        adapter_source_hash=adapter_module_hash(),
        units={
            "time": "ms",
            "longitudinal": "m",
            "trajectory_speed": "m/s",
            "edie_flow": "vph_all_lanes_corridor",
            "edie_density": "veh_per_km_all_lanes",
            "edie_speed": "km_per_h_space_mean",
            "flow_scope": FLOW_SCOPE_ALL_LANES,
            "edie_qkv_units_note": "q_vph = k_vpkm * v_kmh when all three finite (definitional)",
        },
        coverage_flags=coverage,
        policy=policy,
    )
    return trajectories, prov


class I24SkipReason(Exception):
    """Raised when I-24 JSON cannot be mapped to metric longitudinal trajectories."""


class UnsupportedCoordinateSystem(I24SkipReason):
    """world_position axis basis is not verified as metric longitudinal coordinates."""


def load_i24_motion_json(path: Path, *, expected_sha256: str | None = None) -> tuple[list[NormalizedTrajectory], AdapterProvenance]:
    """Probe I-24 example JSON; fail closed unless metric longitudinal basis is explicitly declared."""
    path = Path(path)
    raw_sha = _sha256_file(path)
    if expected_sha256 and raw_sha != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}")

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("metadata", {})
    frames = payload.get("data", {})
    if not isinstance(frames, dict):
        raise I24SkipReason("data field is not a frame-indexed dict")

    axis_basis = (
        meta.get("world_position_units")
        or meta.get("coordinate_system")
        or payload.get("coordinate_system")
    )
    pixel_only = 0
    world_pts = 0
    for vehicles in frames.values():
        if not isinstance(vehicles, list):
            continue
        for veh in vehicles:
            if veh.get("world_position"):
                world_pts += 1
            elif veh.get("image_position"):
                pixel_only += 1

    if world_pts == 0 and pixel_only > 0:
        raise I24SkipReason(
            f"Only image_position present ({pixel_only} detections); no calibration to longitudinal axis"
        )
    if world_pts == 0:
        raise I24SkipReason("No world_position detections in sample")

    raise UnsupportedCoordinateSystem(
        "world_position present but longitudinal axis units are unverified "
        f"(metadata world_position_units={axis_basis!r}; "
        f"{world_pts} detections, {pixel_only} image_position-only). "
        "Example-repo values may be satellite/pipeline world points, not surveyed meters; "
        "refusing blind unit conversion."
    )


def assign_non_overlapping_spatial_cells(
    x_min_m: float,
    x_max_m: float,
    n_cells: int,
) -> list[tuple[float, float]]:
    edges = np.linspace(x_min_m, x_max_m, n_cells + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_cells)]


def build_time_windows(t_start_ms: int, t_end_ms: int, dt_ms: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    t = t_start_ms
    while t + dt_ms <= t_end_ms:
        windows.append((t, t + dt_ms))
        t += dt_ms
    return windows


def macro_grid_from_trajectories(
    trajectories: Sequence[NormalizedTrajectory],
    *,
    n_spatial_cells: int,
    dt_ms: int,
    x_min_m: float | None = None,
    x_max_m: float | None = None,
) -> tuple[list[MacroCellAggregate], dict[str, Any]]:
    if not trajectories:
        return [], {"error": "no trajectories"}

    xs = np.concatenate([tr.x_m for tr in trajectories])
    t_all = np.concatenate([tr.t_ms for tr in trajectories])
    x_lo = float(x_min_m if x_min_m is not None else xs.min())
    x_hi = float(x_max_m if x_max_m is not None else xs.max())
    t_lo = int(t_all.min())
    t_hi = int(t_all.max())

    cells = assign_non_overlapping_spatial_cells(x_lo, x_hi, n_spatial_cells)
    times = build_time_windows(t_lo, t_hi, dt_ms)

    aggregates: list[MacroCellAggregate] = []
    for t0, t1 in times:
        for xs_m, xe_m in cells:
            aggregates.append(
                aggregate_cell_edie(
                    trajectories,
                    t_start_ms=t0,
                    t_end_ms=t1,
                    x_start_m=xs_m,
                    x_end_m=xe_m,
                )
            )

    meta = {
        "n_spatial_cells": n_spatial_cells,
        "n_time_windows": len(times),
        "x_range_m": [x_lo, x_hi],
        "t_range_ms": [t_lo, t_hi],
        "dt_ms": dt_ms,
        "cells_non_overlapping": True,
    }
    return aggregates, meta


def snapshots_to_link_storage_matrix(
    aggregates: Sequence[MacroCellAggregate],
    n_spatial_cells: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Rows: unique time windows; cols: spatial cell index → snapshot_count_end."""
    if not aggregates:
        return np.zeros((0, 0)), []

    keys = sorted({(a.t_start_ms, a.t_end_ms) for a in aggregates})
    key_index = {k: i for i, k in enumerate(keys)}
    mat = np.zeros((len(keys), n_spatial_cells), dtype=np.float64)
    for a in aggregates:
        # Infer cell index from x_start ordering — caller must keep consistent spacing.
        j = None
        # Placeholder; measured_runner maps explicitly.
        _ = j
    return mat, keys


def flux_balance_residual(
    aggregates: Sequence[MacroCellAggregate],
) -> dict[str, float]:
    """Per-window FD check: ΔN vs integer crossings (excludes birth/death slack terms)."""
    flux_res = []
    for a in aggregates:
        dt_s = (a.t_end_ms - a.t_start_ms) / 1000.0
        if dt_s <= 0:
            continue
        net_cross_veh = float(a.crossing_count_in - a.crossing_count_out)
        flux_res.append(float(a.fd_storage_delta - net_cross_veh))
    if not flux_res:
        return {"mean_residual_veh": float("nan"), "rmse_residual_veh": float("nan"), "n_cells": 0}
    arr = np.array(flux_res)
    return {
        "mean_residual_veh": float(np.mean(arr)),
        "rmse_residual_veh": float(np.sqrt(np.mean(arr ** 2))),
        "n_cells": len(arr),
        "note": "ignores interior births/deaths; see storage_identity_residual for full balance",
    }


def storage_identity_audit(
    aggregates: Sequence[MacroCellAggregate],
) -> dict[str, Any]:
    """Exact count identity per cell–window: ΔN − (in−out+births−deaths)."""
    residuals = [int(a.storage_identity_residual_veh) for a in aggregates]
    if not residuals:
        return {"n_cells": 0, "max_abs_residual_veh": float("nan")}
    arr = np.asarray(residuals, dtype=np.int64)
    return {
        "n_cells": int(arr.size),
        "mean_residual_veh": float(np.mean(arr)),
        "mean_abs_residual_veh": float(np.mean(np.abs(arr))),
        "rmse_residual_veh": float(np.sqrt(np.mean(arr.astype(float) ** 2))),
        "max_abs_residual_veh": int(np.max(np.abs(arr))),
        "n_exact_matches": int(np.sum(arr == 0)),
        "identity": "N_end - N_start == crossing_in - crossing_out + births - deaths",
    }


def exclusive_vehicle_cell_assignment(
    trajectories: Sequence[NormalizedTrajectory],
    cells: Sequence[tuple[float, float]],
    t_mid_ms: float,
) -> dict[str, int]:
    """Assign each vehicle to at most one cell at t_mid via midpoint containment."""
    assigned: dict[str, int] = {}
    for tr in trajectories:
        x = position_at_time(tr, t_mid_ms)
        if x is None:
            continue
        for j, (xs, xe) in enumerate(cells):
            if xs <= x < xe:
                if tr.vehicle_key in assigned:
                    raise ValueError(f"vehicle {tr.vehicle_key} would lie in multiple cells at t={t_mid_ms}")
                assigned[tr.vehicle_key] = j
                break
    return assigned
