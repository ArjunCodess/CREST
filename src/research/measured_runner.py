"""Development-only measured-data pilot (NGSIM slice); no final-protocol claims."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import baselines as bl
from .boundary_conservation import fit_boundary_conservation
from .config_paths import REPO_ROOT
from .provenance import research_code_content_hash
from .boundary_conservation import boundary_aggregate_lp_bundle
from .conservation_constraints import assemble_conservation_polytope, evaluate_affine_constraints
from .lp_core import linprog_extremum
from .metrics import dynamics_balance_rmse, state_rmse, timed
from .synthetic_sim import _chain_operators
from .trajectory_adapter import (
    MacroCellAggregate,
    UnsupportedCoordinateSystem,
    adapter_module_hash,
    flux_balance_residual,
    load_ngsim_i80_csv,
    load_i24_motion_json,
    I24SkipReason,
    macro_grid_from_trajectories,
    storage_identity_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "research" / "measured_pilot.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "research_measured"
NGSIM_SHA = "38173aad33806922b4f22f516363f8a4fce53038554e6d85744e151f05870d1b"

# NGSIM I-80 Emeryville study corridor (see docs/research_data.md); development metadata only.
NGSIM_I80_MAINLINE_LANE_COUNT_DEFAULT = 6
NGSIM_I80_PER_LANE_CAPACITY_VPH_DEFAULT = 2500.0
FLOW_SCOPE_ALL_LANES = "all_lanes_corridor_cross_section"


def load_measured_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _aggregate_table(
    aggregates: list[MacroCellAggregate], n_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(T+1)×L boundary snapshots N[0..T]; flux[k] crosses during [t_k, t_{k+1})."""
    times = sorted({(a.t_start_ms, a.t_end_ms) for a in aggregates})
    n_win = len(times)
    L = n_cells
    storage = np.full((n_win + 1, L), np.nan)
    flux_in = np.zeros((n_win, L))
    flux_out = np.zeros((n_win, L))
    cross_net_int = np.zeros((n_win, L), dtype=np.int64)

    x_starts = sorted({round(a.x_start_m, 6) for a in aggregates})
    x_to_j = {x: j for j, x in enumerate(x_starts)}
    t_index = {k: i for i, k in enumerate(times)}

    for a in aggregates:
        wi = t_index[(a.t_start_ms, a.t_end_ms)]
        j = x_to_j.get(round(a.x_start_m, 6))
        if j is None:
            continue
        storage[wi, j] = float(a.snapshot_count_start)
        storage[wi + 1, j] = float(a.snapshot_count_end)
        flux_in[wi, j] = a.crossing_flux_in_vph
        flux_out[wi, j] = a.crossing_flux_out_vph
        cross_net_int[wi, j] = a.crossing_count_in - a.crossing_count_out

    return storage, flux_in, flux_out, cross_net_int


def all_lane_vph_from_per_lane(per_lane_vph: float, lane_count: int) -> float:
    """Convert declared per-lane saturation flow to corridor all-lane vph (z/u upper bounds)."""
    if lane_count < 1:
        raise ValueError("lane_count must be >= 1")
    return float(per_lane_vph) * int(lane_count)


def resolve_flow_capacity_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Physical envelope metadata; independent of prefix calibration and holdout eval."""
    lane_count = int(cfg.get("mainline_lane_count", NGSIM_I80_MAINLINE_LANE_COUNT_DEFAULT))
    per_lane = float(
        cfg.get(
            "per_lane_capacity_vph",
            cfg.get("max_edge_flow_vph", NGSIM_I80_PER_LANE_CAPACITY_VPH_DEFAULT),
        )
    )
    all_lane = all_lane_vph_from_per_lane(per_lane, lane_count)
    x_min = cfg.get("x_min_m")
    x_max = cfg.get("x_max_m")
    corridor_m = float(x_max) - float(x_min) if x_min is not None and x_max is not None else None
    return {
        "flow_flux_scope": cfg.get("flow_flux_scope", FLOW_SCOPE_ALL_LANES),
        "mainline_lane_count": lane_count,
        "per_lane_capacity_vph": per_lane,
        "all_lane_capacity_vph": all_lane,
        "per_lane_capacity_rationale": cfg.get(
            "per_lane_capacity_rationale",
            "declared_freeway_saturation_order_of_magnitude; see docs/research_data.md (FHWA NGSIM I-80 geometry)",
        ),
        "corridor_longitudinal_coverage_m": corridor_m,
        "spatial_cell_scope": "longitudinal_bins_aggregate_all_lane_crossings_and_snapshots",
        "note": "Adapter crossing_flux_*_vph counts all vehicles crossing a plane (all lanes), not per-lane.",
    }


def physical_internal_flow_bounds(
    Tm1: int,
    n_edge: int,
    *,
    all_lane_capacity_vph: float,
) -> tuple[np.ndarray, str]:
    if n_edge <= 0:
        return np.zeros((Tm1, 0)), "no_internal_edges"
    z_hi = np.full((Tm1, n_edge), float(all_lane_capacity_vph))
    return z_hi, "declared_physical_envelope_all_lanes"


def calibrate_internal_flow_bounds(
    flux_out: np.ndarray,
    flux_in: np.ndarray,
    *,
    calib_steps: int,
    all_lane_fallback_vph: float,
    margin: float = 1.25,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Prefix max of observed inter-cell crossing rates (all-lane vph); floor is physical all-lane cap."""
    Tm1, L = flux_out.shape
    n_edge = max(L - 1, 0)
    diag: dict[str, Any] = {
        "calibration_prefix_steps": calib_steps,
        "margin": margin,
        "all_lane_fallback_vph": all_lane_fallback_vph,
        "prefix_sample_count": 0,
        "prefix_max_observed_vph": None,
        "applied_all_lane_floor_vph": all_lane_fallback_vph,
    }
    if n_edge == 0:
        return np.zeros((Tm1, 0)), "no_internal_edges", diag
    z_hi = np.full((Tm1, n_edge), all_lane_fallback_vph)
    n_use = min(calib_steps, Tm1)
    samples: list[float] = []
    for t in range(n_use):
        for e in range(n_edge):
            samples.append(float(max(flux_out[t, e], flux_in[t, e + 1])))
    diag["prefix_sample_count"] = len(samples)
    if samples:
        prefix_max = max(samples)
        diag["prefix_max_observed_vph"] = prefix_max
        prefix_cap = prefix_max * margin
        applied = max(all_lane_fallback_vph, prefix_cap)
        diag["applied_all_lane_floor_vph"] = applied
        z_hi[:, :] = applied
        return z_hi, "prefix_internal_crossing_calibration", diag
    return z_hi, "insufficient_internal_flux_fixed_broad", diag


def diagnose_lp_infeasibility(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    obs_mask: np.ndarray,
    obs_values: np.ndarray,
    *,
    z_upper: np.ndarray,
    obs_tolerance: float,
) -> dict[str, Any]:
    """Separate flow-cap infeasibility from storage-observation vs chain-dynamics mismatch."""
    vals = np.nan_to_num(obs_values, nan=0.0)
    no_obs = np.zeros_like(obs_mask, dtype=bool)
    z_huge = np.full_like(z_upper, 1.0e9)

    w_config = lp_feasibility_witness(
        B, R, dt_hours, n0, u_lo, u_hi, obs_mask, vals, z_upper=z_upper, obs_tolerance=obs_tolerance
    )
    w_no_obs = lp_feasibility_witness(
        B, R, dt_hours, n0, u_lo, u_hi, no_obs, vals, z_upper=z_upper, obs_tolerance=0.0
    )
    w_loose_caps = lp_feasibility_witness(
        B, R, dt_hours, n0, u_lo, u_hi, obs_mask, vals, z_upper=z_huge, obs_tolerance=obs_tolerance
    )

    lo, hi = 0.0, max(float(obs_tolerance) * 50.0, 100.0)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        wm = lp_feasibility_witness(
            B, R, dt_hours, n0, u_lo, u_hi, obs_mask, vals, z_upper=z_upper, obs_tolerance=mid
        )
        if wm.get("feasible"):
            hi = mid
        else:
            lo = mid
    approx_min_tol = hi

    out: dict[str, Any] = {
        "feasible_at_config_tolerance": bool(w_config.get("feasible")),
        "feasible_dynamics_only_no_storage_observations": bool(w_no_obs.get("feasible")),
        "feasible_with_unbounded_internal_flow_at_config_tolerance": bool(w_loose_caps.get("feasible")),
        "approx_min_obs_tolerance_veh_for_lp_feasibility": approx_min_tol,
        "n_observation_inequalities_at_config_tolerance": w_config.get("n_observation_inequalities"),
    }
    if w_no_obs.get("feasible") and not w_loose_caps.get("feasible"):
        out["constraint_witness"] = "storage_hard_constraints_inconsistent_with_chain_dynamics"
        out["core_api_gap"] = (
            "Measured per-link snapshots are exact in the adapter but the conservation LP cannot satisfy "
            "all storage rows within the stated tolerance even when internal flow caps are unbounded "
            f"(approx_min_obs_tolerance_veh≈{approx_min_tol:.2f}). "
            "Remaining gap is model/data coupling (not the retired single-u shared boundary). "
            "Do not invent per-cell slack without explicit tracking-bounded sources."
        )
    elif not w_no_obs.get("feasible"):
        out["constraint_witness"] = "boundary_or_box_infeasible"
    elif w_loose_caps.get("feasible"):
        out["constraint_witness"] = "feasible_under_declared_bounds"
    return out


def lp_feasibility_witness(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    obs_mask: np.ndarray,
    obs_values: np.ndarray,
    *,
    z_upper: np.ndarray,
    obs_tolerance: float,
) -> dict[str, Any]:
    """Feasibility LP: min 0 s.t. polytope; primal violation if optimum fails."""
    poly = assemble_conservation_polytope(
        B,
        R,
        dt_hours,
        n0,
        u_lo,
        u_hi,
        obs_mask=obs_mask,
        obs_values=obs_values,
        obs_tolerance=obs_tolerance,
        z_upper=z_upper,
    )
    c = np.zeros(poly.n_vars)
    res = linprog_extremum(
        c,
        A_ub=poly.A_ub,
        b_ub=poly.b_ub,
        lower=poly.lower,
        upper=poly.upper,
        sense="min",
    )
    witness: dict[str, Any] = {
        "lp_status": res.status,
        "n_vars": poly.n_vars,
        "n_observation_inequalities": int(poly.observation_constraints_applied()),
        "message": res.message,
    }
    if res.status == "optimal" and res.primal is not None:
        feas = evaluate_affine_constraints(res.primal, poly.A_ub, poly.b_ub)
        box_viol = float(np.max(np.maximum(poly.lower - res.primal, res.primal - poly.upper)))
        witness["feasible"] = feas["feasible"] and box_viol <= 1e-7
        witness["max_affine_violation"] = feas["max_violation"]
        witness["max_box_violation"] = max(0.0, box_viol)
    else:
        witness["feasible"] = False
    return witness


def calibrate_boundary_bounds(
    flux_in: np.ndarray,
    flux_out: np.ndarray,
    *,
    calib_steps: int,
    fallback_lower: float,
    fallback_upper: float,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Independent inlet (ch0) and outlet (ch1) bounds from disjoint prefix flux samples."""
    n_bound = 2
    if flux_in.size == 0:
        lo = np.full((1, n_bound), fallback_lower)
        hi = np.full((1, n_bound), fallback_upper)
        return lo, hi, "insufficient_calibration_fixed_broad"

    n_use = min(calib_steps, flux_in.shape[0])
    upstream = flux_in[:n_use, 0]
    downstream = flux_out[:n_use, -1] if flux_out.shape[1] else np.zeros(n_use)
    downstream = np.maximum(downstream, 0.0)

    Tm1 = flux_in.shape[0]
    u_lo = np.zeros((Tm1, n_bound))
    u_hi = np.zeros((Tm1, n_bound))

    def _channel_bounds(samples: np.ndarray, lo0: float, hi0: float) -> tuple[float, float, str]:
        if len(samples) < min_samples:
            return lo0, hi0, "insufficient_calibration_fixed_broad"
        lo_val = float(np.percentile(samples, 5))
        hi_val = float(np.percentile(samples, 95))
        lo_val = max(0.0, lo_val * 0.5)
        hi_val = max(hi_val * 1.5, lo_val + 50.0)
        return lo_val, hi_val, "prefix_calibration"

    lo_in, hi_in, st_in = _channel_bounds(upstream, fallback_lower, fallback_upper)
    lo_out, hi_out, st_out = _channel_bounds(downstream, fallback_lower, fallback_upper)
    status = st_in if st_in == st_out else f"{st_in}+{st_out}"

    u_lo[:, 0] = lo_in
    u_hi[:, 0] = hi_in
    u_lo[:, 1] = lo_out
    u_hi[:, 1] = hi_out
    return u_lo, u_hi, status


def physical_boundary_bounds(
    Tm1: int,
    *,
    all_lane_capacity_vph: float,
    fallback_lower_vph: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    n_bound = 2
    u_lo = np.full((Tm1, n_bound), float(fallback_lower_vph))
    u_hi = np.full((Tm1, n_bound), float(all_lane_capacity_vph))
    return u_lo, u_hi, "declared_physical_envelope_all_lanes"


def empirical_boundary_vehicle_totals(
    flux_in: np.ndarray,
    flux_out: np.ndarray,
    dt_hours: float,
) -> dict[str, float]:
    """Reference totals from adapter flux (not LP); outlet magnitudes nonnegative."""
    inflow_vph = float(np.sum(flux_in[:, 0])) if flux_in.size else 0.0
    outflow_vph = float(np.sum(flux_out[:, -1])) if flux_out.size else 0.0
    return {
        "cumulative_inflow_vehicles": inflow_vph * dt_hours,
        "cumulative_outflow_vehicles": outflow_vph * dt_hours,
        "net_inflow_vehicles": (inflow_vph - outflow_vph) * dt_hours,
    }


def _attempt_inference(
    label: str,
    mask: np.ndarray,
    obs_values: np.ndarray,
    *,
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_prior: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    fit_kwargs: dict[str, Any],
    feas_witness: dict[str, Any],
    obs_tol: float,
    snapshot_fd: dict[str, Any],
) -> dict[str, Any]:
    vals_for_fit = np.nan_to_num(np.where(mask, obs_values, np.nan), nan=0.0)
    lp_ok = bool(feas_witness.get("feasible"))
    if not lp_ok:
        return {
            "label": label,
            "status": "skipped_lp_infeasible",
            "elapsed_s": 0.0,
            "dynamics_algebraic_residual_rmse": None,
            "primal_feasibility_max_violation": None,
            "failure_category": _classify_inference_failure(
                "skipped_lp_infeasible",
                0.0,
                obs_tol,
                snapshot_fd,
                lp_feasible=False,
            ),
            "lp_feasible_witness": feas_witness,
        }
    est, elapsed = timed(
        fit_boundary_conservation,
        B,
        R,
        dt_hours,
        n0,
        mask,
        vals_for_fit,
        u_prior,
        u_lo,
        u_hi,
        **fit_kwargs,
    )
    row = {
        "label": label,
        "status": est.status,
        "elapsed_s": elapsed,
        "dynamics_algebraic_residual_rmse": est.dynamics_algebraic_residual_rmse,
        "primal_feasibility_max_violation": est.primal_feasibility_max_violation,
        "lp_feasible_witness": feas_witness,
    }
    row["failure_category"] = _classify_inference_failure(
        est.status,
        float(est.primal_feasibility_max_violation or 0.0),
        obs_tol,
        snapshot_fd,
        lp_feasible=lp_ok,
    )
    return row


def run_measured_pilot(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(cfg.get("repo_root", REPO_ROOT))
    ngsim_path = repo / cfg["ngsim_path"]
    trajectories, prov = load_ngsim_i80_csv(ngsim_path, expected_sha256=cfg.get("ngsim_sha256", NGSIM_SHA))

    n_cells = int(cfg.get("n_spatial_cells", 4))
    dt_ms = int(cfg.get("dt_ms", 15000))
    aggregates, grid_meta = macro_grid_from_trajectories(
        trajectories,
        n_spatial_cells=n_cells,
        dt_ms=dt_ms,
        x_min_m=cfg.get("x_min_m"),
        x_max_m=cfg.get("x_max_m"),
    )

    storage_snap, flux_in, flux_out, cross_net_int = _aggregate_table(aggregates, n_cells)
    Tm1_obs, L = storage_snap.shape[0] - 1, storage_snap.shape[1]
    if Tm1_obs < 1 or L < 2:
        return {
            "status": "insufficient_evidence",
            "message": "Time/spatial coverage too small for inference grid",
            "provenance": prov.to_dict(),
            "grid_meta": grid_meta,
        }

    dt_hours = dt_ms / 3600.0 / 1000.0
    B, R = _chain_operators(L)
    n0 = np.nan_to_num(storage_snap[0], nan=0.0)
    flow_meta = resolve_flow_capacity_metadata(cfg)
    all_lane_cap = float(flow_meta["all_lane_capacity_vph"])
    calib_steps = int(cfg.get("calibration_prefix_steps", max(1, Tm1_obs // 2)))
    fallback_u_lo = float(cfg.get("fallback_u_lower_vph", 0.0))
    fallback_u_hi_legacy = float(cfg.get("fallback_u_upper_vph", all_lane_cap))
    # Legacy configs stored per-lane-ish magnitudes; treat sub-all-lane fallback as mis-scoped.
    if fallback_u_hi_legacy < all_lane_cap:
        fallback_u_hi_legacy = all_lane_cap

    u_lo_prefix, u_hi_prefix, calib_status = calibrate_boundary_bounds(
        flux_in,
        flux_out,
        calib_steps=calib_steps,
        fallback_lower=fallback_u_lo,
        fallback_upper=fallback_u_hi_legacy,
    )
    z_prefix, z_prefix_status, z_prefix_diag = calibrate_internal_flow_bounds(
        flux_out,
        flux_in,
        calib_steps=calib_steps,
        all_lane_fallback_vph=all_lane_cap,
    )

    u_lo_phys, u_hi_phys, u_phys_status = physical_boundary_bounds(
        Tm1_obs,
        all_lane_capacity_vph=all_lane_cap,
        fallback_lower_vph=fallback_u_lo,
    )
    z_phys, z_phys_status = physical_internal_flow_bounds(
        Tm1_obs,
        max(L - 1, 0),
        all_lane_capacity_vph=all_lane_cap,
    )
    n_edge = max(L - 1, 0)
    observed_peak_internal = 0.0
    if n_edge > 0 and flux_out.size:
        for t in range(flux_out.shape[0]):
            for e in range(n_edge):
                observed_peak_internal = max(
                    observed_peak_internal, float(flux_out[t, e]), float(flux_in[t, e + 1])
                )

    mask_seed = int(cfg.get("mask_seed", 42))
    missing_rate = float(cfg.get("missing_rate", 0.2))
    rng = np.random.default_rng(mask_seed)
    obs_mask = np.ones_like(storage_snap, dtype=bool)
    n_missing = int(round(missing_rate * obs_mask.size))
    flat_idx = rng.choice(obs_mask.size, size=n_missing, replace=False)
    obs_mask.ravel()[flat_idx] = False
    obs_values = storage_snap.copy()

    obs_tol = float(cfg.get("obs_tolerance_veh", 3.0))
    tracking = flux_balance_residual(aggregates)
    identity_audit = storage_identity_audit(aggregates)
    ta_summary = _summarize_track_accounting(aggregates)
    snapshot_fd = _snapshot_flux_accounting_residual(
        storage_snap, cross_net_int, flux_in, flux_out, dt_hours
    )

    masked_vals = np.nan_to_num(np.where(obs_mask, obs_values, np.nan), nan=0.0)
    imputed = bl.masked_mean_imputer(masked_vals, obs_mask)
    impute_err = state_rmse(imputed, storage_snap, obs_mask, region="missing")

    flow_scenarios: list[dict[str, Any]] = []
    for scenario_id, u_lo, u_hi, z_upper_mat, z_status, extra in [
        (
            "physical_envelope",
            u_lo_phys,
            u_hi_phys,
            z_phys,
            z_phys_status,
            {"boundary_status": u_phys_status},
        ),
        (
            "prefix_empirical",
            u_lo_prefix,
            u_hi_prefix,
            z_prefix,
            z_prefix_status,
            {"boundary_status": calib_status, "prefix_calibration": z_prefix_diag},
        ),
    ]:
        u_prior = 0.5 * (u_lo + u_hi)
        z_fill = (
            z_upper_mat
            if z_upper_mat.size
            else np.full((max(Tm1_obs, 1), B.shape[1]), all_lane_cap)
        )
        fit_kwargs = dict(
            w_obs=0.0,
            w_prior=0.1,
            z_upper=z_fill,
            z0=np.full((max(Tm1_obs, 1), B.shape[1]), all_lane_cap / max(L, 1)),
            obs_tolerance=obs_tol,
            use_hard_observation_constraints=True,
        )
        feas_full = lp_feasibility_witness(
            B,
            R,
            dt_hours,
            n0,
            u_lo,
            u_hi,
            np.ones_like(obs_mask),
            np.nan_to_num(storage_snap, nan=0.0),
            z_upper=z_fill,
            obs_tolerance=obs_tol,
        )
        feas_masked = lp_feasibility_witness(
            B,
            R,
            dt_hours,
            n0,
            u_lo,
            u_hi,
            obs_mask,
            np.nan_to_num(storage_snap, nan=0.0),
            z_upper=z_fill,
            obs_tolerance=obs_tol,
        )
        lp_diagnosis = diagnose_lp_infeasibility(
            B,
            R,
            dt_hours,
            n0,
            u_lo,
            u_hi,
            obs_mask,
            storage_snap,
            z_upper=z_fill,
            obs_tolerance=obs_tol,
        )
        bnd_lp = boundary_aggregate_lp_bundle(
            B,
            R,
            dt_hours,
            n0,
            u_lo,
            u_hi,
            obs_mask=obs_mask,
            obs_values=np.nan_to_num(storage_snap, nan=0.0),
            obs_tolerance=obs_tol,
            z_upper=z_fill,
            include_observations=True,
        )
        bnd_lp_box = boundary_aggregate_lp_bundle(
            B,
            R,
            dt_hours,
            n0,
            u_lo,
            u_hi,
            z_upper=z_fill,
            include_observations=False,
        )
        attempts = [
            _attempt_inference(
                "full_obs",
                np.ones_like(obs_mask),
                obs_values,
                B=B,
                R=R,
                dt_hours=dt_hours,
                n0=n0,
                u_prior=u_prior,
                u_lo=u_lo,
                u_hi=u_hi,
                fit_kwargs=fit_kwargs,
                feas_witness=feas_full,
                obs_tol=obs_tol,
                snapshot_fd=snapshot_fd,
            ),
            _attempt_inference(
                "masked",
                obs_mask,
                obs_values,
                B=B,
                R=R,
                dt_hours=dt_hours,
                n0=n0,
                u_prior=u_prior,
                u_lo=u_lo,
                u_hi=u_hi,
                fit_kwargs=fit_kwargs,
                feas_witness=feas_masked,
                obs_tol=obs_tol,
                snapshot_fd=snapshot_fd,
            ),
        ]
        n_fail = sum(1 for a in attempts if a["status"] not in ("optimal",))
        failure_rate = n_fail / max(len(attempts), 1)

        masked_est = None
        bounded_rmse_missing = None
        bounded_rmse_observed = None
        dyn_rmse = None
        if attempts[1]["status"] == "optimal":
            masked_est, _ = timed(
                fit_boundary_conservation,
                B,
                R,
                dt_hours,
                n0,
                obs_mask,
                np.nan_to_num(np.where(obs_mask, obs_values, 0.0)),
                u_prior,
                u_lo,
                u_hi,
                **fit_kwargs,
            )
            bounded_rmse_missing = state_rmse(masked_est.storage, storage_snap, obs_mask, region="missing")
            bounded_rmse_observed = state_rmse(masked_est.storage, storage_snap, obs_mask, region="observed")
            dyn_rmse = float(
                dynamics_balance_rmse(
                    masked_est.storage,
                    B,
                    R,
                    masked_est.internal_flow,
                    masked_est.boundary_flow,
                    dt_hours,
                )
            )

        if scenario_id == "prefix_empirical" and "prefix_calibration" in extra:
            legacy_cap = float(cfg.get("max_edge_flow_vph", float(flow_meta["per_lane_capacity_vph"])))
            extra["prefix_calibration"]["legacy_mis_scoped_per_lane_as_all_lane_vph"] = legacy_cap
            extra["prefix_calibration"]["observed_peak_internal_vph"] = observed_peak_internal
            extra["prefix_calibration"]["prefix_under_observed_peak"] = (
                extra["prefix_calibration"].get("applied_all_lane_floor_vph", 0.0) < observed_peak_internal
            )

        flow_scenarios.append({
            "scenario_id": scenario_id,
            "internal_flow_bounds_status": z_status,
            "internal_flow_upper_vph": float(np.max(z_upper_mat)) if z_upper_mat.size else all_lane_cap,
            "lp_infeasibility_diagnosis": lp_diagnosis,
            "boundary_aggregate_lp_data_conditioned": bnd_lp,
            "boundary_aggregate_lp_box_only": bnd_lp_box,
            "feasibility_lp_witness_full_obs": feas_full,
            "feasibility_lp_witness_masked": feas_masked,
            "inference_attempts": attempts,
            "numerical_failure_rate": failure_rate,
            "conditional_performance": {
                "note": "RMSE only when masked attempt is optimal; same mask_seed as baseline imputer",
                "missing_region_storage_rmse": bounded_rmse_missing,
                "observed_region_storage_rmse": bounded_rmse_observed,
                "masked_mean_imputer_missing_rmse": impute_err,
                "dynamics_balance_rmse_masked": dyn_rmse,
            },
            **extra,
        })

    primary = next(s for s in flow_scenarios if s["scenario_id"] == "physical_envelope")
    attempts = primary["inference_attempts"]
    feas_witness = primary["feasibility_lp_witness_masked"]
    failure_rate = primary["numerical_failure_rate"]
    z_upper_mat = z_phys
    z_calib_status = z_phys_status

    i24_status = None
    i24_path = repo / cfg.get("i24_path", "data/research_external/i24_motion_sample_3.json")
    if cfg.get("probe_i24", True) and i24_path.exists():
        try:
            _, i24_prov = load_i24_motion_json(i24_path)
            i24_status = {"status": "loaded", "provenance": i24_prov.to_dict()}
        except (I24SkipReason, UnsupportedCoordinateSystem) as exc:
            i24_status = {"status": "skipped", "reason": str(exc), "fail_closed": isinstance(exc, UnsupportedCoordinateSystem)}

    inference_ok = any(
        a["status"] == "optimal"
        for s in flow_scenarios
        for a in s["inference_attempts"]
    )
    overall_status = "dev_pilot_complete" if inference_ok else "dev_pilot_aggregation_only_inference_insufficient"
    code_hash = research_code_content_hash(repo)
    empirical_bnd = empirical_boundary_vehicle_totals(flux_in, flux_out, dt_hours)

    payload: dict[str, Any] = {
        "status": overall_status,
        "protocol_role": "development_measured",
        "supersedes_note": (
            "Prior measured artifacts used R[:,0] shared inlet/outlet (constant corridor sum N); "
            "superseded by independent inlet/outlet channels R[:,0], R[:,1]."
        ),
        "limitations": [
            "NGSIM slice capped at 25000 rows; development-only — not independent measured validation or real OD.",
            "storage_count_identity is exact track bookkeeping (births/deaths) on the adapter grid, not proof of physical conservation.",
            "prefix_empirical with calibration_prefix_steps < sample windows falls back to physical 15k vph — often identical to physical_envelope on this pilot; see flow_assumption_scenarios.",
            "prefix_empirical z_upper can under-cap later windows (compare observed_peak_internal_vph).",
            "Declared per-lane capacity metadata is envelope metadata, not empirical calibration from these windows.",
            "No OD / route truth — boundary inference only on corridor chain.",
            "Edie q= k_vpkm * v_kmh is definitional, not fundamental-diagram validation.",
            "Temporal carry-in/out are accounting labels, not physical boundary sources.",
            "obs_tolerance_veh and R²-style storage fits on this slice are debug/development metrics only.",
        ],
        "provenance": prov.to_dict(),
        "research_code_hash": code_hash,
        "config_echo": cfg,
        "adapter_source_hash": adapter_module_hash(),
        "mask_seed": mask_seed,
        "split_times_ms": grid_meta.get("t_range_ms"),
        "units": prov.units,
        "coverage_flags": prov.coverage_flags,
        "grid_meta": grid_meta,
        "boundary_operators": {"n_bound": int(R.shape[1]), "inlet_channel": 0, "outlet_channel": 1},
        "empirical_boundary_flux_totals": empirical_bnd,
        "flow_capacity_metadata": flow_meta,
        "flow_assumption_scenarios": flow_scenarios,
        "calibration_status": calib_status,
        "internal_flow_bounds_status": z_calib_status,
        "internal_flow_upper_vph": float(np.max(z_upper_mat)) if z_upper_mat.size else all_lane_cap,
        "observed_peak_internal_vph": observed_peak_internal,
        "feasibility_lp_witness": feas_witness,
        "storage_count_identity": identity_audit,
        "sample_durations": {
            "n_snapshot_times": int(storage_snap.shape[0]),
            "n_intervals": int(Tm1_obs),
            "dt_ms": dt_ms,
            "wall_duration_s": prov.coverage_flags.get("duration_s"),
            "n_trajectories": prov.coverage_flags.get("n_trajectories"),
            "n_spatial_cells": n_cells,
        },
        "track_derived_observations": {
            "storage_T_plus_1_boundary_snapshots": True,
            "flux_interval_aligns_with_snapshot_deltas": True,
            "density_edie_separate_from_snapshot_fd": True,
        },
        "flux_and_tracking_residual": tracking,
        "snapshot_flux_fd_residual": snapshot_fd,
        "track_accounting_summary": ta_summary,
        "trace_accounting_discrepancy": {
            "unobserved_track_events": {
                "internal_appearances_cell_sum": ta_summary.get("internal_appearances"),
                "internal_disappearances_cell_sum": ta_summary.get("internal_disappearances"),
                "time_window_carry_in_sum": ta_summary.get("time_window_carry_in"),
                "time_window_carry_out_sum": ta_summary.get("time_window_carry_out"),
            },
            "note": "Appearances are not fed as estimator slack; boundary conservation uses snapshots and flux only.",
        },
        "inference_attempts": attempts,
        "numerical_failure_rate": failure_rate,
        "conditional_performance": primary["conditional_performance"],
        "i24_probe": i24_status,
        "no_od_evaluation": True,
    }

    _write_outputs(payload, aggregates, output_dir)
    return payload


def _summarize_track_accounting(aggregates: list[MacroCellAggregate]) -> dict[str, Any]:
    keys = [
        "internal_appearances",
        "internal_disappearances",
        "time_window_carry_in",
        "time_window_carry_out",
        "fragmentation_events",
        "dataset_boundary_births",
        "dataset_boundary_deaths",
        "negative_speed_samples",
        "stationary_samples",
    ]
    out: dict[str, Any] = {k: 0 for k in keys}
    for a in aggregates:
        tdict = a.track_accounting.to_dict()
        for k in keys:
            out[k] += int(tdict.get(k, 0))
    out["aggregation_scope"] = "summed_over_all_spacetime_cells_not_unique_vehicles"
    out["n_spacetime_cells"] = len(aggregates)
    return out


def _snapshot_flux_accounting_residual(
    storage: np.ndarray,
    cross_net_int: np.ndarray,
    flux_in: np.ndarray,
    flux_out: np.ndarray,
    dt_hours: float,
) -> dict[str, Any]:
    """N[t+1]-N[t] vs integer crossings (flux row t) and continuous-flux diagnostic."""
    if storage.shape[0] < 2 or cross_net_int.shape[0] < 1:
        return {"n_steps": 0, "mean_abs_residual_veh": float("nan")}
    int_residuals: list[float] = []
    flx_residuals: list[float] = []
    for t in range(storage.shape[0] - 1):
        for j in range(storage.shape[1]):
            d_n = float(storage[t + 1, j] - storage[t, j])
            int_residuals.append(d_n - float(cross_net_int[t, j]))
            net_cross = (flux_in[t, j] - flux_out[t, j]) * dt_hours
            flx_residuals.append(d_n - net_cross)
    arr = np.asarray(int_residuals, dtype=np.float64)
    arr_flx = np.asarray(flx_residuals, dtype=np.float64)
    return {
        "n_steps": int(arr.size),
        "mean_abs_residual_veh": float(np.mean(np.abs(arr))),
        "rmse_residual_veh": float(np.sqrt(np.mean(arr ** 2))),
        "max_abs_residual_veh": float(np.max(np.abs(arr))),
        "mean_abs_birth_death_excluded_veh": float(np.mean(np.abs(arr))),
        "continuous_flux_max_abs_veh": float(np.max(np.abs(arr_flx))),
        "interpretation": "integer residual excludes births/deaths; nonzero ⇒ track events or grid boundary",
    }


def _classify_inference_failure(
    est_status: str,
    primal_violation: float,
    obs_tol: float,
    snapshot_residual: dict[str, Any],
    *,
    lp_feasible: bool = True,
) -> str:
    if est_status == "optimal":
        return "ok"
    if not lp_feasible:
        return "infeasible_lp_polytope_witness"
    if primal_violation > obs_tol and snapshot_residual.get("max_abs_residual_veh", 0) > obs_tol:
        return "infeasible_observation_vs_conservation"
    if est_status == "numerical_failure":
        return "optimization_failure"
    return est_status


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _csv_value(v: Any) -> Any:
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    return v


def _write_outputs(payload: dict[str, Any], aggregates: list[MacroCellAggregate], output_dir: Path) -> None:
    (output_dir / "measured_pilot_results.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "macro_cell_aggregates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if not aggregates:
            return
        sample = aggregates[0].to_dict()
        flat_fields = [k for k in sample if k != "track_accounting"]
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for a in aggregates:
            row = a.to_dict()
            row.pop("track_accounting", None)
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CREST measured-data development pilot")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ngsim",
        type=Path,
        default=None,
        help="Override NGSIM CSV path (optional; else from config)",
    )
    args = parser.parse_args(argv)
    cfg = load_measured_config(args.config.expanduser().resolve())
    if args.ngsim is not None:
        cfg["ngsim_path"] = str(args.ngsim.relative_to(REPO_ROOT)) if args.ngsim.is_relative() else str(args.ngsim)

    try:
        payload = run_measured_pilot(cfg, args.output.expanduser().resolve())
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"status": payload.get("status"), "output": str(args.output)}, indent=2))
    return 0 if payload.get("status") != "insufficient_evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
