"""Evaluation-only checks: generative truth vs stated polytopes (never used for fitting)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .conservation_constraints import assemble_conservation_polytope, evaluate_affine_constraints
from .partial_identification import aggregate_bounds, flow_interval_feasibility


def truth_in_boundary_box(
    u_true: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    *,
    tol: float = 1e-9,
) -> bool:
    u_true = np.asarray(u_true, dtype=np.float64)
    return bool(
        np.all(u_true >= u_lower - tol) and np.all(u_true <= u_upper + tol)
    )


def truth_in_observation_tubes(
    storage_true: np.ndarray,
    obs_mask: np.ndarray,
    obs_values: np.ndarray,
    obs_tolerance: float | np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[bool, float]:
    """True if every observed cell's generative storage lies inside the declared tube."""
    stor = np.asarray(storage_true, dtype=np.float64)
    mask = np.asarray(obs_mask, dtype=bool)
    obs = np.asarray(obs_values, dtype=np.float64)
    tol_arr = np.broadcast_to(np.asarray(obs_tolerance, dtype=np.float64), stor.shape)
    max_excess = 0.0
    ok = True
    for t in range(stor.shape[0]):
        for ell in range(stor.shape[1]):
            if not mask[t, ell] or not np.isfinite(obs[t, ell]):
                continue
            err = abs(float(stor[t, ell] - obs[t, ell]))
            tube = float(tol_arr[t, ell])
            excess = max(0.0, err - tube)
            max_excess = max(max_excess, excess)
            if excess > tol:
                ok = False
    return ok, float(max_excess)


def truth_in_od_counts(
    incidence: np.ndarray,
    path_flow_true: np.ndarray,
    count_lower: np.ndarray,
    count_upper: np.ndarray,
    observed_mask: np.ndarray,
    *,
    tol: float = 1e-9,
) -> bool:
    loaded = np.asarray(incidence, dtype=np.float64) @ np.asarray(path_flow_true, dtype=np.float64).ravel()
    lo = np.asarray(count_lower, dtype=np.float64).ravel()
    hi = np.asarray(count_upper, dtype=np.float64).ravel()
    mask = np.asarray(observed_mask, dtype=bool).ravel()
    for i in range(loaded.size):
        if not mask[i] or not np.isfinite(lo[i]) or not np.isfinite(hi[i]):
            continue
        if loaded[i] < lo[i] - tol or loaded[i] > hi[i] + tol:
            return False
    return True


def truth_within_path_caps(path_flow_true: np.ndarray, path_flow_cap: np.ndarray, *, tol: float = 1e-9) -> bool:
    f = np.asarray(path_flow_true, dtype=np.float64).ravel()
    cap = np.asarray(path_flow_cap, dtype=np.float64).ravel()
    return bool(np.all(f <= cap + tol))


def evaluate_truth_feasibility(scenario) -> dict[str, Any]:
    """Flags for whether generative truth satisfies each stated constraint class."""
    caps = getattr(scenario, 'path_flow_cap', None)
    obs_tol = scenario.metadata.get('obs_tolerance', 0.0)
    obs_ok, obs_excess = truth_in_observation_tubes(
        scenario.storage_true,
        scenario.obs_mask,
        scenario.obs_values,
        obs_tol,
    )
    od_counts_ok = truth_in_od_counts(
        scenario.incidence,
        scenario.path_flow_true,
        scenario.count_lower,
        scenario.count_upper,
        scenario.count_observed_mask,
    )
    caps_ok = True if caps is None else truth_within_path_caps(scenario.path_flow_true, caps)
    flow_lo = np.zeros(scenario.incidence.shape[1])
    flow_hi = caps if caps is not None else np.full(scenario.incidence.shape[1], np.inf)
    od_feas = flow_interval_feasibility(
        scenario.incidence,
        flow_lo,
        flow_hi,
        scenario.count_lower,
        scenario.count_upper,
        observed_mask=scenario.count_observed_mask,
    )
    poly = assemble_conservation_polytope(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        obs_mask=scenario.obs_mask,
        obs_values=np.nan_to_num(scenario.obs_values, nan=0.0),
        obs_tolerance=obs_tol,
        z_upper=scenario.metadata.get('max_edge_flow_vph', np.inf),
        capacity=scenario.capacity,
    )
    x_truth = np.concatenate([scenario.z_true.ravel(), scenario.u_true.ravel()])
    aff = evaluate_affine_constraints(x_truth, poly.A_ub, poly.b_ub)
    box_viol = float(np.max(np.maximum(poly.lower - x_truth, x_truth - poly.upper)))
    truth_in_affine = bool(aff['feasible'] and box_viol <= 1e-5)
    truth_in_capacity = scenario.truth_respects_capacity()
    boundary_ok = scenario.boundary_truth_in_interval()

    pipes = {
        'boundary_box': boundary_ok,
        'storage_capacity': truth_in_capacity,
        'observation_tubes': obs_ok,
        'observation_tube_max_excess_veh': obs_excess,
        'od_counts': od_counts_ok,
        'path_caps': caps_ok,
        'od_polytope_nonempty': bool(od_feas.get('feasible')),
        'conservation_point_polytope': truth_in_affine,
        'joint_within_assumption': (
            boundary_ok
            and truth_in_capacity
            and obs_ok
            and od_counts_ok
            and caps_ok
            and truth_in_affine
            and bool(od_feas.get('feasible'))
        ),
    }
    return {
        'flags': pipes,
        'od_feasibility_status': od_feas.get('status'),
        'od_feasibility_message': od_feas.get('message'),
        'truth_witness_path_flow_cap': None if caps is None else caps.tolist(),
        'truth_witness_total_flow': float(np.sum(scenario.path_flow_true)),
    }


def classify_bounded_lp_status(
    scenario,
    optimizer_status: str,
    primal_violation: float | None,
    *,
    primal_tol: float = 1e-5,
) -> dict[str, Any]:
    """Separate generative-truth certificate from optimizer / initial-candidate outcome."""
    eval_truth = evaluate_truth_feasibility(scenario)
    joint = eval_truth['flags']['joint_within_assumption']
    poly_ok = eval_truth['flags']['conservation_point_polytope']
    out: dict[str, Any] = {
        'eval_truth_feasibility': eval_truth['flags'],
        'optimizer_status': optimizer_status,
        'initial_truth_in_polytope': poly_ok,
        'initial_truth_joint_within_assumption': joint,
    }
    if optimizer_status == 'optimal':
        viol = primal_violation if primal_violation is not None else float('inf')
        out['lp_status_reason'] = 'optimal' if viol <= primal_tol else 'numerical_failure_primal_residual'
        return out
    if optimizer_status == 'infeasible_polytope':
        if poly_ok:
            out['lp_status_reason'] = 'infeasible_polytope_initial_candidate_only'
        elif not eval_truth['flags']['observation_tubes']:
            out['lp_status_reason'] = 'infeasible_polytope_observation_tube_excludes_truth'
        elif not eval_truth['flags']['boundary_box']:
            out['lp_status_reason'] = 'infeasible_polytope_boundary_excludes_truth'
        else:
            out['lp_status_reason'] = 'infeasible_polytope'
        return out
    if optimizer_status == 'numerical_failure':
        out['lp_status_reason'] = 'numerical_failure'
        return out
    out['lp_status_reason'] = optimizer_status
    return out


def classify_od_lp_status(
    scenario,
    aggregate_status: str,
    *,
    message: str | None = None,
) -> str:
    """Explain OD aggregate LP status without attributing rank deficiency to infeasibility."""
    eval_truth = evaluate_truth_feasibility(scenario)
    caps = getattr(scenario, 'path_flow_cap', None)
    if aggregate_status == 'optimal':
        return 'optimal'
    if aggregate_status == 'no_count_constraints':
        return 'no_observed_count_rows'
    if aggregate_status == 'unbounded':
        if caps is None or not np.isfinite(caps).all():
            return 'unbounded_missing_path_caps'
        return 'unbounded_caps_too_loose'
    if aggregate_status == 'infeasible':
        if scenario.metadata.get('inconsistent_od_counts'):
            return 'infeasible_deliberate_count_stress'
        if not eval_truth['flags']['od_counts']:
            return 'infeasible_counts_exclude_generative_truth'
        if not eval_truth['flags']['od_polytope_nonempty']:
            return 'infeasible_contradictory_count_intervals'
        return 'infeasible_polytope'
    if aggregate_status == 'numerical_failure':
        return 'numerical_failure'
    return 'infeasible_polytope' if not message else f'{aggregate_status}:{message}'
