"""Deployable and oracle baselines for controlled synthetic experiments."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .boundary_conservation import ConservationEstimate, fit_boundary_conservation, pack_state
from .partial_identification import aggregate_bounds


def masked_mean_imputer(obs_values: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(obs_values, dtype=np.float64)
    mask = np.asarray(obs_mask, dtype=bool)
    fill = np.nanmean(x, axis=0, keepdims=True)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    out = x.copy()
    out[~mask] = np.broadcast_to(fill, x.shape)[~mask]
    return out


def unconstrained_forward_fill(obs_values: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(obs_values, dtype=np.float64).copy()
    mask = np.asarray(obs_mask, dtype=bool)
    for i in range(out.shape[1]):
        last = np.nan
        for t in range(out.shape[0]):
            if mask[t, i] and np.isfinite(out[t, i]):
                last = out[t, i]
            elif last is not np.nan:
                out[t, i] = last
    out[~mask & np.isnan(out)] = 0.0
    return out


def oracle_exact_boundary(
    B,
    R,
    dt_hours,
    n0,
    z_true,
    u_true,
) -> ConservationEstimate:
    """Diagnostic only: uses generative boundary truth (not deployable)."""
    storage = pack_state(z_true, u_true, n0, B, R, dt_hours)
    mass = storage[1:] - storage[:-1] - dt_hours * (z_true @ B.T + u_true @ R.T)
    return ConservationEstimate(
        storage=storage,
        internal_flow=z_true,
        boundary_flow=u_true,
        status='oracle',
        objective=0.0,
        observation_term=0.0,
        prior_term=0.0,
        dynamics_algebraic_residual_rmse=float(np.sqrt(np.mean(mass ** 2))),
        primal_feasibility_max_violation=0.0,
        stationarity_inf_norm=0.0,
        message='oracle diagnostic',
    )


def assumed_exact_boundary_midpoint(
    B,
    R,
    dt_hours,
    n0,
    z_seed: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
) -> ConservationEstimate:
    """Deployable baseline: treats interval midpoint as known boundary (can be wrong)."""
    u_mid = 0.5 * (u_lower + u_upper)
    z = np.maximum(z_seed, 0.0)
    storage = pack_state(z, u_mid, n0, B, R, dt_hours)
    mass = storage[1:] - storage[:-1] - dt_hours * (z @ B.T + u_mid @ R.T)
    return ConservationEstimate(
        storage=storage,
        internal_flow=z,
        boundary_flow=u_mid,
        status='assumed_midpoint',
        objective=0.0,
        observation_term=0.0,
        prior_term=0.0,
        dynamics_algebraic_residual_rmse=float(np.sqrt(np.mean(mass ** 2))),
        primal_feasibility_max_violation=0.0,
        stationarity_inf_norm=0.0,
        message='midpoint boundary projection',
    )


def bounded_boundary_estimate(scenario, **kwargs) -> ConservationEstimate:
    z_upper = scenario.metadata.get('max_edge_flow_vph', scenario.metadata['demand_vph'] * 1.25)
    z0 = np.full_like(scenario.z_true, 0.5 * scenario.metadata['demand_vph'])
    obs_tol = scenario.metadata.get('obs_tolerance', 25.0)
    return fit_boundary_conservation(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.obs_mask,
        np.nan_to_num(scenario.obs_values, nan=0.0),
        scenario.u_prior,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        z0=z0,
        z_upper=z_upper,
        capacity=scenario.capacity,
        obs_tolerance=obs_tol,
        w_obs=0.0,
        **kwargs,
    )


def interval_baseline_aggregate(u_lower: np.ndarray, u_upper: np.ndarray, dt_hours: float) -> dict:
    """Box-only intervals for inflow, outflow, and net boundary vehicles (no observation fit)."""
    u_lower = np.asarray(u_lower, dtype=np.float64)
    u_upper = np.asarray(u_upper, dtype=np.float64)
    Tm1, n_bound = u_lower.shape
    in_lo = float(np.sum(u_lower[:, 0]) * dt_hours) if n_bound >= 1 else 0.0
    in_hi = float(np.sum(u_upper[:, 0]) * dt_hours) if n_bound >= 1 else 0.0
    out_lo = float(np.sum(u_lower[:, 1]) * dt_hours) if n_bound >= 2 else 0.0
    out_hi = float(np.sum(u_upper[:, 1]) * dt_hours) if n_bound >= 2 else 0.0
    net_lo = in_lo - out_hi
    net_hi = in_hi - out_lo
    return {
        'status': 'interval_only',
        'inflow': {'lower': in_lo, 'upper': in_hi, 'width': in_hi - in_lo},
        'outflow': {'lower': out_lo, 'upper': out_hi, 'width': out_hi - out_lo},
        'net': {'lower': net_lo, 'upper': net_hi, 'width': net_hi - net_lo},
    }


def boundary_truth_vehicle_totals(u_true: np.ndarray, dt_hours: float) -> dict[str, float]:
    u_true = np.asarray(u_true, dtype=np.float64)
    inflow = float(np.sum(u_true[:, 0]) * dt_hours) if u_true.shape[1] >= 1 else 0.0
    outflow = float(np.sum(u_true[:, 1]) * dt_hours) if u_true.shape[1] >= 2 else 0.0
    return {
        'inflow_vehicles': inflow,
        'outflow_vehicles': outflow,
        'net_inflow_vehicles': inflow - outflow,
    }


def od_nnls_baseline(incidence: np.ndarray, count_center: np.ndarray, prior: np.ndarray) -> np.ndarray:
    A = np.asarray(incidence, dtype=np.float64)
    c = np.asarray(count_center, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    x, _ = nnls(np.vstack([A, 0.1 * np.eye(A.shape[1])]), np.concatenate([c, 0.1 * p]))
    return x


def od_lp_bounds_baseline(
    incidence,
    count_lower,
    count_upper,
    weights,
    prior,
    *,
    observed_mask=None,
    path_flow_cap=None,
):
    center = 0.5 * (np.nan_to_num(count_lower, 0) + np.nan_to_num(count_upper, 0))
    flows = od_nnls_baseline(incidence, center, prior)
    bounds = aggregate_bounds(
        incidence,
        count_lower,
        count_upper,
        weights,
        observed_mask=observed_mask,
        flow_upper=path_flow_cap,
    )
    return {'flows_point': flows, 'aggregate_bounds': bounds}
