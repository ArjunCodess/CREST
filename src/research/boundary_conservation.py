"""Discrete conservation with bounded boundary flows and shared data constraints.

Model (vehicles, dt in hours):
    N[t+1] = N[t] + dt * (B @ z[t] + R @ u[t])

B maps internal edge fluxes (vph) to net storage change; R maps boundary fluxes.
Storage at observed times is affine in stacked (z, u); LPs and hard inequalities use the same assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .conservation_constraints import (
    assemble_conservation_polytope,
    boundary_aggregate_objective,
    boundary_aggregate_weights,
    evaluate_affine_constraints,
)
from .lp_core import linprog_extremum, linprog_interval


@dataclass
class ConservationEstimate:
    storage: np.ndarray  # (T, n_links) vehicles
    internal_flow: np.ndarray  # (T-1, n_edges) vph
    boundary_flow: np.ndarray  # (T-1, n_boundary) vph
    status: str
    objective: float
    observation_term: float
    prior_term: float
    dynamics_algebraic_residual_rmse: float
    primal_feasibility_max_violation: float
    stationarity_inf_norm: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'objective': self.objective,
            'observation_term': self.observation_term,
            'prior_term': self.prior_term,
            'dynamics_algebraic_residual_rmse': self.dynamics_algebraic_residual_rmse,
            'primal_feasibility_max_violation': self.primal_feasibility_max_violation,
            'stationarity_inf_norm': self.stationarity_inf_norm,
            'message': self.message,
        }


def pack_state(z: np.ndarray, u: np.ndarray, n0: np.ndarray, B: np.ndarray, R: np.ndarray, dt: float) -> np.ndarray:
    """Integrate dynamics forward from initial storage n0 (vehicles)."""
    Tm1, n_edge = z.shape
    storage = np.zeros((Tm1 + 1, B.shape[0]))
    storage[0] = n0
    for t in range(Tm1):
        storage[t + 1] = storage[t] + dt * (B @ z[t] + R @ u[t])
    return storage


def unpack_state_vector(x: np.ndarray, Tm1: int, n_edge: int, n_bound: int) -> tuple[np.ndarray, np.ndarray]:
    n_z = Tm1 * n_edge
    z = x[:n_z].reshape(Tm1, n_edge)
    u = x[n_z:].reshape(Tm1, n_bound)
    return z, u


def _observation_misfit(
    x: np.ndarray,
    poly,
    obs_mask: np.ndarray,
    obs_values: np.ndarray,
) -> float:
    misfit = 0.0
    n_obs = 0
    for t in range(poly.Tm1 + 1):
        for ell in range(poly.n_links):
            if not obs_mask[t, ell] or not np.isfinite(obs_values[t, ell]):
                continue
            row, const = poly.storage_row(t, ell)
            pred = const + float(row @ x)
            misfit += (pred - obs_values[t, ell]) ** 2
            n_obs += 1
    return float(misfit / max(n_obs, 1))


PRIMAL_FEAS_TOL = 1e-5


def _flow_scale(poly) -> float:
    z_scale = float(np.max(np.abs(poly.upper[: poly.n_z]))) if poly.n_z else 1.0
    u_scale = float(np.max(np.abs(poly.upper[poly.n_z :]))) if poly.n_u else 1.0
    return max(1.0, z_scale, u_scale)


def _feasible_polytope_start(poly, u_prior: np.ndarray) -> np.ndarray | None:
    """Chebyshev-center proxy: min L1 slack toward prior inside the data polytope."""
    n = poly.n_vars
    u_prior_flat = u_prior.ravel()
    c = np.zeros(n)
    if u_prior_flat.size == n - poly.n_z:
        c[poly.n_z :] = (u_prior_flat - 0.5 * (poly.lower[poly.n_z :] + poly.upper[poly.n_z :])) / max(_flow_scale(poly), 1.0)
    res = linprog_extremum(
        c,
        A_ub=poly.A_ub,
        b_ub=poly.b_ub,
        lower=poly.lower,
        upper=poly.upper,
        sense='min',
    )
    if res.status != 'optimal' or res.primal is None:
        return None
    feas = evaluate_affine_constraints(res.primal, poly.A_ub, poly.b_ub)
    box_viol = float(np.max(np.maximum(poly.lower - res.primal, res.primal - poly.upper)))
    if feas['feasible'] and box_viol <= PRIMAL_FEAS_TOL:
        return res.primal
    return None


def _optimizer_constraint_violation(result) -> float:
    viol = getattr(result, 'violation', None)
    if viol is None:
        return 0.0
    if isinstance(viol, (int, float, np.floating)):
        return float(viol)
    if isinstance(viol, dict):
        return float(max(float(v) for v in viol.values())) if viol else 0.0
    if isinstance(viol, (list, tuple, np.ndarray)):
        arr = np.asarray(viol, dtype=np.float64)
        return float(np.max(arr)) if arr.size else 0.0
    return 0.0


def _check_stationarity_qp(x: np.ndarray, grad: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-5) -> float:
    """Diagnostic only (box-projected gradient); not a KKT certificate with linear inequalities."""
    proj = x.copy()
    for i in range(x.size):
        if lower[i] <= x[i] <= upper[i]:
            if grad[i] < 0:
                proj[i] = lower[i]
            elif grad[i] > 0:
                proj[i] = upper[i]
    return float(np.linalg.norm(x - proj, ord=np.inf))


def fit_boundary_conservation(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    obs_mask: np.ndarray,
    obs_values: np.ndarray,
    u_prior: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    *,
    w_obs: float = 0.0,
    w_prior: float = 0.1,
    z_upper: float | np.ndarray | None = None,
    z0: np.ndarray | None = None,
    capacity: np.ndarray | None = None,
    obs_tolerance: float | np.ndarray | None = None,
    use_hard_observation_constraints: bool = True,
) -> ConservationEstimate:
    """Point estimate: minimize boundary prior subject to shared data polytope.

    Observations enter as hard linear inequalities when use_hard_observation_constraints is True
    (same assembly as conservation_lp_extrema). Optional w_obs adds a soft diagnostic misfit term
  on observed cells but does not replace hard constraints.
    """
    B = np.asarray(B, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_edge = B.shape[1]
    n_bound = R.shape[1]
    Tm1 = u_prior.shape[0]

    if z_upper is None:
        z_upper = np.full((Tm1, n_edge), np.inf)
    poly = assemble_conservation_polytope(
        B,
        R,
        dt_hours,
        n0,
        u_lower,
        u_upper,
        obs_mask=obs_mask,
        obs_values=obs_values,
        obs_tolerance=obs_tolerance,
        z_upper=z_upper,
        capacity=capacity,
    )

    if z0 is None:
        z0 = np.zeros((Tm1, n_edge))
    u_prior_flat = u_prior.ravel()
    x0 = _feasible_polytope_start(poly, u_prior)
    if x0 is None:
        x0 = np.concatenate([np.maximum(z0, 0.0).ravel(), u_prior.ravel()])
        x0 = np.clip(x0, poly.lower, poly.upper)
        feas0 = evaluate_affine_constraints(x0, poly.A_ub, poly.b_ub)
        box0 = float(np.max(np.maximum(poly.lower - x0, x0 - poly.upper)))
        if poly.A_ub is not None and (not feas0['feasible'] or box0 > PRIMAL_FEAS_TOL):
            return ConservationEstimate(
                storage=pack_state(np.maximum(z0, 0.0), u_prior, n0, B, R, dt_hours),
                internal_flow=np.maximum(z0, 0.0),
                boundary_flow=u_prior,
                status='infeasible_polytope',
                objective=float('nan'),
                observation_term=0.0,
                prior_term=0.0,
                dynamics_algebraic_residual_rmse=float('nan'),
                primal_feasibility_max_violation=max(feas0['max_violation'], max(0.0, box0)),
                stationarity_inf_norm=float('nan'),
                message='initial candidate outside hard observation polytope (not a HiGHS infeasibility certificate)',
            )

    n_u = Tm1 * n_bound

    def objective(x: np.ndarray) -> float:
        u = x[poly.n_z :]
        prior_term = float(np.sum((u - u_prior_flat) ** 2))
        soft_obs = _observation_misfit(x, poly, obs_mask, obs_values) if w_obs > 0 else 0.0
        return w_prior * prior_term + w_obs * soft_obs

    def gradient(x: np.ndarray) -> np.ndarray:
        g = np.zeros_like(x)
        u = x[poly.n_z :]
        if w_prior > 0:
            g[poly.n_z :] = 2.0 * w_prior * (u - u_prior_flat)
        if w_obs > 0:
            for t in range(poly.Tm1 + 1):
                for ell in range(poly.n_links):
                    if not obs_mask[t, ell] or not np.isfinite(obs_values[t, ell]):
                        continue
                    row, const = poly.storage_row(t, ell)
                    pred = const + float(row @ x)
                    g += (2.0 * w_obs / max(int(np.sum(obs_mask & np.isfinite(obs_values))), 1)) * (pred - obs_values[t, ell]) * row
        return g

    lin_constraints = []
    if use_hard_observation_constraints and poly.A_ub is not None:
        lin_constraints.append(
            LinearConstraint(poly.A_ub, -np.inf * np.ones(poly.A_ub.shape[0]), poly.b_ub)
        )

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='trust-constr' if lin_constraints else 'L-BFGS-B',
        bounds=list(zip(poly.lower, poly.upper, strict=True)),
        constraints=lin_constraints,
        options={'maxiter': 2000, 'gtol': 1e-7, 'xtol': 1e-10},
    )

    z_hat, u_hat = unpack_state_vector(result.x, Tm1, n_edge, n_bound)
    storage_hat = pack_state(z_hat, u_hat, n0, B, R, dt_hours)

    feas = evaluate_affine_constraints(result.x, poly.A_ub, poly.b_ub)
    box_viol = float(np.max(np.maximum(poly.lower - result.x, result.x - poly.upper)))
    primal_viol = max(feas['max_violation'], max(0.0, box_viol))

    grad_at = gradient(result.x)
    stat_norm = _check_stationarity_qp(result.x, grad_at, poly.lower, poly.upper)

    prior_term = float(np.sum((u_hat.ravel() - u_prior_flat) ** 2))
    obs_term = _observation_misfit(result.x, poly, obs_mask, obs_values)

    mass = storage_hat[1:] - storage_hat[:-1] - dt_hours * (z_hat @ B.T + u_hat @ R.T)
    dyn_algebraic = float(np.sqrt(np.mean(mass ** 2)))

    constr_viol = _optimizer_constraint_violation(result)
    optimal = (
        bool(result.success)
        and primal_viol <= PRIMAL_FEAS_TOL
        and constr_viol <= PRIMAL_FEAS_TOL
    )
    status = 'optimal' if optimal else 'numerical_failure'

    return ConservationEstimate(
        storage=storage_hat,
        internal_flow=z_hat,
        boundary_flow=u_hat,
        status=status,
        objective=float(result.fun),
        observation_term=obs_term,
        prior_term=prior_term,
        dynamics_algebraic_residual_rmse=dyn_algebraic,
        primal_feasibility_max_violation=primal_viol,
        stationarity_inf_norm=stat_norm,
        message=result.message or status,
    )


def conservation_lp_extrema(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    aggregate_weights: np.ndarray,
    *,
    obs_mask: np.ndarray | None = None,
    obs_values: np.ndarray | None = None,
    obs_tolerance: float | np.ndarray | None = None,
    z_upper: float | np.ndarray | None = None,
    capacity: np.ndarray | None = None,
    include_observations: bool = True,
    vehicles_aggregate: bool = True,
) -> dict[str, Any]:
    """LP bounds on a linear aggregate over the data-conditioned polytope (not box-only)."""
    if z_upper is None:
        Tm1 = u_lower.shape[0]
        n_edge = B.shape[1]
        z_upper = np.full((Tm1, n_edge), np.inf)

    Tm1 = u_lower.shape[0]
    n_links = B.shape[0]
    shape = (Tm1 + 1, n_links)
    if obs_mask is None:
        obs_mask = np.zeros(shape, dtype=bool)
    if obs_values is None:
        obs_values = np.full(shape, np.nan)
    mask = obs_mask if include_observations else np.zeros(shape, dtype=bool)
    vals = obs_values if include_observations else np.full(shape, np.nan)

    poly = assemble_conservation_polytope(
        B,
        R,
        dt_hours,
        n0,
        u_lower,
        u_upper,
        obs_mask=mask,
        obs_values=vals,
        obs_tolerance=obs_tolerance if include_observations else 0.0,
        z_upper=z_upper,
        capacity=capacity,
    )

    c = boundary_aggregate_objective(poly, aggregate_weights, vehicles_not_vph=vehicles_aggregate)

    lo_res, hi_res = linprog_interval(
        c,
        A_ub=poly.A_ub,
        b_ub=poly.b_ub,
        lower=poly.lower,
        upper=poly.upper,
    )
    meta = {
        'n_observation_rows': poly.observation_constraints_applied(),
        'vehicles_aggregate': vehicles_aggregate,
        'include_observations': include_observations,
    }
    if lo_res.status != 'optimal' or hi_res.status != 'optimal':
        st = lo_res.status if lo_res.status != 'optimal' else hi_res.status
        return {'status': st, 'lower': None, 'upper': None, 'message': lo_res.message, **meta}
    return {
        'status': 'optimal',
        'lower': lo_res.value,
        'upper': hi_res.value,
        'width': float(hi_res.value - lo_res.value),
        'message': 'ok',
        **meta,
    }


def boundary_aggregate_lp_bundle(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    *,
    obs_mask: np.ndarray | None = None,
    obs_values: np.ndarray | None = None,
    obs_tolerance: float | np.ndarray | None = None,
    z_upper: float | np.ndarray | None = None,
    capacity: np.ndarray | None = None,
    include_observations: bool = True,
) -> dict[str, Any]:
    """Separate LP intervals for cumulative inflow, outflow, and net boundary vehicles."""
    Tm1 = u_lower.shape[0]
    n_bound = R.shape[1]
    modes = ('inflow', 'outflow', 'net') if n_bound >= 2 else ('inflow',)
    out: dict[str, Any] = {}
    for mode in modes:
        w = boundary_aggregate_weights(Tm1, n_bound, mode=mode)
        out[mode] = conservation_lp_extrema(
            B,
            R,
            dt_hours,
            n0,
            u_lower,
            u_upper,
            w,
            obs_mask=obs_mask,
            obs_values=obs_values,
            obs_tolerance=obs_tolerance,
            z_upper=z_upper,
            capacity=capacity,
            include_observations=include_observations,
            vehicles_aggregate=True,
        )
    return out


def conservation_lp_box_only(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    n0: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    aggregate_weights: np.ndarray,
    *,
    z_upper: float | np.ndarray | None = None,
    capacity: np.ndarray | None = None,
    vehicles_aggregate: bool = True,
) -> dict[str, Any]:
    """Reference: interval bounds on u and z only, no storage observations."""
    return conservation_lp_extrema(
        B,
        R,
        dt_hours,
        n0,
        u_lower,
        u_upper,
        aggregate_weights,
        obs_mask=np.zeros((u_lower.shape[0] + 1, B.shape[0]), dtype=bool),
        obs_values=np.zeros((u_lower.shape[0] + 1, B.shape[0])),
        z_upper=z_upper,
        capacity=capacity,
        include_observations=False,
        vehicles_aggregate=vehicles_aggregate,
    )
