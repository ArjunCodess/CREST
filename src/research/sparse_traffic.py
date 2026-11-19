"""Robust traffic estimation and assumption-conditional bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

TOLERANCE = 1e-5
Status = Literal["optimal", "infeasible", "unbounded", "timeout", "numerical_failure"]


@dataclass(frozen=True)
class Constraints:
    equality: sparse.csc_matrix
    rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class Estimate:
    status: Status
    values: np.ndarray | None
    balance_error: float | None = None
    bound_violation: float | None = None
    solver_message: str = ""


@dataclass(frozen=True)
class Interval:
    status: Status
    lower: float | None = None
    upper: float | None = None


def conservation_system(
    incidence: sparse.spmatrix | np.ndarray,
    initial_storage: np.ndarray,
    capacity: np.ndarray,
    flow_lower: np.ndarray,
    flow_upper: np.ndarray,
    dt_hours: float,
) -> Constraints:
    """Store each variable locally as [storage at T+1 times, flux over T intervals]."""
    graph = sparse.csc_matrix(incidence, dtype=float)
    cells, channels = graph.shape
    lower, upper = np.asarray(flow_lower, dtype=float), np.asarray(flow_upper, dtype=float)
    initial, cap = np.asarray(initial_storage, dtype=float), np.asarray(capacity, dtype=float)
    if lower.ndim != 2 or not lower.shape[0] or lower.shape != upper.shape or lower.shape[1] != channels:
        raise ValueError("Flow bounds must cover each interval and flux channel")
    if initial.shape != (cells,) or cap.shape != (cells,) or np.any(initial > cap) or np.any(initial < 0):
        raise ValueError("Initial storage must fit each cell capacity")
    if (not np.isfinite(dt_hours) or dt_hours <= 0 or not np.isfinite(graph.data).all()
            or not np.isfinite(initial).all() or np.isnan(cap).any()
            or not np.isfinite(lower).all() or np.isnan(upper).any() or np.any(lower < 0) or np.any(lower > upper)):
        raise ValueError("Invalid conservation coefficients or bounds")
    steps = lower.shape[0]
    difference = sparse.diags((-np.ones(steps), np.ones(steps)), (0, 1), shape=(steps, steps + 1), format="csc")
    equality = sparse.hstack((
        sparse.kron(difference, sparse.eye(cells), format="csc"),
        -dt_hours * sparse.kron(sparse.eye(steps), graph, format="csc"),
    ), format="csc")
    return Constraints(equality, np.zeros(steps * cells),
                       np.r_[initial, np.zeros(steps * cells), lower.ravel()],
                       np.r_[initial, np.tile(cap, steps), upper.ravel()])


def observation_rows(indices: np.ndarray, size: int) -> sparse.csc_matrix:
    return sparse.csc_matrix((np.ones(len(indices)), (np.arange(len(indices)), indices)), shape=(len(indices), size))


def robust_fit(
    system: Constraints,
    prior: np.ndarray,
    observed_indices: np.ndarray,
    observations: np.ndarray,
    *,
    prior_weight: float | np.ndarray = 1.0,
    huber_delta: float = 3.0,
    smoothness: sparse.spmatrix | None = None,
    time_limit: float = 10.0,
) -> Estimate:
    """Solve the Huber loss with a quadratic residual and positive/negative excess."""
    import osqp

    n = len(system.lower)
    prior, indices, values = np.asarray(prior), np.asarray(observed_indices), np.asarray(observations)
    weights = np.broadcast_to(prior_weight, (n,))
    if (prior.shape != (n,) or not np.isfinite(prior).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or indices.shape != values.shape or np.any(indices < 0) or np.any(indices >= n)
            or not np.isfinite(values).all() or not np.isfinite(huber_delta) or huber_delta <= 0
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("Invalid point estimation inputs")
    m = len(indices)
    base = sparse.diags(weights, format="csc")
    if smoothness is not None:
        base = base + smoothness.T @ smoothness
    penalty = sparse.block_diag((base, sparse.eye(m), sparse.csc_matrix((2 * m, 2 * m))), format="csc")
    objective = np.r_[-weights * prior, np.zeros(m), np.full(2 * m, huber_delta)]
    observation = sparse.hstack((observation_rows(indices, n), -sparse.eye(m), -sparse.eye(m), sparse.eye(m)))
    balanced = sparse.hstack((system.equality, sparse.csc_matrix((len(system.rhs), 3 * m))))
    constraint = sparse.vstack((balanced, observation, sparse.eye(n + 3 * m)), format="csc")
    lo = np.r_[system.rhs, values, system.lower, np.full(m, -np.inf), np.zeros(2 * m)]
    hi = np.r_[system.rhs, values, system.upper, np.full(3 * m, np.inf)]
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(penalty, format="csc"), q=objective, A=constraint, l=lo, u=hi,
                 verbose=False, polishing=False, eps_abs=1e-7, eps_rel=1e-8, max_iter=30_000, time_limit=time_limit)
    solved = solver.solve(raise_error=False)
    status = solved.info.status_val
    if status in (3, 4):
        return Estimate("infeasible", None, solver_message=solved.info.status)
    if status in (5, 6):
        return Estimate("unbounded", None, solver_message=solved.info.status)
    if status == 8:
        return Estimate("timeout", None, solver_message=solved.info.status)
    if status not in (1, 2) or solved.x is None or not np.isfinite(solved.x).all():
        return Estimate("numerical_failure", None, solver_message=solved.info.status)
    x = solved.x[:n]
    balance = float(np.max(np.abs(system.equality @ x - system.rhs), initial=0))
    violation = float(max(0, np.max(system.lower - x), np.max(x - system.upper)))
    valid = balance <= TOLERANCE and violation <= TOLERANCE
    return Estimate("optimal" if valid else "numerical_failure", x if valid else None, balance, violation, solved.info.status)


def conditional_interval(
    system: Constraints,
    objective: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    tolerance: float | np.ndarray,
    *,
    time_limit: float = 10.0,
) -> Interval:
    indices, values = np.asarray(indices), np.asarray(values)
    widths = np.broadcast_to(tolerance, values.shape)
    if (objective.shape != system.lower.shape or not np.isfinite(objective).all()
            or indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.shape != values.shape
            or np.any(indices < 0) or np.any(indices >= len(objective)) or not np.isfinite(values).all()
            or not np.isfinite(widths).all() or np.any(widths < 0)
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("Invalid interval query")
    observed = observation_rows(indices, len(objective))
    a_ub = sparse.vstack((observed, -observed), format="csc") if len(indices) else None
    b_ub = np.r_[values + widths, widths - values] if len(indices) else None
    bounds = np.column_stack((system.lower, system.upper))
    extrema = []
    for sign in (1, -1):
        result = linprog(sign * objective, A_ub=a_ub, b_ub=b_ub, A_eq=system.equality, b_eq=system.rhs,
                         bounds=bounds, method="highs", options={"time_limit": time_limit})
        if not result.success:
            status = {1: "timeout", 2: "infeasible", 3: "unbounded"}.get(result.status, "numerical_failure")
            return Interval(status)
        extrema.append(float(sign * result.fun))
    return Interval("optimal", *extrema)
