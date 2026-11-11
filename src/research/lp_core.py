"""Robust LP helpers; solver failure never becomes a numeric estimate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linprog

SolverStatus = Literal['optimal', 'infeasible', 'unbounded', 'numerical_failure', 'no_objective']


@dataclass(frozen=True)
class LPResult:
    status: SolverStatus
    value: float | None
    primal: np.ndarray | None
    message: str


def _prepare_bounds(
    n: int,
    lower: np.ndarray | float | None,
    upper: np.ndarray | float | None,
) -> list[tuple[float, float]]:
    if lower is None:
        lower = 0.0
    if upper is None:
        upper = np.inf
    low = np.broadcast_to(np.asarray(lower, dtype=np.float64), (n,))
    up = np.broadcast_to(np.asarray(upper, dtype=np.float64), (n,))
    return [(float(lo), float(hi)) for lo, hi in zip(low, up, strict=True)]


def linprog_extremum(
    objective: np.ndarray,
    *,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    lower: np.ndarray | float | None = 0.0,
    upper: np.ndarray | float | None = None,
    sense: Literal['min', 'max'] = 'min',
) -> LPResult:
    """Solve a bounded linear program; return explicit status on failure."""
    c = np.asarray(objective, dtype=np.float64).ravel()
    if c.size == 0 or not np.isfinite(c).all():
        return LPResult('numerical_failure', None, None, 'invalid objective')
    if sense == 'max':
        c = -c
    bounds = _prepare_bounds(c.size, lower, upper)
    try:
        solved = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
        )
    except Exception as exc:  # scipy should not raise; guard anyway
        return LPResult('numerical_failure', None, None, str(exc))
    if solved.status == 0 and solved.success:
        value = float(solved.fun)
        if sense == 'max':
            value = -value
        primal = np.asarray(solved.x, dtype=np.float64)
        return LPResult('optimal', value, primal, solved.message or 'optimal')
    if solved.status == 2:
        return LPResult('infeasible', None, None, solved.message or 'infeasible')
    if solved.status == 3:
        return LPResult('unbounded', None, None, solved.message or 'unbounded')
    return LPResult('numerical_failure', None, None, solved.message or f'status={solved.status}')


def linprog_interval(
    objective: np.ndarray,
    **kwargs,
) -> tuple[LPResult, LPResult]:
    """Min and max of a linear functional over the same feasible set."""
    return linprog_extremum(objective, sense='min', **kwargs), linprog_extremum(
        objective, sense='max', **kwargs
    )
