"""Nonnegative OD flows under count intervals: observability and LP aggregate bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import svdvals

from .lp_core import LPResult, linprog_interval


def _scale_tolerance(matrix: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> float:
    """Frobenius-norm scale for rank decisions."""
    if matrix.size == 0:
        return atol
    scale = float(np.linalg.norm(matrix, ord='fro'))
    return max(rtol * scale, atol)


@dataclass(frozen=True)
class ObservabilityDiagnostics:
    n_paths: int
    n_links: int
    rank_observed: int
    nullity: int
    row_space_identifiable: bool
    singular_values: np.ndarray
    tolerance: float
    identification_source: str  # 'row_space' | 'constraints' | 'both' | 'none'

    def to_dict(self) -> dict[str, Any]:
        return {
            'n_paths': self.n_paths,
            'n_links': self.n_links,
            'rank_observed': self.rank_observed,
            'nullity': self.nullity,
            'row_space_identifiable': self.row_space_identifiable,
            'tolerance': self.tolerance,
            'identification_source': self.identification_source,
            'singular_values': self.singular_values.tolist(),
        }


def observability_diagnostics(
    incidence: np.ndarray,
    observed_mask: np.ndarray,
    *,
    caps: np.ndarray | None = None,
    rtol: float = 1e-8,
) -> ObservabilityDiagnostics:
    """Rank of observed incidence rows; constraint-induced identification flagged separately."""
    A = np.asarray(incidence, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool).ravel()
    if A.ndim != 2:
        raise ValueError('incidence must be two-dimensional')
    if mask.size != A.shape[0]:
        raise ValueError('observed_mask length must match incidence rows')
    A_obs = A[mask] if mask.any() else A[:0]
    tol = _scale_tolerance(A_obs, rtol=rtol)
    svs = svdvals(A_obs) if A_obs.size else np.array([], dtype=np.float64)
    rank = int(np.sum(svs > tol)) if svs.size else 0
    n_paths = A.shape[1]
    nullity = n_paths - rank
    row_id = nullity == 0
    has_finite_caps = (
        caps is not None
        and np.asarray(caps).size == n_paths
        and np.isfinite(caps).all()
        and np.all(np.asarray(caps) > 0)
    )
    if row_id:
        source = 'row_space'
    else:
        source = 'none'
    if not row_id and has_finite_caps:
        lo = np.zeros(n_paths)
        hi = np.asarray(caps, dtype=np.float64)
        feas = flow_interval_feasibility(
            A,
            lo,
            hi,
            np.full(A.shape[0], -np.inf),
            np.full(A.shape[0], np.inf),
            observed_mask=mask,
        )
        if feas['feasible'] and feas.get('unique_aggregate', False):
            source = 'constraints'
    return ObservabilityDiagnostics(
        n_paths=n_paths,
        n_links=A.shape[0],
        rank_observed=rank,
        nullity=nullity,
        row_space_identifiable=row_id,
        singular_values=svs,
        tolerance=tol,
        identification_source=source,
    )


def _interval_rows(
    incidence: np.ndarray,
    count_lower: np.ndarray,
    count_upper: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    A = np.asarray(incidence, dtype=np.float64)
    lo = np.asarray(count_lower, dtype=np.float64).ravel()
    hi = np.asarray(count_upper, dtype=np.float64).ravel()
    mask = np.asarray(observed_mask, dtype=bool).ravel()
    if lo.shape != (A.shape[0],) or hi.shape != (A.shape[0],):
        raise ValueError('count bounds must match incidence rows')
    if not mask.any():
        return None, None, None, None
    A_obs = A[mask]
    lo_obs = lo[mask]
    hi_obs = hi[mask]
    A_ub = np.vstack([A_obs, -A_obs])
    b_ub = np.concatenate([hi_obs, -lo_obs])
    return A_ub, b_ub, lo_obs, hi_obs


def flow_interval_feasibility(
    incidence: np.ndarray,
    flow_lower: np.ndarray,
    flow_upper: np.ndarray,
    count_lower: np.ndarray,
    count_upper: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check whether any nonnegative path flow satisfies count intervals."""
    n_paths = incidence.shape[1]
    if observed_mask is None:
        observed_mask = np.isfinite(count_lower) & np.isfinite(count_upper)
    A_ub, b_ub, _, _ = _interval_rows(incidence, count_lower, count_upper, observed_mask)
    lo = np.zeros(n_paths) if flow_lower is None else np.asarray(flow_lower, dtype=np.float64)
    hi = np.full(n_paths, np.inf) if flow_upper is None else np.asarray(flow_upper, dtype=np.float64)
    if A_ub is None:
        return {'feasible': True, 'status': 'no_count_constraints', 'unique_aggregate': False}
    probe = np.ones(n_paths)
    lo_res, hi_res = linprog_interval(
        probe,
        A_ub=A_ub,
        b_ub=b_ub,
        lower=lo,
        upper=hi,
    )
    feasible = lo_res.status == 'optimal' and hi_res.status == 'optimal'
    unique = False
    if feasible:
        width = hi_res.value - lo_res.value if lo_res.value is not None and hi_res.value is not None else np.inf
        unique = width is not None and width < 1e-6 * max(1.0, abs(lo_res.value))
    status = lo_res.status if lo_res.status != 'optimal' else hi_res.status
    return {
        'feasible': feasible,
        'status': status,
        'total_flow_lower': lo_res.value,
        'total_flow_upper': hi_res.value,
        'unique_aggregate': unique,
        'message': lo_res.message,
    }


def aggregate_bounds(
    incidence: np.ndarray,
    count_lower: np.ndarray,
    count_upper: np.ndarray,
    weights: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
    flow_upper: np.ndarray | None = None,
) -> dict[str, Any]:
    """LP lower/upper bounds on w'f under interval counts and f >= 0."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    n_paths = incidence.shape[1]
    if w.size != n_paths:
        raise ValueError('weights must match path count')
    if observed_mask is None:
        observed_mask = np.isfinite(count_lower) & np.isfinite(count_upper)
    A_ub, b_ub, _, _ = _interval_rows(incidence, count_lower, count_upper, observed_mask)
    if A_ub is None:
        return {'status': 'no_count_constraints', 'lower': None, 'upper': None, 'identifiable': False}
    hi = np.full(n_paths, np.inf) if flow_upper is None else np.asarray(flow_upper, dtype=np.float64)
    lo_res, hi_res = linprog_interval(w, A_ub=A_ub, b_ub=b_ub, lower=0.0, upper=hi)
    if lo_res.status != 'optimal' or hi_res.status != 'optimal':
        status = lo_res.status if lo_res.status != 'optimal' else hi_res.status
        return {
            'status': status,
            'lower': None,
            'upper': None,
            'identifiable': False,
            'message': lo_res.message,
        }
    width = float(hi_res.value - lo_res.value)
    return {
        'status': 'optimal',
        'lower': lo_res.value,
        'upper': hi_res.value,
        'width': width,
        'identifiable': width < 1e-8 * max(1.0, abs(lo_res.value), abs(hi_res.value)),
        'message': 'ok',
    }


def od_synthetic_cases() -> list[dict[str, Any]]:
    """Known null-space and identifiable aggregate examples for oracles."""
    cases = []
    # Two paths share one observed link: null space dim 1, total flow identifiable.
    A = np.array([[1.0, 1.0]])
    cases.append(
        {
            'name': 'shared_link_identifiable_total',
            'incidence': A,
            'count_lower': np.array([10.0]),
            'count_upper': np.array([10.0]),
            'weights': np.array([1.0, 1.0]),
            'expected_identifiable': True,
            'expected_rank': 1,
            'expected_nullity': 1,
        }
    )
    # Full rank 2x2: point identifiable in row space.
    A2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    cases.append(
        {
            'name': 'diagonal_point_identifiable',
            'incidence': A2,
            'count_lower': np.array([3.0, 7.0]),
            'count_upper': np.array([3.0, 7.0]),
            'weights': np.array([1.0, 0.0]),
            'expected_identifiable': True,
            'expected_rank': 2,
            'expected_nullity': 0,
        }
    )
    # Underdetermined without tight counts: wide interval.
    A3 = np.array([[1.0, 1.0, 0.0]])
    cases.append(
        {
            'name': 'wide_interval',
            'incidence': A3,
            'count_lower': np.array([0.0]),
            'count_upper': np.array([100.0]),
            'weights': np.array([1.0, 1.0, 0.0]),
            'expected_identifiable': False,
            'expected_rank': 1,
            'expected_nullity': 2,
            'expected_width_min': 10.0,
        }
    )
    return cases
