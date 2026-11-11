"""Shared linear data constraints for conservation LP and point estimation.

Units:
  - Internal/boundary flows z, u are in vehicles per hour (vph).
  - Storage N is in vehicles.
  - Discrete balance: N[t+1] = N[t] + dt_hours * (B @ z[t] + R @ u[t]).
  - A time-aggregated boundary flux in vehicles uses sum_t dt_hours * u[t, k].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConservationPolytope:
    n_vars: int
    n_z: int
    n_u: int
    Tm1: int
    n_links: int
    n_edge: int
    n_bound: int
    dt_hours: float
    n0: np.ndarray
    N_coeff: np.ndarray  # (Tm1+1, n_links, n_vars)
    lower: np.ndarray
    upper: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    obs_mask: np.ndarray
    obs_values: np.ndarray
    obs_tolerance: np.ndarray
    z_upper_vec: np.ndarray
    capacity: np.ndarray | None

    def index_z(self, t: int, e: int) -> int:
        return t * self.n_edge + e

    def index_u(self, t: int, b: int) -> int:
        return self.n_z + t * self.n_bound + b

    def storage_row(self, t: int, ell: int) -> tuple[np.ndarray, float]:
        """Affine map N_t[ell] = const + row @ x."""
        row = self.N_coeff[t, ell]
        return row, float(self.n0[ell])

    def observation_constraints_applied(self) -> int:
        return int(np.sum(self.obs_mask & np.isfinite(self.obs_values)))


def _build_N_coeff(
    B: np.ndarray,
    R: np.ndarray,
    dt_hours: float,
    Tm1: int,
    n_vars: int,
    n_edge: int,
    n_bound: int,
    n_links: int,
) -> np.ndarray:
    N_coeff = np.zeros((Tm1 + 1, n_links, n_vars))
    n_z = Tm1 * n_edge

    def index_z(t: int, e: int) -> int:
        return t * n_edge + e

    def index_u(t: int, b: int) -> int:
        return n_z + t * n_bound + b

    for t in range(1, Tm1 + 1):
        for s in range(t):
            for e in range(n_edge):
                coeff = dt_hours * B[:, e]
                for ell in range(n_links):
                    N_coeff[t, ell, index_z(s, e)] = coeff[ell]
            for b in range(n_bound):
                coeff = dt_hours * R[:, b]
                for ell in range(n_links):
                    N_coeff[t, ell, index_u(s, b)] = coeff[ell]
    return N_coeff


def assemble_conservation_polytope(
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
    z_upper: float | np.ndarray | 1.0 = 1.0,
    z_lower: float | np.ndarray | None = None,
    capacity: np.ndarray | None = None,
) -> ConservationPolytope:
    """Data-conditioned feasible set: box bounds, storage observations, capacity caps."""
    B = np.asarray(B, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n0 = np.asarray(n0, dtype=np.float64).ravel()
    n_links = B.shape[0]
    n_edge = B.shape[1]
    n_bound = R.shape[1]
    u_lower = np.asarray(u_lower, dtype=np.float64)
    u_upper = np.asarray(u_upper, dtype=np.float64)
    Tm1 = u_lower.shape[0]
    n_z = Tm1 * n_edge
    n_u = Tm1 * n_bound
    n_vars = n_z + n_u

    if dt_hours <= 0 or not np.isfinite(dt_hours):
        raise ValueError('dt_hours must be positive and finite')
    if np.any(u_lower > u_upper):
        raise ValueError('inconsistent boundary bounds')

    if obs_mask is None:
        obs_mask = np.zeros((Tm1 + 1, n_links), dtype=bool)
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)
    if obs_mask.shape != (Tm1 + 1, n_links):
        raise ValueError('obs_mask must match (Tm1+1, n_links)')

    if obs_values is None:
        obs_values = np.full(obs_mask.shape, np.nan)
    else:
        obs_values = np.asarray(obs_values, dtype=np.float64)
    if obs_values.shape != obs_mask.shape:
        raise ValueError('obs_values must match obs_mask shape')

    if obs_tolerance is None:
        obs_tol = np.zeros(obs_mask.shape)
    else:
        obs_tol = np.broadcast_to(np.asarray(obs_tolerance, dtype=np.float64), obs_mask.shape).copy()

    z_hi = np.broadcast_to(np.asarray(z_upper, dtype=np.float64), (Tm1, n_edge)).ravel()
    if z_lower is None:
        z_lo = np.zeros(n_z)
    else:
        z_lo = np.broadcast_to(np.asarray(z_lower, dtype=np.float64), (Tm1, n_edge)).ravel()
    if np.any(z_lo > z_hi):
        raise ValueError('inconsistent internal flow bounds')

    cap = None
    if capacity is not None:
        cap = np.asarray(capacity, dtype=np.float64).ravel()
        if cap.shape != (n_links,):
            raise ValueError('capacity must be per link')

    lower = np.concatenate([z_lo, u_lower.ravel()])
    upper = np.concatenate([z_hi, u_upper.ravel()])

    N_coeff = _build_N_coeff(B, R, dt_hours, Tm1, n_vars, n_edge, n_bound, n_links)

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    for t in range(Tm1 + 1):
        for ell in range(n_links):
            if not obs_mask[t, ell] or not np.isfinite(obs_values[t, ell]):
                continue
            row, const = N_coeff[t, ell], n0[ell]
            y = float(obs_values[t, ell])
            tol = float(obs_tol[t, ell])
            lo = y - tol
            hi = y + tol
            A_rows.append(row.copy())
            b_rows.append(hi - const)
            A_rows.append(-row.copy())
            b_rows.append(-(lo - const))

    if cap is not None:
        for t in range(1, Tm1 + 1):
            for ell in range(n_links):
                row = N_coeff[t, ell]
                A_rows.append(row.copy())
                b_rows.append(float(cap[ell] - n0[ell]))

    A_ub = np.vstack(A_rows) if A_rows else None
    b_ub = np.asarray(b_rows, dtype=np.float64) if b_rows else None

    z_upper_vec = np.broadcast_to(np.asarray(z_upper, dtype=np.float64), (Tm1, n_edge))

    return ConservationPolytope(
        n_vars=n_vars,
        n_z=n_z,
        n_u=n_u,
        Tm1=Tm1,
        n_links=n_links,
        n_edge=n_edge,
        n_bound=n_bound,
        dt_hours=float(dt_hours),
        n0=n0,
        N_coeff=N_coeff,
        lower=lower,
        upper=upper,
        A_ub=A_ub,
        b_ub=b_ub,
        obs_mask=obs_mask,
        obs_values=obs_values,
        obs_tolerance=obs_tol,
        z_upper_vec=z_upper_vec,
        capacity=cap,
    )


def boundary_aggregate_weights(
    Tm1: int,
    n_bound: int,
    *,
    mode: str = 'inflow',
) -> np.ndarray:
    """Stacked weights on u decision vars for standard corridor boundary aggregates."""
    w = np.zeros(Tm1 * n_bound, dtype=np.float64)
    if mode == 'inflow' and n_bound >= 1:
        for t in range(Tm1):
            w[t * n_bound + 0] = 1.0
    elif mode == 'outflow' and n_bound >= 2:
        for t in range(Tm1):
            w[t * n_bound + 1] = 1.0
    elif mode == 'net' and n_bound >= 2:
        for t in range(Tm1):
            w[t * n_bound + 0] = 1.0
            w[t * n_bound + 1] = -1.0
    elif mode == 'sum_all_nonneg':
        w[:] = 1.0
    else:
        raise ValueError(f'unsupported boundary aggregate mode={mode!r} for n_bound={n_bound}')
    return w


def boundary_aggregate_objective(
    poly: ConservationPolytope,
    aggregate_weights: np.ndarray,
    *,
    vehicles_not_vph: bool = True,
) -> np.ndarray:
    """Linear functional on decision vector for min/max LP.

    When vehicles_not_vph is True, weights on u slots are multiplied by dt_hours so
    positive weights sum boundary inflow in vehicles (vph * h).
    """
    w = np.asarray(aggregate_weights, dtype=np.float64).ravel()
    if w.size != poly.n_u:
        raise ValueError('aggregate_weights must match stacked boundary flows (Tm1 * n_bound)')
    c = np.zeros(poly.n_vars)
    scale = poly.dt_hours if vehicles_not_vph else 1.0
    c[poly.n_z :] = w * scale
    return c


def evaluate_affine_constraints(x: np.ndarray, A_ub: np.ndarray | None, b_ub: np.ndarray | None) -> dict[str, Any]:
    if A_ub is None or b_ub is None:
        return {'max_violation': 0.0, 'feasible': True}
    slack = b_ub - A_ub @ x
    max_viol = float(np.max(-slack)) if slack.size else 0.0
    return {'max_violation': max_viol, 'feasible': max_viol <= 1e-7}
