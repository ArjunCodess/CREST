"""Declared valid domain for generative chain scenarios (static checks only)."""
from __future__ import annotations

from typing import Any

import numpy as np

# Discrete-time supply cap: one step cannot move more than stock/cap divided by dt.
# Implemented as per-link send cap stock/dt and receive cap (cap-stock)/dt (start-of-step).
CFL_POLICY_DOC = (
    'Explicit Euler with start-of-step flux caps: send_f <= N_i/dt, recv_f <= (cap_i - N_i)/dt '
    'for each cell (vehicles, dt in hours, flux vph). Not full CTM/LWR; boundary channels independent.'
)


def _reject_bool(value: Any, field: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f'{field}: booleans are not valid numeric fields')


def _require_integral_count(value: Any, field: str, *, minimum: int) -> int:
    _reject_bool(value, field)
    if isinstance(value, str):
        raise ValueError(f'{field}: expected finite integral count')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field}: expected finite integral count') from exc
    if not np.isfinite(x):
        raise ValueError(f'{field}: must be finite')
    if x != int(x):
        raise ValueError(f'{field}: must be integral')
    n = int(x)
    if n < minimum:
        raise ValueError(f'{field}: must be >= {minimum}')
    return n


def _require_finite_scalar(
    value: Any,
    field: str,
    *,
    minimum: float,
    allow_equality: bool = True,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> float:
    _reject_bool(value, field)
    if isinstance(value, str):
        raise ValueError(f'{field}: expected finite number')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field}: expected finite number') from exc
    if not np.isfinite(x):
        raise ValueError(f'{field}: must be finite')
    if allow_equality:
        if x < minimum:
            raise ValueError(f'{field}: must be >= {minimum}')
    elif x <= minimum:
        raise ValueError(f'{field}: must be > {minimum}')
    if maximum is not None:
        if maximum_exclusive:
            if x >= maximum:
                raise ValueError(f'{field}: must be < {maximum}')
        elif x > maximum:
            raise ValueError(f'{field}: must be <= {maximum}')
    return x


def validate_generative_scenario_params(params: dict[str, Any], *, layout: str = 'chain') -> dict[str, Any]:
    """Reject static infeasible or ill-typed geometry before protocol registration (RNG-independent)."""
    if layout != 'chain':
        raise ValueError('generative layout must be chain')
    n_links = _require_integral_count(params.get('n_links'), 'scenario_params.n_links', minimum=2)
    n_steps = _require_integral_count(params.get('n_steps'), 'scenario_params.n_steps', minimum=2)
    demand_vph = _require_finite_scalar(
        params.get('demand_vph'),
        'scenario_params.demand_vph',
        minimum=0.0,
        allow_equality=True,
    )
    capacity_vehicles = _require_finite_scalar(
        params.get('capacity_vehicles'),
        'scenario_params.capacity_vehicles',
        minimum=0.0,
        allow_equality=False,
    )
    noise_std = _require_finite_scalar(
        params.get('noise_std'),
        'scenario_params.noise_std',
        minimum=0.0,
        allow_equality=True,
    )
    missing_rate = _require_finite_scalar(
        params.get('missing_rate', 0.0),
        'scenario_params.missing_rate',
        minimum=0.0,
        allow_equality=True,
        maximum=1.0,
        maximum_exclusive=True,
    )
    # Stress variants may raise boundary_shift; the feasible flux integrator throttles at runtime.
    return {
        'layout': layout,
        'n_links': n_links,
        'n_steps': n_steps,
        'demand_vph': demand_vph,
        'capacity_vehicles': capacity_vehicles,
        'noise_std': noise_std,
        'missing_rate': missing_rate,
        'cfl_policy': CFL_POLICY_DOC,
    }
