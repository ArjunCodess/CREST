"""Controlled synthetic networks with coherent boundary truth (vehicles, vph, 5-minute dt)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .synthetic_domain import CFL_POLICY_DOC, validate_generative_scenario_params

DT_HOURS_DEFAULT = 5.0 / 60.0  # five-minute steps


@dataclass
class SyntheticScenario:
    name: str
    B: np.ndarray
    R: np.ndarray
    dt_hours: float
    n0: np.ndarray
    z_true: np.ndarray
    u_true: np.ndarray
    storage_true: np.ndarray
    capacity: np.ndarray
    obs_mask: np.ndarray
    obs_values: np.ndarray
    u_interval_lower: np.ndarray
    u_interval_upper: np.ndarray
    u_prior: np.ndarray
    incidence: np.ndarray
    path_flow_true: np.ndarray
    count_lower: np.ndarray
    count_upper: np.ndarray
    count_observed_mask: np.ndarray
    path_flow_cap: np.ndarray
    metadata: dict[str, Any]

    def boundary_truth_in_interval(self) -> bool:
        return bool(
            np.all(self.u_true >= self.u_interval_lower) and np.all(self.u_true <= self.u_interval_upper)
        )

    def truth_respects_capacity(self) -> bool:
        return bool(np.all(self.storage_true <= self.capacity[:, None].T + 1e-9))


def _chain_operators(n_links: int) -> tuple[np.ndarray, np.ndarray]:
    """Mainline chain with independent inlet (ch0) and outlet (ch1) boundary fluxes (both nonnegative vph)."""
    n_edge = n_links - 1
    B = np.zeros((n_links, n_edge))
    R = np.zeros((n_links, 2))
    for i in range(n_edge):
        B[i, i] = -1.0
        B[i + 1, i] = 1.0
    R[0, 0] = 1.0
    R[-1, 1] = -1.0
    return B, R


def _chain_operators_legacy_shared_inlet_outlet(n_links: int) -> tuple[np.ndarray, np.ndarray]:
    """Explicit legacy fixture: one boundary channel couples inlet/outlet (corridor sum N constant)."""
    n_edge = n_links - 1
    B = np.zeros((n_links, n_edge))
    R = np.zeros((n_links, 1))
    for i in range(n_edge):
        B[i, i] = -1.0
        B[i + 1, i] = 1.0
    R[0, 0] = 1.0
    R[-1, 0] = -1.0
    return B, R


def _overlapping_path_incidence(n_paths: int, n_links: int) -> np.ndarray:
    """Paths with shared links so row rank < n_paths when counts are partial."""
    A = np.zeros((n_links, n_paths))
    for p in range(n_paths):
        if p < n_links:
            A[p, p] = 1.0
        elif p < n_links + max(1, n_links - 2):
            span = min(2, n_links - 1)
            for ell in range(span):
                A[ell, p] = 1.0
        else:
            A[min(p % n_links, n_links - 1), p] = 1.0
    return A


def _send_cap_vph(stock_veh: float, dt: float) -> float:
    """Max discharge rate (vph) so dt * rate <= start-of-step stock."""
    return max(0.0, stock_veh / dt) if dt > 0 else 0.0


def _recv_cap_vph(stock_veh: float, cap_veh: float, dt: float) -> float:
    """Max inflow rate (vph) so start-of-step stock plus dt * rate <= cap."""
    return max(0.0, (cap_veh - stock_veh) / dt) if dt > 0 else 0.0


def _feasible_chain_fluxes(
    n: np.ndarray,
    capacity: np.ndarray,
    dt: float,
    z_target: np.ndarray,
    u_in_target: float,
    u_out_target: float,
) -> tuple[np.ndarray, float, float]:
    """Cell-transmission-style caps at interval start; independent inlet/outlet."""
    n_links = n.shape[0]
    u_in = min(max(0.0, u_in_target), _recv_cap_vph(float(n[0]), float(capacity[0]), dt))
    u_out = min(max(0.0, u_out_target), _send_cap_vph(float(n[-1]), dt))
    z = np.zeros(n_links - 1, dtype=float)
    for e in range(n_links - 1):
        send = _send_cap_vph(float(n[e]), dt)
        recv = _recv_cap_vph(float(n[e + 1]), float(capacity[e + 1]), dt)
        z[e] = min(max(0.0, z_target[e]), send, recv)
    return z, u_in, u_out


def _simulate_chain_trajectory(
    B: np.ndarray,
    R: np.ndarray,
    dt: float,
    n0: np.ndarray,
    capacity: np.ndarray,
    z_targets: np.ndarray,
    u_in_targets: np.ndarray,
    u_out_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate with supply/demand-limited flux each step (no post-hoc storage clipping)."""
    Tm1 = z_targets.shape[0]
    n_steps = Tm1 + 1
    n_links = B.shape[0]
    storage = np.zeros((n_steps, n_links))
    storage[0] = np.minimum(np.maximum(n0, 0.0), capacity)
    z_true = np.zeros((Tm1, n_links - 1))
    u_true = np.zeros((Tm1, 2))
    for t in range(Tm1):
        z_t, u_in, u_out = _feasible_chain_fluxes(
            storage[t],
            capacity,
            dt,
            z_targets[t],
            float(u_in_targets[t]),
            float(u_out_targets[t]),
        )
        z_true[t] = z_t
        u_true[t, 0] = u_in
        u_true[t, 1] = u_out
        storage[t + 1] = storage[t] + dt * (B @ z_t + R @ u_true[t])
        if np.any(storage[t + 1] < -1e-9) or np.any(storage[t + 1] > capacity + 1e-6):
            raise ValueError('internal generative invariant broken after feasible flux step')
    return storage, z_true, u_true


def _integrate_storage(
    B: np.ndarray,
    R: np.ndarray,
    dt: float,
    n0: np.ndarray,
    z_true: np.ndarray,
    u_true: np.ndarray,
    capacity: np.ndarray,
) -> np.ndarray:
    """Forward integrate a fixed flux trajectory (tests / oracle paths)."""
    Tm1 = z_true.shape[0]
    n_steps = Tm1 + 1
    storage = np.zeros((n_steps, B.shape[0]))
    storage[0] = np.minimum(n0, capacity)
    for t in range(Tm1):
        storage[t + 1] = storage[t] + dt * (B @ z_true[t] + R @ u_true[t])
    if np.any(storage > capacity + 1e-6) or np.any(storage < -1e-9):
        raise ValueError('generative flows violate capacity or negativity')
    return storage


def generate_scenario(
    *,
    seed: int,
    layout: str = 'chain',
    n_links: int = 5,
    n_steps: int = 8,
    demand_vph: float = 800.0,
    capacity_vehicles: float = 200.0,
    noise_std: float = 15.0,
    missing_rate: float = 0.2,
    count_missing_rate: float = 0.25,
    contiguous_dropout: bool = False,
    boundary_shift: float = 0.0,
    wrong_interval: bool = False,
    truth_outside_interval: bool = False,
    od_prior_shift: float = 0.0,
    obs_tolerance: float | None = None,
    obs_noise_model: str = 'uniform_bounded',
    obs_noise_scale: float | None = None,
    inconsistent_od_counts: bool = False,
) -> SyntheticScenario:
    validated = validate_generative_scenario_params(
        {
            'layout': layout,
            'n_links': n_links,
            'n_steps': n_steps,
            'demand_vph': demand_vph,
            'capacity_vehicles': capacity_vehicles,
            'noise_std': noise_std,
            'missing_rate': missing_rate,
        },
        layout=layout,
    )
    n_links = validated['n_links']
    n_steps = validated['n_steps']
    demand_vph = validated['demand_vph']
    capacity_vehicles = validated['capacity_vehicles']
    noise_std = validated['noise_std']
    missing_rate = validated['missing_rate']

    rng = np.random.default_rng(seed)
    dt = DT_HOURS_DEFAULT
    if layout != 'chain':
        raise ValueError('only chain layout in v1 pilot')
    if obs_noise_model not in ('uniform_bounded', 'gaussian_misspecification'):
        raise ValueError(f'unknown obs_noise_model {obs_noise_model}')
    if obs_noise_scale is None:
        obs_noise_scale = float(noise_std)
    if obs_tolerance is None:
        if obs_noise_model == 'uniform_bounded':
            obs_tolerance = float(obs_noise_scale)
        else:
            obs_tolerance = max(noise_std * 2.5, 1.0)

    B, R = _chain_operators(n_links)
    n_edge = B.shape[1]
    Tm1 = n_steps - 1
    capacity = np.full(n_links, capacity_vehicles)

    u_in_pulse = max(0.0, demand_vph * (1.0 + boundary_shift))
    u_out_pulse = max(0.0, demand_vph * (0.88 + 0.25 * boundary_shift))
    z_cap = demand_vph * 1.25
    z_level = min(demand_vph, z_cap)
    z_targets = np.broadcast_to(z_level, (Tm1, n_edge)).copy()
    z_targets += rng.normal(0, 0.05 * demand_vph, z_targets.shape)
    z_targets = np.clip(z_targets, 0.0, z_cap)

    u_in_targets = np.full(Tm1, u_in_pulse)
    u_out_targets = np.full(Tm1, u_out_pulse)
    n0 = np.full(n_links, 0.35 * capacity_vehicles)
    storage, z_true, u_true = _simulate_chain_trajectory(
        B, R, dt, n0, capacity, z_targets, u_in_targets, u_out_targets
    )

    obs_mask = rng.random(storage.shape) > missing_rate
    if contiguous_dropout and n_steps >= 4:
        start = int(rng.integers(1, n_steps - 2))
        obs_mask[start : start + 2, :] = False
    obs_values = storage.copy()
    if obs_noise_model == 'uniform_bounded':
        noise = rng.uniform(-obs_noise_scale, obs_noise_scale, storage.shape)
    else:
        noise = rng.normal(0, noise_std, storage.shape)
    obs_values = obs_values + noise
    obs_values = np.where(obs_mask, obs_values, np.nan)

    u_in = float(np.mean(u_true[:, 0]))
    u_out = float(np.mean(u_true[:, 1]))
    half_width_in = 0.15 * abs(u_in) + 50.0
    half_width_out = 0.15 * abs(u_out) + 50.0
    if wrong_interval:
        half_width_in *= 0.5
        half_width_out *= 0.5
    u_lo_in = u_in - half_width_in
    u_hi_in = u_in + half_width_in
    u_lo_out = u_out - half_width_out
    u_hi_out = u_out + half_width_out
    if truth_outside_interval:
        offset = 2.5 * half_width_in
        u_lo_in = u_in + offset
        u_hi_in = u_in + offset + half_width_in * 0.5
        u_lo_out = max(0.0, u_out - half_width_out * 0.25)
        u_hi_out = u_out + half_width_out * 0.25
    u_interval_lower = np.zeros((Tm1, 2))
    u_interval_upper = np.zeros((Tm1, 2))
    u_interval_lower[:, 0] = u_lo_in
    u_interval_upper[:, 0] = u_hi_in
    u_interval_lower[:, 1] = np.maximum(0.0, u_lo_out)
    u_interval_upper[:, 1] = u_hi_out
    u_prior = np.zeros((Tm1, 2))
    u_prior[:, 0] = u_in * (1.0 + 0.05 * od_prior_shift)
    u_prior[:, 1] = u_out * (1.0 + 0.05 * od_prior_shift * 0.5)

    n_paths = max(n_links + 2, 6)
    incidence = _overlapping_path_incidence(n_paths, n_links)
    shares = rng.dirichlet(np.ones(n_paths))
    total = float(np.sum(u_true[:, 0]) * dt)
    path_flow_true = np.maximum(shares * total * (1.0 + od_prior_shift), 0.0)
    path_flow_cap = path_flow_true * 1.5 + 0.5
    loaded = incidence @ path_flow_true
    width = 0.12 * np.maximum(loaded, 1.0) + 8.0
    count_lower = np.maximum(loaded - width, 0.0)
    count_upper = loaded + width

    count_observed_mask = rng.random(n_links) > count_missing_rate
    if not count_observed_mask.any():
        count_observed_mask[rng.integers(0, n_links)] = True
    if inconsistent_od_counts:
        for i in range(len(loaded)):
            if not count_observed_mask[i]:
                continue
            count_lower[i] = float(loaded[i]) + 500.0
            count_upper[i] = float(loaded[i]) + 50.0
            break
    count_lower = np.where(count_observed_mask, count_lower, np.nan)
    count_upper = np.where(count_observed_mask, count_upper, np.nan)

    meta = {
        'seed': seed,
        'layout': layout,
        'n_links': n_links,
        'n_steps': n_steps,
        'demand_vph': demand_vph,
        'capacity_vehicles': capacity_vehicles,
        'max_edge_flow_vph': z_cap,
        'obs_tolerance': obs_tolerance,
        'missing_rate': missing_rate,
        'count_missing_rate': count_missing_rate,
        'contiguous_dropout': contiguous_dropout,
        'boundary_shift': boundary_shift,
        'wrong_interval': wrong_interval,
        'truth_outside_interval': truth_outside_interval,
        'od_prior_shift': od_prior_shift,
        'obs_noise_model': obs_noise_model,
        'obs_noise_scale': float(obs_noise_scale),
        'inconsistent_od_counts': inconsistent_od_counts,
        'cfl_policy': CFL_POLICY_DOC,
        'generative_flux_mode': 'feasible_chain_supply_demand_caps',
        'coverage_note': (
            'uniform_bounded: marginal tube coverage matches declared obs_tolerance; '
            'no nominal joint-Gaussian confidence guarantee.'
            if obs_noise_model == 'uniform_bounded'
            else 'gaussian_misspecification: Gaussian draws with hard ±2.5σ-style tube; '
            'generative truth may lie outside observation polytope (joint misspecification stress).'
        ),
        'units': {
            'storage': 'vehicles',
            'flow': 'vph',
            'dt_hours': dt,
            'boundary_channels': ['inlet_vph_nonneg', 'outlet_magnitude_vph_nonneg'],
            'boundary_inflow_vehicles': 'sum_t dt*u[:,0]',
            'boundary_outflow_vehicles': 'sum_t dt*u[:,1]',
            'boundary_net_inflow_vehicles': 'sum_t dt*(u[:,0]-u[:,1])',
        },
    }
    return SyntheticScenario(
        name=f'{layout}_s{seed}',
        B=B,
        R=R,
        dt_hours=dt,
        n0=n0,
        z_true=z_true,
        u_true=u_true,
        storage_true=storage,
        capacity=capacity,
        obs_mask=obs_mask,
        obs_values=np.where(obs_mask, obs_values, np.nan),
        u_interval_lower=u_interval_lower,
        u_interval_upper=u_interval_upper,
        u_prior=u_prior,
        incidence=incidence,
        path_flow_true=path_flow_true,
        count_lower=count_lower,
        count_upper=count_upper,
        count_observed_mask=count_observed_mask,
        path_flow_cap=path_flow_cap,
        metadata=meta,
    )
