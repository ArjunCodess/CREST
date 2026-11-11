import json
from pathlib import Path

import numpy as np
import pytest

from src.research.stress_variants import STRESS_VARIANTS
from src.research.synthetic_sim import (
    DT_HOURS_DEFAULT,
    _chain_operators,
    _simulate_chain_trajectory,
    generate_scenario,
)

DEV_PROBE_SEEDS = [11, 22, 33, 44, 55]
LAYOUT_LINK_COUNTS = [2, 3, 5, 7]


def test_seed_deterministic():
    a = generate_scenario(seed=8810)
    b = generate_scenario(seed=8810)
    np.testing.assert_allclose(a.storage_true, b.storage_true)
    assert a.metadata['seed'] == 8810


def test_nonnegative_admissible_truth():
    s = generate_scenario(seed=8811, demand_vph=600.0)
    assert np.all(s.z_true >= 0)
    assert np.all(s.storage_true >= 0)
    assert s.truth_respects_capacity()


def test_no_truth_leakage_in_obs_mask():
    s = generate_scenario(seed=8812, missing_rate=0.4)
    hidden = ~s.obs_mask
    assert np.isnan(s.obs_values[hidden]).all()


def test_two_boundary_channels_in_scenario():
    s = generate_scenario(seed=8814)
    assert s.R.shape[1] == 2
    assert s.u_true.shape[1] == 2


def test_truth_outside_interval_keeps_coherent_truth():
    s = generate_scenario(seed=8813, truth_outside_interval=True)
    resid = s.storage_true[1:] - s.storage_true[:-1] - s.dt_hours * (
        s.z_true @ s.B.T + s.u_true @ s.R.T
    )
    assert np.max(np.abs(resid)) < 1e-9
    assert not s.boundary_truth_in_interval()


def _balance_residual(scenario) -> float:
    resid = scenario.storage_true[1:] - scenario.storage_true[:-1] - scenario.dt_hours * (
        scenario.z_true @ scenario.B.T + scenario.u_true @ scenario.R.T
    )
    return float(np.max(np.abs(resid)))


@pytest.mark.parametrize('n_links', LAYOUT_LINK_COUNTS)
@pytest.mark.parametrize('seed', DEV_PROBE_SEEDS)
def test_property_feasible_chain_dev_seeds(n_links: int, seed: int) -> None:
    s = generate_scenario(
        seed=seed,
        n_links=n_links,
        demand_vph=1200.0,
        capacity_vehicles=90.0,
        n_steps=12,
    )
    assert _balance_residual(s) < 1e-9
    assert np.all(s.storage_true >= -1e-12)
    assert s.truth_respects_capacity()
    assert np.all(s.z_true >= 0) and np.all(s.u_true >= 0)


@pytest.mark.parametrize('seed', DEV_PROBE_SEEDS)
def test_high_demand_shifted_stress_dev_only(seed: int) -> None:
    overrides = STRESS_VARIANTS['shifted_demand']
    s = generate_scenario(
        seed=seed,
        n_links=7,
        demand_vph=750.0,
        capacity_vehicles=180.0,
        n_steps=8,
        **overrides,
    )
    assert s.truth_respects_capacity()
    assert _balance_residual(s) < 1e-9


@pytest.mark.parametrize('variant', list(STRESS_VARIANTS.keys()))
def test_all_stress_variants_dev_matrix(variant: str) -> None:
    kw = dict(STRESS_VARIANTS[variant])
    s = generate_scenario(seed=44, **kw)
    assert _balance_residual(s) < 1e-9
    assert s.truth_respects_capacity()


def test_zero_demand_decays_link_storage() -> None:
    z0 = generate_scenario(seed=7702, demand_vph=0.0, n_steps=6, capacity_vehicles=200.0)
    assert np.allclose(z0.z_true, 0.0)
    assert _balance_residual(z0) < 1e-9
    assert float(z0.storage_true[-1].sum()) <= float(z0.storage_true[0].sum()) + 1e-9


def test_saturated_initial_high_capacity_pressure() -> None:
    cap = 120.0
    n_links = 3
    n_steps = 10
    demand_vph = 900.0
    dt = DT_HOURS_DEFAULT
    B, R = _chain_operators(n_links)
    capacity = np.full(n_links, cap)
    n0 = capacity.copy()
    tm1 = n_steps - 1
    n_edge = n_links - 1
    z_targets = np.full((tm1, n_edge), demand_vph * 1.25)
    u_in_targets = np.full(tm1, demand_vph)
    u_out_targets = np.full(tm1, demand_vph * 0.88)
    storage, z_true, u_true = _simulate_chain_trajectory(
        B, R, dt, n0, capacity, z_targets, u_in_targets, u_out_targets
    )
    assert np.allclose(storage[0], cap)
    assert np.all(storage <= capacity[:, None].T + 1e-9)
    assert np.all(storage >= -1e-12)
    net = float(np.sum(u_true[:, 0] - u_true[:, 1]) * dt)
    delta = float(storage[-1].sum() - storage[0].sum())
    assert abs(delta - net) < 1e-9


@pytest.mark.parametrize(
    'overrides',
    [
        {'n_links': 2.5},
        {'n_links': True},
        {'n_links': '3'},
        {'n_steps': 1.9},
        {'n_links': 1},
        {'n_steps': 1},
        {'capacity_vehicles': -10.0},
        {'capacity_vehicles': float('inf')},
        {'demand_vph': float('nan')},
        {'missing_rate': 1.0},
        {'missing_rate': '0.1'},
    ],
)
def test_generate_scenario_rejects_invalid_domain(overrides: dict) -> None:
    base = {
        'seed': 7703,
        'n_links': 3,
        'n_steps': 8,
        'demand_vph': 600.0,
        'capacity_vehicles': 120.0,
    }
    base.update(overrides)
    with pytest.raises(ValueError):
        generate_scenario(**base)


def test_analytic_two_cell_net_mass_oracle() -> None:
    dt = DT_HOURS_DEFAULT
    s = generate_scenario(seed=8800, n_links=2, n_steps=3, demand_vph=600.0, capacity_vehicles=500.0)
    net = float(np.sum(s.u_true[:, 0] - s.u_true[:, 1]) * dt)
    delta = float(s.storage_true[-1].sum() - s.storage_true[0].sum())
    assert abs(delta - net) < 1e-9


def test_v3_manifest_seeds_not_executed_in_property_tests() -> None:
    manifest = Path(__file__).resolve().parents[1] / 'configs' / 'research' / 'final_holdout_v3_seed_manifest.json'
    archived = set(json.loads(manifest.read_text(encoding='utf-8'))['seeds'])
    assert archived.isdisjoint(DEV_PROBE_SEEDS)


def test_v4_manifest_seeds_not_probed_in_unit_tests() -> None:
    manifest = Path(__file__).resolve().parents[1] / 'configs' / 'research' / 'final_holdout_v4_seed_manifest.json'
    active = set(json.loads(manifest.read_text(encoding='utf-8'))['seeds'])
    assert active.isdisjoint(DEV_PROBE_SEEDS)


@pytest.mark.parametrize('variant', list(STRESS_VARIANTS.keys()))
def test_v4_scenario_params_feasible_dev_proxy_seed_only(variant: str) -> None:
    """Static final layout/stress declaration checked with dev seeds only (v4 holdout not probed)."""
    kw = dict(STRESS_VARIANTS[variant])
    s = generate_scenario(
        seed=44,
        n_links=7,
        n_steps=8,
        demand_vph=750.0,
        capacity_vehicles=180.0,
        **kw,
    )
    assert s.truth_respects_capacity()
    assert _balance_residual(s) < 1e-9
