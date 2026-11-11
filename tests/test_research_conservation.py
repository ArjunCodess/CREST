import numpy as np

from src.research.boundary_conservation import (
    conservation_lp_box_only,
    conservation_lp_extrema,
    fit_boundary_conservation,
    pack_state,
)
from src.research.conservation_constraints import boundary_aggregate_weights
from src.research.synthetic_sim import (
    _chain_operators,
    _chain_operators_legacy_shared_inlet_outlet,
    generate_scenario,
)


def test_pack_state_units_conservation():
    scenario = generate_scenario(seed=9001, n_steps=5)
    storage = pack_state(scenario.z_true, scenario.u_true, scenario.n0, scenario.B, scenario.R, scenario.dt_hours)
    resid = storage[1:] - storage[:-1] - scenario.dt_hours * (
        scenario.z_true @ scenario.B.T + scenario.u_true @ scenario.R.T
    )
    assert np.max(np.abs(resid)) < 1e-9


def test_fit_distinguishes_observation_and_prior_terms():
    scenario = generate_scenario(seed=9002, missing_rate=0.0, noise_std=5.0)
    obs = np.nan_to_num(scenario.obs_values, nan=0.0)
    est = fit_boundary_conservation(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.obs_mask,
        obs,
        scenario.u_prior,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        w_obs=0.0,
        w_prior=0.05,
        z_upper=scenario.metadata['max_edge_flow_vph'],
        capacity=scenario.capacity,
        obs_tolerance=scenario.metadata['obs_tolerance'],
    )
    assert est.status == 'optimal'
    assert est.prior_term >= 0
    assert est.dynamics_algebraic_residual_rmse < 1e-6


def test_independent_boundary_channels_shape():
    B, R = _chain_operators(3)
    assert R.shape == (3, 2)
    assert R[0, 0] == 1.0 and R[-1, 1] == -1.0
    assert R[0, 1] == 0.0 and R[-1, 0] == 0.0


def test_unequal_boundaries_change_corridor_storage():
    scenario = generate_scenario(seed=9010, n_links=3, n_steps=5, demand_vph=900.0)
    total_n0 = float(scenario.storage_true[0].sum())
    total_nT = float(scenario.storage_true[-1].sum())
    assert abs(total_nT - total_n0) > 0.5
    net = float(np.sum(scenario.u_true[:, 0] - scenario.u_true[:, 1]) * scenario.dt_hours)
    assert abs((total_nT - total_n0) - net) < 1e-6


def test_legacy_shared_u_fixture_constant_corridor_sum():
    B, R = _chain_operators_legacy_shared_inlet_outlet(4)
    assert R.shape[1] == 1
    dt = 5.0 / 60.0
    Tm1 = 3
    z = np.full((Tm1, B.shape[1]), 400.0)
    u = np.full((Tm1, 1), 800.0)
    n0 = np.zeros(4)
    storage = pack_state(z, u, n0, B, R, dt)
    assert abs(float(storage[-1].sum() - storage[0].sum())) < 1e-9


def test_lp_observations_sharpen_vs_box():
    scenario = generate_scenario(seed=9003, missing_rate=0.05, noise_std=8.0)
    agg = boundary_aggregate_weights(scenario.u_interval_lower.shape[0], scenario.R.shape[1], mode='net')
    z_upper = scenario.metadata['max_edge_flow_vph']
    tol = scenario.metadata['obs_tolerance']
    kw = dict(
        z_upper=z_upper,
        capacity=scenario.capacity,
        obs_mask=scenario.obs_mask,
        obs_values=np.nan_to_num(scenario.obs_values, nan=0.0),
        obs_tolerance=tol,
    )
    data_bounds = conservation_lp_extrema(
        scenario.B, scenario.R, scenario.dt_hours, scenario.n0,
        scenario.u_interval_lower, scenario.u_interval_upper, agg, **kw,
    )
    box_bounds = conservation_lp_box_only(
        scenario.B, scenario.R, scenario.dt_hours, scenario.n0,
        scenario.u_interval_lower, scenario.u_interval_upper, agg,
        z_upper=z_upper, capacity=scenario.capacity,
    )
    assert data_bounds['status'] == 'optimal' and box_bounds['status'] == 'optimal'
    assert data_bounds['width'] <= box_bounds['width'] + 1e-3


def test_wrong_boundary_interval_uncovered_or_infeasible():
    scenario = generate_scenario(seed=9004, truth_outside_interval=True)
    net_w = boundary_aggregate_weights(scenario.u_interval_lower.shape[0], scenario.R.shape[1], mode='net')
    bounds = conservation_lp_extrema(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        net_w,
        obs_mask=scenario.obs_mask,
        obs_values=np.nan_to_num(scenario.obs_values, nan=0.0),
        obs_tolerance=scenario.metadata['obs_tolerance'],
        z_upper=scenario.metadata['max_edge_flow_vph'],
        capacity=scenario.capacity,
    )
    true_net = float(np.sum(scenario.u_true[:, 0] - scenario.u_true[:, 1]) * scenario.dt_hours)
    if bounds['status'] == 'optimal':
        covered = bounds['lower'] <= true_net <= bounds['upper']
        assert not covered


def test_analytic_three_cell_net_lp_contains_truth():
    B, R = _chain_operators(3)
    dt = 5.0 / 60.0
    Tm1 = 2
    z_hi = 5000.0
    n0 = np.array([1.0, 1.0, 1.0])
    u_lo = np.zeros((Tm1, 2))
    u_hi = np.full((Tm1, 2), 2000.0)
    obs_mask = np.zeros((Tm1 + 1, 3), dtype=bool)
    obs_values = np.full((Tm1 + 1, 3), np.nan)
    obs_mask[-1, :] = True
    obs_values[-1] = n0 + dt * np.array([200.0, 220.0, 180.0])
    u_true = np.array([[900.0, 700.0], [850.0, 720.0]])
    z_true = np.full((Tm1, 2), 400.0)
    storage = pack_state(z_true, u_true, n0, B, R, dt)
    obs_values = storage.copy()
    net_w = boundary_aggregate_weights(Tm1, 2, mode='net')
    bounds = conservation_lp_extrema(
        B, R, dt, n0, u_lo, u_hi, net_w,
        obs_mask=obs_mask, obs_values=obs_values, obs_tolerance=0.5, z_upper=z_hi,
    )
    true_net = float(np.sum(u_true[:, 0] - u_true[:, 1]) * dt)
    assert bounds['status'] == 'optimal'
    assert bounds['lower'] <= true_net <= bounds['upper']
    box = conservation_lp_box_only(B, R, dt, n0, u_lo, u_hi, net_w, z_upper=z_hi)
    assert box['width'] >= bounds['width'] - 1e-6


def test_independent_recovery_error_nonzero_for_wrong_midpoint():
    scenario = generate_scenario(seed=9005, truth_outside_interval=True)
    wrong_u = scenario.u_interval_upper
    z = np.full_like(scenario.z_true, 50.0)
    storage = pack_state(z, wrong_u, scenario.n0, scenario.B, scenario.R, scenario.dt_hours)
    err = float(np.sqrt(np.mean((storage - scenario.storage_true) ** 2)))
    assert err > 1.0
    assert scenario.u_true[0, 0] < scenario.u_interval_lower[0, 0]
