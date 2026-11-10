import numpy as np
import pandas as pd
from src.state_v6 import panel_weights, ramp_features, uncertainty_projection
from src.research_diagnostics import noise_nugget, missing_run_lengths


def test_noise_nugget_on_known_white_noise_and_missing_runs():
    x = np.random.default_rng(87).normal(0, 3, (20000, 2))
    report = noise_nugget(x, np.ones_like(x, bool))
    assert abs(report['sigma_linear']-3) < .1
    runs = missing_run_lengths(np.array([[0, 1], [1, 1], [1, 0], [0, 1]], bool))
    np.testing.assert_array_equal(runs, [[0, 2], [2, 2], [2, 0], [0, 1]])


def test_macro_weights_equalize_regime_sample_counts_and_use_per_lane_error():
    regimes = np.array([.2, .3, .3, .3])
    residual = np.tile([2., 600.], (4, 1))
    lanes = np.full(4, 3.)
    weights, scales = panel_weights(regimes, residual, lanes)
    np.testing.assert_allclose(weights[0], weights[1:].sum(0))
    np.testing.assert_allclose(scales['0.2'], [2., 200.])


def test_unknown_flux_is_missing_and_not_a_zero_ramp():
    params = pd.DataFrame({'capacity_vph': [6000.]*3, 'free_speed_kmh': [100.]*3, 'k_jam': [400.]*3})
    values = np.tile(np.array([[90., 1500.]]), (5, 3, 1))
    inc = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    flux = (inc, inc.T, np.zeros((5, 3)), np.zeros((5, 3), bool))
    x = ramp_features(values, params, flux)
    assert np.isnan(x[..., :6]).all()
    assert not x[..., 6:8].any()
    known = np.ones((5, 3), bool)
    known[:, [0, 2]] = False
    flux = (inc, inc.T, np.zeros((5, 3)), known)
    assert np.isfinite(ramp_features(values, params, flux)[:, 1, 0]).all()
    values[:, 0, 1] = np.nan
    assert np.isnan(ramp_features(values, params, flux)[:, 1, 0]).all()


def test_uncertainty_projection_preserves_observations_and_reduces_mass_error():
    params = pd.DataFrame({'length_km': [1.]*3, 'free_speed_kmh': [100.]*3, 'capacity_vph': [6000.]*3})
    states = np.tile(np.array([[80., 1200.]]), (8, 3, 1))
    states[3:5, 1, 1] = 2200
    observed = states.copy(); observed[:, 1] = np.nan
    inc = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    known = np.zeros((8, 3), bool); known[:, 1] = True
    flux = (inc, inc.T, np.zeros((8, 3)), known)
    corrected, report = uncertainty_projection(states, observed, params, flux,
        np.tile([10., 800.], (3, 1)), weight=2., max_nfev=30)
    np.testing.assert_array_equal(corrected[np.isfinite(observed)], observed[np.isfinite(observed)])
    assert report['after']['mass_rmse'] < report['before']['mass_rmse']
    assert np.isfinite(corrected).all() and (corrected >= 0).all()
    # Unknown public flux makes an honest no-op, regardless of regularization.
    empty_flux = (inc, inc.T, np.zeros((8, 3)), np.zeros((8, 3), bool))
    unchanged, report = uncertainty_projection(states, observed, params, empty_flux, np.ones((3, 2)))
    np.testing.assert_array_equal(unchanged, states)
    assert not report['optimized']
