import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.od import solve_od
from src.od_v6 import (
    BASELINE_REGULARIZATION,
    DELTA_TOTAL_PER_0_01_S_OD,
    S_OD_COEFFICIENT_IN_TOTAL,
    SolverError,
    align_counts,
    attraction_observability,
    counts_from_paths,
    experimental_correction_smooth,
    held_link_l1,
    holdout_size,
    independent_lognormal_truth,
    inventory_tables,
    load_train_panel,
    nested_link_assessment,
    pool_prior_shares,
    promotion_gate,
    relative_deviation_weights,
    repeated_holdouts,
    scipy_nnls_reference,
    select_config,
    solve_weighted_od,
    temporal_smoothed_biased_shares,
    timestamp_adjacency,
)


def test_composite_coefficient_of_path_term():
    assert S_OD_COEFFICIENT_IN_TOTAL == pytest.approx(0.09)
    assert DELTA_TOTAL_PER_0_01_S_OD == pytest.approx(0.0009)


def test_absolute_weights_match_existing_solver_and_constant_rescaling():
    rng = np.random.default_rng(19)
    matrix = (rng.random((12, 40)) < 0.25).astype(float)
    operator = csr_matrix(matrix)
    prior = rng.uniform(0, 100, size=40)
    counts = matrix @ rng.uniform(0, 150, size=40)
    expected, _ = solve_od(operator, counts, prior, regularization=0.05)
    actual, diagnostics = solve_weighted_od(operator, counts, prior, regularization=0.05, weight_mode='absolute')
    ones, _ = solve_weighted_od(operator, counts, prior, regularization=0.05, weights=np.ones(40))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(ones, expected, rtol=1e-5, atol=1e-3)
    scaled_prior_penalty, _ = solve_od(operator, counts, prior, regularization=0.2)
    weighted, _ = solve_weighted_od(operator, counts, prior, regularization=0.05, weights=np.full(40, 4.0))
    np.testing.assert_allclose(weighted, scaled_prior_penalty, rtol=1e-5, atol=1e-3)
    assert diagnostics['converged'] and diagnostics['kkt_ok']
    assert 'S_dev' not in diagnostics
    assert 'prior_deviation_l1' in diagnostics and 'prior_deviation_l2' in diagnostics


def test_weighted_solution_matches_scipy_nnls():
    rng = np.random.default_rng(23)
    matrix = (rng.random((8, 15)) < 0.4).astype(float)
    operator = csr_matrix(matrix)
    prior = rng.uniform(0, 30, size=15)
    prior[0] = 0
    counts = matrix @ rng.uniform(0, 40, size=15)
    weights, floor = relative_deviation_weights(prior)
    expected = scipy_nnls_reference(matrix, counts, prior, 0.05, weights)
    actual, _ = solve_weighted_od(operator, counts, prior, regularization=0.05, weights=weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
    assert floor > 0


def test_relative_floor_is_shared_by_zero_and_tiny_priors():
    weights, floor = relative_deviation_weights(np.array([0.0, 1e-12, 5.0, 20.0]))
    assert floor == pytest.approx(0.125)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] == pytest.approx((12.5 / floor)**2)
    assert weights[3] == pytest.approx((12.5 / 20)**2)
    assert np.isfinite(weights).all()
    rescaled, rescaled_floor = relative_deviation_weights(np.array([0.0, 1e-12, 5.0, 20.0])*1000)
    np.testing.assert_allclose(rescaled, weights)
    assert rescaled_floor == pytest.approx(floor*1000)


def test_failed_iteration_does_not_return_a_flow():
    operator = csr_matrix(np.eye(3))
    with pytest.raises(SolverError):
        solve_weighted_od(operator, np.array([1.0, 2.0, 3.0]), np.ones(3), max_iter=0)
    with pytest.raises(ValueError):
        solve_weighted_od(operator, np.array([1.0, 2.0, 3.0]), np.ones(3), regularization=0)
    with pytest.raises(ValueError):
        solve_weighted_od(operator, np.array([1.0, np.nan, 3.0]), np.array([1.0, -1.0, 1.0]))


def test_missing_count_is_not_treated_as_zero_and_null_component_stays_with_prior():
    operator = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    flows, diagnostics = solve_weighted_od(operator, np.array([50.0, np.nan]), np.array([10.0, 25.0]))
    assert diagnostics['n_measured_links'] == 1
    with pytest.raises(ValueError, match='finite or NaN'):
        solve_weighted_od(operator, np.array([50.0, np.inf]), np.array([10.0, 25.0]))
    assert flows[1] == pytest.approx(25.0, rel=1e-5, abs=1e-4)
    rank_deficient = csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    null = np.array([1.0, -1.0, 0.0])
    base = np.array([4.0, 4.0, 2.0])
    truth = base + 3 * null
    assert np.allclose(rank_deficient @ null, 0)
    recovered, recovered_diagnostics = solve_weighted_od(rank_deficient, rank_deficient @ truth, base, regularization=0.05)
    assert recovered_diagnostics['count_normalized_l1'] < 1e-6
    assert np.linalg.norm(recovered - base) < np.linalg.norm(recovered - truth)


def test_holdouts_are_disjoint_and_cover_ten_to_twenty_percent():
    groups, width = repeated_holdouts(81, 4, 0.15, np.random.default_rng(4))
    assert width == holdout_size(81)
    assert 0.10 * 81 - 1e-8 <= width <= 0.20 * 81
    stacked = np.concatenate(groups)
    assert len(stacked) == len(np.unique(stacked))
    assert all(len(group) == width for group in groups)


def test_selection_cannot_see_assessment_counts():
    rng = np.random.default_rng(5)
    matrix = (rng.random((10, 12)) < 0.5).astype(float)
    np.fill_diagonal(matrix, 1)
    operator = csr_matrix(matrix)
    truth = rng.uniform(5, 20, size=12)
    counts = matrix @ truth
    prior = np.full(12, 8.0)
    eligible = np.arange(2, 10)
    poisoned = counts.copy()
    poisoned[:2] = 1e9
    choice, info = select_config(operator, poisoned, prior, eligible, np.random.default_rng(6), inner_folds=2)
    clean_choice, clean_info = select_config(operator, counts, prior, eligible, np.random.default_rng(6), inner_folds=2)
    assert choice == clean_choice and info == clean_info
    assert info['n_finite_observed'] == len(eligible)
    assert info['max_abs_count'] == pytest.approx(np.max(np.abs(counts[eligible])))
    assert choice[0] in (0.005, 0.05, 0.5)
    assert choice[1] in ('absolute', 'relative')
    assessment = nested_link_assessment(operator, counts, prior, folds=2, seed=8)
    for row in assessment['outer']:
        assert row['role'] == 'outer_assessment' and row['used_for_gate']
        assert set(row) >= {'held_link_l1', 'prior_deviation_l1', 'prior_deviation_l2'}
        assert 'S_dev' not in row
    inner_roles = {row['role'] for row in assessment['inner']}
    assert inner_roles == {'inner_selection'}
    assert all(row['used_for_gate'] is False for row in assessment['inner'])


def test_promotion_gate_requires_every_mechanism():
    synthetic = {
        'additive_absolute': {'baseline_path_l1': 0.40, 'candidate_path_l1': 0.20},
        'lognormal_multiplicative': {'baseline_path_l1': 0.30, 'candidate_path_l1': 0.50},
        'temporal_smoothed_biased_shares': {'baseline_path_l1': 0.30, 'candidate_path_l1': 0.30},
    }
    rejected = promotion_gate(0.20, 0.10, synthetic)
    assert rejected['decision'] == 'keep_baseline' and not rejected['gate_passed']
    assert rejected['production_change'] is False
    synthetic['lognormal_multiplicative']['candidate_path_l1'] = 0.25
    accepted = promotion_gate(0.20, 0.10, synthetic)
    assert accepted['decision'] == 'gate_passed_not_deployed' and accepted['gate_passed']
    assert accepted['production_change'] is False
    incomplete = promotion_gate(0.20, 0.10, {'additive_absolute': synthetic['additive_absolute']})
    assert incomplete['decision'] == 'keep_baseline'
    assert 'lognormal_multiplicative' in incomplete['missing_mechanisms']
    no_link_gain = promotion_gate(0.20, 0.199, synthetic)
    assert no_link_gain['decision'] == 'keep_baseline' and not no_link_gain['held_link_benefit']


def test_single_static_interval_does_not_fabricate_dynamics():
    operator = csr_matrix(np.eye(2))
    result = pool_prior_shares(operator, [np.array([1.0, 2.0])], [np.array([1.0, 2.0])], ['SYN_PM_TRAIN'])
    assert result['status'] == 'not_applicable_single_static_interval'
    assert result['flows'] is None and result['fabricated_periods'] == 0
    with pytest.raises(ValueError):
        timestamp_adjacency(['interval-10', 'interval-2'], '1h')
    parsed, edges = timestamp_adjacency(
        ['2030-06-01T18:00:00Z', '2030-06-01T15:00:00Z', '2030-06-03T15:00:00Z'], '4h'
    )
    assert edges == [(1, 0)]
    assert parsed[1].startswith('2030-06-01T15:00:00')


def test_prior_share_pooling_recovers_rank_one_truth_and_not_a_moving_null_component():
    operator = csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
    share = np.array([2.0, 1.0, 3.0])
    scales = np.array([4.0, 7.0])
    priors = [scale * share for scale in scales]
    counts = [np.asarray(operator @ prior).ravel() for prior in priors]
    timestamps = ['2030-06-01T15:00:00Z', '2030-06-01T16:00:00Z']
    solved = pool_prior_shares(operator, counts, priors, timestamps, regularization=0.05)
    assert solved['status'] == 'solved' and solved['adjacency_used'] is False
    for recovered, truth in zip(solved['flows'], priors):
        np.testing.assert_allclose(recovered, truth, rtol=1e-4, atol=1e-3)
    rank_deficient = csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    null = np.array([1.0, -1.0, 0.0])
    base = np.array([4.0, 4.0, 2.0])
    truths = [base + 3 * null, base - 2 * null]
    blind = pool_prior_shares(
        rank_deficient,
        [np.asarray(rank_deficient @ truth).ravel() for truth in truths],
        [base, base],
        timestamps,
        regularization=0.05,
    )
    assert np.linalg.norm(blind['flows'][0] - base) < np.linalg.norm(blind['flows'][0] - truths[0])
    cosine = float(np.dot(truths[0], truths[1]) / (np.linalg.norm(truths[0]) * np.linalg.norm(truths[1])))
    assert cosine < 0.99


def test_zero_smoothness_matches_separate_solves_and_large_systems_are_refused():
    operator = csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    counts = [np.array([5.0, 2.0]), np.array([8.0, 1.0])]
    priors = [np.array([3.0, 2.0, 1.0]), np.array([4.0, 1.0, 2.0])]
    timestamps = ['2030-06-01T15:00:00Z', '2030-06-01T16:00:00Z']
    smoothed = experimental_correction_smooth(
        operator, counts, priors, timestamps, regularization=0.05, smoothness=0.0, max_gap='2h'
    )
    assert smoothed['edges'] == [(0, 1)]
    for period in range(2):
        separate, _ = solve_weighted_od(operator, counts[period], priors[period], regularization=0.05)
        np.testing.assert_allclose(smoothed['flows'][period], separate, rtol=1e-4, atol=1e-3)
    refused = experimental_correction_smooth(
        csr_matrix(np.eye(30)),
        [np.ones(30), np.ones(30)],
        [np.ones(30), np.ones(30)],
        timestamps,
        column_cap=40,
    )
    assert refused['status'] == 'refused_dense_factorization' and refused['flows'] is None


def test_synthetic_truth_does_not_read_a_prior_and_unobserved_counts_stay_missing():
    assert 'prior' not in inspect.signature(independent_lognormal_truth).parameters
    truth = independent_lognormal_truth(6, np.random.default_rng(11))
    truths, priors, timestamps, assumptions = temporal_smoothed_biased_shares(
        6, 3, np.array(['Z0', 'Z1', 'Z0', 'Z2', 'Z1', 'Z2']), np.random.default_rng(12)
    )
    assert assumptions['uses_future_latent_share'] is False and assumptions['task2_forecast'] is False
    assert list(timestamps) == sorted(timestamps)
    assert all(np.isfinite(item).all() and np.all(item >= 0) for item in truths + priors)
    operator = csr_matrix(np.eye(4))
    counts = counts_from_paths(operator, truth[:4], np.array([0, 2]))
    assert np.isnan(counts[[1, 3]]).all() and np.isfinite(counts[[0, 2]]).all()


def test_attraction_comparison_needs_zone_time_and_unit_alignment():
    prior = pd.DataFrame({
        'path_id': ['p0', 'p1'],
        'destination_zone': ['Z1', 'Z2'],
        'path_flow': [30.0, 10.0],
        'departure_time': ['2030-06-01T15:00:00Z', '2030-06-01T15:00:00Z'],
        'unit': ['vehicles', 'vehicles'],
    })
    aligned = pd.DataFrame({
        'destination_zone': ['Z2', 'Z1'],
        'departure_time': ['2030-06-01T15:00:00Z', '2030-06-01T15:00:00Z'],
        'unit': ['vehicles', 'vehicles'],
        'count': [20.0, 60.0],
    })
    compared = attraction_observability(prior, aligned)
    assert compared['comparison_performed'] and compared['prior_attraction_vs_offramp_l1'] == pytest.approx(0.0)
    assert compared['official_s_attr'] is False
    mismatched = aligned.copy()
    mismatched.loc[0, 'destination_zone'] = 'Z9'
    refused = attraction_observability(prior, mismatched)
    assert refused['status'] == 'unidentifiable' and not refused['comparison_performed']
    token_prior = prior.drop(columns=['unit'])
    token_prior['departure_time'] = 'SYN_PM_TRAIN'
    counts = pd.DataFrame({'link_id': ['L1', 'ROFF-Z1'], 'count': [5.0, np.nan]})
    incidence = pd.DataFrame({'link_id': ['L1', 'ROFF-Z1', 'CONN-1'], 'path_id': ['p0', 'p0', 'p0']})
    paths = pd.DataFrame({'path_id': ['p0'], 'origin_zone': ['Z0'], 'destination_zone': ['Z1']})
    topology = pd.DataFrame({
        'off_ramp_link_ids': ['SYN_RAMP_1'],
        'on_ramp_link_ids': [''],
        'incoming_link_ids': ['CONN-9'],
        'outgoing_link_ids': ['L1'],
    })
    report = inventory_tables({
        'panel': 'fixture',
        'paths': paths,
        'incidence': incidence,
        'counts': counts.dropna(),
        'prior': token_prior,
        'topology': topology,
        'attachment': pd.DataFrame({'ramp_link_id': ['SYN_RAMP_1'], 'ramp_type': ['FR']}),
        'ramp_info': {'has_destination_zone': False, 'sampled_ramp_ids': ['SYN_RAMP_1']},
        'offramp_counts': None,
    })
    assert report['attraction']['comparison_performed'] is False
    assert report['coverage']['connector_counts'] is None
    assert report['coverage']['unknown_connectors_imputed_as_zero'] is False
    assert report['coverage']['connectors_with_counts'] == 0
    assert report['temporal']['pooling'] == 'not_applicable_single_static_interval'
    assert report['counts']['missing_counts_filled_as_zero'] is False
    assert 'ROFF-<zone>' in ' '.join(report['attraction']['missing_fields']) or any(
        'ROFF' in field for field in report['attraction']['missing_fields']
    )
    with pytest.raises(ValueError):
        load_train_panel('unused', 'D12_I405_N', split='validation')
    reindexed = align_counts(pd.DataFrame({'link_id': ['L1'], 'count': [4.0]}), ['L1', 'CONN-1'])
    assert reindexed[0] == 4 and np.isnan(reindexed[1])


def test_outer_assessment_does_not_train_on_held_counts():
    operator = csr_matrix(np.eye(8))
    counts = np.linspace(10, 24, 8)
    prior = np.full(8, 12.0)
    groups, _ = repeated_holdouts(8, 2, 0.15, np.random.default_rng([16, 17]))
    assessment = nested_link_assessment(operator, counts, prior, folds=2, seed=16)
    for fold, held_local in enumerate(groups):
        held = held_local
        train = np.setdiff1d(np.arange(8), held)
        masked = np.full(8, np.nan)
        masked[train] = counts[train]
        flows, _ = solve_weighted_od(operator, masked, prior, regularization=0.05, weight_mode='absolute')
        leaked, _ = solve_weighted_od(operator, counts, prior, regularization=0.05, weight_mode='absolute')
        np.testing.assert_allclose(flows[held], prior[held], atol=1e-6)
        assert np.max(np.abs(flows[held] - leaked[held])) > 1
        reported = next(row for row in assessment['outer'] if row['fold'] == fold and row['model'] == 'baseline')
        assert reported['held_link_l1'] == pytest.approx(held_link_l1(operator, flows, counts, held))
        assert reported['n_train'] + reported['n_held'] == 8
