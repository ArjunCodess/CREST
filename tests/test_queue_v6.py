import numpy as np
import pandas as pd
import pytest
from src.queue_events import QueueEvent
from src.latent_queue_v6 import (
    NoiseModel, Q_LEVEL_MIN, Q_SLOPE_MIN, assert_events_before, estimate_noise,
    directed_neighbor_signals, forecast_window, horizon_targets, kalman_filter,
    linear_shortcut_variance, mask_low_quality, predict_step, propagate_trend, queue_v6_features,
    smooth_series, smooth_training_series, update_step, _filter_batch, _smooth_batch,
    QueueArmModel,
)
from src.queue_scenarios_v6 import (
    OnsetSeedModel, aggregate_macro, best_exact_set, brute_min_energy,
    directed_offsets, exact_expected_iou, flip_rates,
    footprint_summary, graph_cut, greedy_expected_iou, integer_energy, match_marginals,
    paired_day_block_bootstrap, scenario_decode, scaled_costs,
    seed_footprint_decode,
)


def params(n=3, free=100.):
    return pd.DataFrame({
        'length_km': np.ones(n), 'lanes': np.full(n, 3.), 'free_speed_kmh': np.full(n, free),
        'capacity_vph': np.full(n, 6000.), 'k_jam': np.full(n, 300.),
    })


def noise_for(n, observation=1e-4, level=1e-5, slope=1e-6):
    return NoiseModel(np.full(n, observation), np.full(n, level), np.full(n, slope),
                      np.full(n, np.sqrt(observation)), np.full(n, np.sqrt(observation)),
                      ['test'] * n, ['test'] * n)


def operators_from(adjacency):
    def normalize(matrix):
        return matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    adjacency = np.asarray(adjacency, float)
    return normalize(adjacency.T), normalize(adjacency)


def make_event(origin, condition, truth, history=None, when='2030-10-15T00:00:00Z'):
    links = truth.shape[1]
    times = pd.date_range(when, periods=13, freq='5min', tz='UTC')
    if history is None:
        history = np.ones((13, links, 2)) * [90., 2500.]
    auxiliary = np.ones((13, links, 3))
    auxiliary[..., 2] = 100
    return QueueEvent(origin, condition, history, auxiliary, times, np.asarray(truth, bool),
                      np.ones_like(truth, bool), 6 if condition == 'queue_onset' else 1)


def test_horizon_covariance_matches_iteration_and_rejects_linear_shortcut():
    mean = np.array([0.4, -0.05])
    cov = np.array([[0.2, 0.03], [0.03, 0.04]])
    q_level, q_slope = np.array([0.01]), np.array([0.002])
    level, level_var, slope, slope_var = propagate_trend(mean, cov, q_level, q_slope)
    state = mean[None].copy()
    state_cov = cov[None].copy()
    iterated_level, iterated_var, iterated_slope, iterated_slope_var = [], [], [], []
    for _ in range(6):
        state, state_cov = predict_step(state, state_cov, q_level, q_slope)
        iterated_level.append(state[0, 0])
        iterated_var.append(state_cov[0, 0, 0])
        iterated_slope.append(state[0, 1])
        iterated_slope_var.append(state_cov[0, 1, 1])
    np.testing.assert_allclose(level, iterated_level)
    np.testing.assert_allclose(level_var, iterated_var)
    np.testing.assert_allclose(slope, iterated_slope)
    np.testing.assert_allclose(slope_var, iterated_slope_var)
    shortcut = linear_shortcut_variance(cov[None], q_level)
    assert np.max(np.abs(level_var - shortcut[:, 0])) > 1e-3
    one = predict_step(np.array([[1., 0.2]]), np.array([[[0.4, 0.1], [0.1, 0.3]]]), np.array([0.05]), np.array([0.02]))
    np.testing.assert_allclose(one[0], [[1.2, 0.2]])
    np.testing.assert_allclose(one[1][0], [[0.95, 0.4], [0.4, 0.32]])
    updated, updated_cov = update_step(np.zeros((1, 2)), np.eye(2)[None], np.array([1.]), np.array([1.]))
    np.testing.assert_allclose(updated, [[0.5, 0.]])
    np.testing.assert_allclose(updated_cov[0], [[0.5, 0.], [0., 1.]])


def test_filter_is_causal_handles_missing_and_smoother_stays_psd():
    spec = noise_for(1, 1e-3, 1e-4, 1e-5)
    series = np.linspace(1.0, 0.7, 16)[:, None]
    series[5] = np.nan
    means, covs = kalman_filter(series, spec)
    predicted, _ = predict_step(means[4], covs[4], spec.level_process_variance, spec.slope_process_variance)
    np.testing.assert_allclose(means[5], predicted)
    shocked = series.copy()
    shocked[10:] = 0.1
    shocked_means, _ = kalman_filter(shocked, spec)
    np.testing.assert_allclose(means[:10], shocked_means[:10])
    original = smooth_series(series, spec)
    changed = smooth_series(shocked, spec)
    assert not np.allclose(original.level[4], changed.level[4])
    np.testing.assert_allclose(original.level[-1], means[-1, :, 0])
    _, _, _, smoothed_cov = _smooth_batch(*_filter_batch(
        series, spec.observation_variance, spec.level_process_variance, spec.slope_process_variance),
        spec.level_process_variance, spec.slope_process_variance)
    eigenvalues = np.linalg.eigvalsh(smoothed_cov)
    assert np.min(eigenvalues) >= -1e-8
    assert np.all((original.probability >= 0) & (original.probability <= 1))
    low = smooth_series(np.full((24, 1), 0.2), spec)
    high = smooth_series(np.full((24, 1), 1.0), spec)
    assert low.probability[-1, 0] > 0.9
    assert high.probability[-1, 0] < 0.1
    assert low.hard[-1, 0] and not high.hard[-1, 0]


def test_training_noise_ignores_a_held_out_suffix_and_stays_positive():
    rng = np.random.default_rng(11)
    train = 1 + rng.normal(0, 0.02, size=(180, 2))
    first = estimate_noise(train)
    altered = 1 + rng.normal(0, 0.25, size=(200, 2))
    second = estimate_noise(train.copy())
    np.testing.assert_allclose(first.observation_variance, second.observation_variance)
    np.testing.assert_allclose(first.level_process_variance, second.level_process_variance)
    third = estimate_noise(np.vstack([train, altered]))
    assert not np.allclose(first.observation_variance, third.observation_variance)
    assert np.all(first.observation_variance > 0)
    assert np.all(first.level_process_variance >= Q_LEVEL_MIN)
    assert np.all(first.slope_process_variance >= Q_SLOPE_MIN)
    white = 1 + rng.normal(0, 0.02, size=(160, 1))
    white_noise = estimate_noise(white)
    assert white_noise.process_source[0] == 'second_difference'
    assert white_noise.level_process_variance[0] < 5e-4
    assert white_noise.slope_process_variance[0] < 5e-4
    times = pd.date_range('2030-09-01', periods=len(train), freq='5min', tz='UTC')
    smooth_training_series(train, first, times, '2030-10-01T00:00:00Z')
    with pytest.raises(ValueError):
        smooth_training_series(train, first, times, times[20])


def test_forecast_restarts_inside_the_window_and_drops_low_quality():
    spec = noise_for(1, 1e-4, Q_LEVEL_MIN, Q_SLOPE_MIN)
    long = np.full((40, 1), 0.3)
    long[-13:] = 0.95
    fresh = forecast_window(long[-13:], spec)
    window_mean, window_cov = kalman_filter(long[-13:], spec)
    restarted, _, _, _ = propagate_trend(
        window_mean[-1, 0], window_cov[-1, 0], spec.level_process_variance, spec.slope_process_variance)
    np.testing.assert_allclose(fresh.level_mean[:, 0], restarted)
    warm_mean, warm_cov = kalman_filter(long, spec)
    carried, _, _, _ = propagate_trend(
        warm_mean[-1, 0], warm_cov[-1, 0], spec.level_process_variance, spec.slope_process_variance)
    assert abs(fresh.level_mean.mean() - carried.mean()) > 0.05
    with pytest.raises(ValueError):
        forecast_window(long, spec)
    values = np.ones((13, 1, 2)) * [100., 2000.]
    values[-6:, 0, 0] = 20
    auxiliary = np.ones((13, 1, 3)) * [5., 10., 100.]
    trusted, _ = mask_low_quality(values, auxiliary)
    hidden_aux = auxiliary.copy()
    hidden_aux[-6:, 0, 2] = 10
    hidden, _ = mask_low_quality(values, hidden_aux)
    assert not np.isfinite(hidden[-6:, 0, 0]).any()
    assert np.isfinite(hidden[:-6, 0, 0]).all()
    trusted_forecast = forecast_window(trusted[..., 0] / 100, spec)
    hidden_forecast = forecast_window(hidden[..., 0] / 100, spec)
    assert trusted_forecast.probability.mean() > hidden_forecast.probability.mean() + 0.2


def test_neighbor_signals_use_directed_edges_not_index_order():
    adjacency = np.zeros((3, 3))
    adjacency[0, 2] = 1
    adjacency[2, 1] = 1
    up, down = operators_from(adjacency)
    level = np.ones((6, 3))
    level[:, 2] = 0.2
    sd = np.full((6, 3), 0.1)
    slope = np.zeros((6, 3))
    probability = np.full((6, 3), 0.2)
    probability[:, 2] = 0.9
    upstream, downstream = directed_neighbor_signals(level, sd, slope, probability, up, down)
    assert downstream[0, 0, 0] == pytest.approx(0.2)
    assert upstream[0, 1, 0] == pytest.approx(0.2)
    assert abs(0 - 1) == 1
    offsets = directed_offsets(adjacency)
    assert offsets[0, 1] == 2
    assert offsets[0, 2] == 1
    assert offsets[1, 0] == -2
    report = footprint_summary([np.array([True, True, True])], adjacency, min_components=1)
    assert '-2' in report['offset_counts']
    assert report['unsupported_components'] == 0
    branch = np.zeros((3, 3))
    branch[0, 1] = 1
    branch[0, 2] = 1
    unsupported = footprint_summary([np.ones(3, bool)], branch, min_components=1)
    assert unsupported['unsupported_components'] == 1
    disconnected = footprint_summary([np.array([True, True, False])], np.zeros((3, 3)), min_components=1)
    assert disconnected['disconnected_windows'] == 1
    assert disconnected['supported_components'] == 2


def test_exact_expected_iou_scenario_decoder_and_onset_prefix():
    assert exact_expected_iou(np.array([0.5, 0.5]), np.array([True, False])) == pytest.approx(0.375)
    best, value = best_exact_set(np.array([0.95, 0.05]))
    np.testing.assert_array_equal(best, [True, False])
    assert value == pytest.approx(exact_expected_iou([0.95, 0.05], best))
    greedy = greedy_expected_iou(np.array([0.95, 0.05, 0.02]), n_scenarios=400, seed=7)
    np.testing.assert_array_equal(greedy, [True, False, False])
    repeat = greedy_expected_iou(np.array([0.95, 0.05, 0.02]), n_scenarios=400, seed=7)
    np.testing.assert_array_equal(greedy, repeat)
    probability = np.full((6, 4), 0.9)
    decoded = scenario_decode(probability, 'queue_onset', n_scenarios=30, seed=3)
    assert not decoded[:-1].any()
    assert decoded[-1].any()
    again = scenario_decode(probability, 'queue_onset', n_scenarios=30, seed=3)
    np.testing.assert_array_equal(decoded, again)


def test_graph_cut_matches_brute_force_energy_and_unary_sign():
    unary, beta_i = scaled_costs(np.array([0.2, 0.8, 0.1]), 0.5, 0, 10)
    labels = graph_cut(np.array([0.2, 0.8, 0.1]), [], threshold=0.5, beta=0, scale=10)
    np.testing.assert_array_equal(labels, [False, True, False])
    assert integer_energy(labels, unary, [], beta_i) == integer_energy(
        brute_min_energy(unary, [], beta_i)[0], unary, [], beta_i)
    directed = graph_cut(np.array([0.8, 0.1, 0.8]), [(0, 2)], threshold=0.5, beta=1, scale=10)
    np.testing.assert_array_equal(directed, [True, False, True])
    lexical = graph_cut(np.array([0.8, 0.1, 0.8]), [(0, 1), (1, 2)], threshold=0.5, beta=1, scale=10)
    np.testing.assert_array_equal(lexical, [True, True, True])
    rng = np.random.default_rng(19)
    for _ in range(6):
        width = 4
        probability = rng.uniform(0.05, 0.95, size=width)
        pairs = [(int(a), int(b)) for a, b in rng.integers(0, width, size=(3, 2)) if a != b]
        unary, beta_i = scaled_costs(probability, 0.5, 0.35, 100)
        cut = graph_cut(probability, pairs, threshold=0.5, beta=0.35, scale=100)
        brute, energy = brute_min_energy(unary, pairs, beta_i)
        assert integer_energy(cut, unary, pairs, beta_i) == energy
        assert integer_energy(brute, unary, pairs, beta_i) == energy


def test_seed_footprint_decoder_follows_directed_topology():
    adjacency = np.zeros((3, 3))
    adjacency[0, 2] = adjacency[2, 1] = 1
    footprint = {'status': 'empirical', 'inclusion_by_offset': {'0': 1., '-1': 1.}}
    chosen, report = seed_footprint_decode(np.array([0., 1., 0.]), np.array([0., 1., 1.]),
        directed_offsets(adjacency), footprint, n_scenarios=20)
    np.testing.assert_array_equal(chosen, [False, True, True])
    assert report['marginals_reproduced'] and report['l1_gap'] == pytest.approx(0)
    fallback, report = seed_footprint_decode(np.array([0., 1., 0.]), np.array([0., 1., 0.]),
        directed_offsets(adjacency), None, n_scenarios=20)
    np.testing.assert_array_equal(fallback, [False, True, False])
    assert report['joint'] == 'independent_fallback'


def test_marginal_match_reports_an_infeasible_joint():
    fitted, gap, matched = match_marginals(np.array([0.5, 0.5]), np.full((2, 2), 0.4), np.array([0.4, 0.4]))
    assert matched and gap <= 0.05
    _, gap, matched = match_marginals(np.array([0.8, 0.2]), np.eye(2), np.array([0.5, 0.5]))
    assert not matched and gap > 0.05
    assert np.all((fitted >= 0) & (fitted <= 1))


def test_flip_share_macro_weights_and_paired_difference():
    raw = np.zeros((2, 3), bool)
    raw[0, 0] = True
    denoised = raw.copy()
    denoised[0, 1] = True
    valid = np.ones_like(raw)
    rates = flip_rates(raw, denoised, valid)
    assert rates[0]['flip_share'] == pytest.approx(1 / 3)
    assert rates[0]['union_flip_rate'] == pytest.approx(0.5)
    rows = [
        {'panel': 'D12_I5_N', 'family': 'D12_I5', 'condition': 'queue_onset', 'day': '2030-10-15', 'candidate': 0.2, 'baseline': 0.2},
        {'panel': 'D12_I5_N', 'family': 'D12_I5', 'condition': 'queue_ongoing', 'day': '2030-10-16', 'candidate': 0.8, 'baseline': 0.8},
        {'panel': 'D7_I10_E', 'family': 'D7_I10', 'condition': 'queue_onset', 'day': '2030-10-15', 'candidate': 0.0, 'baseline': 0.0},
        {'panel': 'D7_I10_E', 'family': 'D7_I10', 'condition': 'queue_ongoing', 'day': '2030-10-16', 'candidate': 0.0, 'baseline': 0.0},
    ]
    assert aggregate_macro(rows, 'candidate')['overall'] == pytest.approx(0.25)
    report = paired_day_block_bootstrap(rows, 'candidate', 'baseline', n_boot=30, seed=1)
    assert report['delta_point'] == pytest.approx(0)
    assert report['delta_p05'] == pytest.approx(0)
    assert report['delta_p95'] == pytest.approx(0)
    assert 'standard_error' not in report


def test_bootstrap_retains_condition_weights_when_days_do_not_overlap():
    rows = [
        {'panel': 'D12_I5_N', 'family': 'D12_I5', 'condition': 'queue_onset',
         'day': '2030-10-15', 'candidate': .8, 'baseline': .2},
        {'panel': 'D12_I5_N', 'family': 'D12_I5', 'condition': 'queue_ongoing',
         'day': '2030-10-16', 'candidate': .1, 'baseline': .3},
    ]
    report = paired_day_block_bootstrap(rows, 'candidate', 'baseline', n_boot=100)
    # Constant differences inside each stratum imply no sampling variation.
    assert report['delta_point'] == pytest.approx(.2)
    assert report['delta_p05'] == pytest.approx(.2)
    assert report['delta_p95'] == pytest.approx(.2)


def test_arms_keep_forecast_shapes_and_ignore_unpassed_labels():
    spec = noise_for(3)
    frame = params()
    adjacency = np.zeros((3, 3))
    adjacency[0, 2] = 1
    adjacency[2, 1] = 1
    up, down = operators_from(adjacency)
    ongoing, onset = [], []
    for index in range(6):
        truth = np.zeros((6, 3), bool)
        truth[index % 6, index % 3] = True
        truth[-1, :] = index % 2 == 0
        history = np.ones((13, 3, 2)) * [90., 2500.]
        history[:, index % 3, 0] = 40 + index
        ongoing.append(make_event(20 + index, 'queue_ongoing', truth, history, when='2030-09-10T00:00:00Z'))
        onset_truth = np.zeros((6, 3), bool)
        onset_truth[-1, 0] = True
        onset_truth[-1, 2] = index % 2 == 0
        onset.append(make_event(30 + index, 'queue_onset', onset_truth, history, when='2030-10-16T08:00:00Z'))
    targets = {(event.origin, event.condition): event.truth.astype(float) for event in ongoing}
    model = QueueArmModel(arm='raw', rounds=8, rows_per_window=24, min_child_samples=1, seed=4)
    model.fit(ongoing, up, down, frame, spec, targets)
    scored = model.probabilities(ongoing[0], up, down, frame)
    assert scored.shape == (6, 3)
    again = QueueArmModel(arm='raw', rounds=8, rows_per_window=24, min_child_samples=1, seed=4)
    rogue = dict(targets)
    rogue[(999, 'queue_ongoing')] = np.ones((6, 3))
    again.fit(ongoing, up, down, frame, spec, rogue)
    np.testing.assert_allclose(scored, again.probabilities(ongoing[0], up, down, frame))
    flipped = {key: 1 - value for key, value in targets.items()}
    changed = QueueArmModel(arm='raw', rounds=8, rows_per_window=24, min_child_samples=1, seed=4)
    changed.fit(ongoing, up, down, frame, spec, flipped)
    assert not np.allclose(scored, changed.probabilities(ongoing[0], up, down, frame))
    soft_targets = {key: np.clip(value * 0.7 + 0.1, 0, 1) for key, value in targets.items()}
    soft = QueueArmModel(arm='smoothed_soft', rounds=8, rows_per_window=24, min_child_samples=1, seed=4)
    soft.fit(ongoing, up, down, frame, spec, soft_targets)
    soft_scores = soft.probabilities(ongoing[0], up, down, frame)
    assert soft_scores.shape == (6, 3)
    assert np.all((soft_scores >= 0) & (soft_scores <= 1))
    early = [event for event in onset]
    late = []
    for event in onset:
        truth = event.truth.copy()
        truth[:-1] = ~truth[:-1]
        late.append(QueueEvent(event.origin, event.condition, event.history, event.auxiliary, event.times,
                                truth, event.valid, event.first_queue_step))
    onset_targets = {(event.origin, event.condition): event.truth.astype(float) for event in early}
    onset_targets_b = {(event.origin, event.condition): event.truth.astype(float) for event in late}
    left = QueueArmModel(arm='raw', rounds=6, rows_per_window=20, min_child_samples=1, seed=5)
    right = QueueArmModel(arm='raw', rounds=6, rows_per_window=20, min_child_samples=1, seed=5)
    left.fit(early, up, down, frame, spec, onset_targets)
    right.fit(late, up, down, frame, spec, onset_targets_b)
    left_scores = left.probabilities(early[0], up, down, frame)
    np.testing.assert_allclose(left_scores, right.probabilities(early[0], up, down, frame))
    assert left_scores.shape == (6, 3) and not left_scores[:-1].any()
    hidden = early[0]
    visible = queue_v6_features(*__import__('src.queue_events', fromlist=['visible_event']).visible_event(hidden, 'history_only'),
                                 hidden.times, up, down, frame, spec)
    hidden.history[-1] = -50
    hidden.auxiliary[-1] = 80
    mutated = queue_v6_features(*__import__('src.queue_events', fromlist=['visible_event']).visible_event(hidden, 'history_only'),
                                 hidden.times, up, down, frame, spec)
    np.testing.assert_allclose(visible, mutated, equal_nan=True)
    master = pd.date_range('2030-09-01', periods=80, freq='5min', tz='UTC')
    sample = make_event(12, 'queue_ongoing', np.zeros((6, 3), bool), when=master[0])
    assert_events_before([sample], master, '2030-10-01T00:00:00Z')
    with pytest.raises(ValueError):
        assert_events_before([sample], master, master[12])
    smooth = smooth_series(np.full((40, 3), 0.9), spec)
    sliced = horizon_targets(QueueEvent(12, 'queue_ongoing', sample.history, sample.auxiliary, sample.times,
                                         sample.truth, sample.valid, 1), smooth.probability, smooth.hard)
    assert sliced['smoothed_hard'].shape == (6, 3)
    small = OnsetSeedModel(rounds=5, min_groups=8, min_child_samples=1, seed=6)
    small.fit(onset[:3], up, down, frame, spec, adjacency)
    assert small.status == 'skipped_small_sample' and small.footprint['robust'] is False
    ranker = OnsetSeedModel(rounds=8, min_groups=4, folds=2, min_child_samples=1, seed=6)
    ranker.fit(onset[:4], up, down, frame, spec, adjacency)
    assert ranker.status == 'fitted'
    assert ranker.max_positives_in_group >= 2
    seed_scores = ranker.probabilities(onset[0], up, down, frame, spec)
    assert seed_scores.shape == (6, 3) and not seed_scores[:-1].any()
    assert seed_scores[-1].sum() == pytest.approx(1)
    november = [make_event(40, 'queue_onset', onset[0].truth, when='2030-11-03T00:00:00Z')]
    with pytest.raises(ValueError):
        ranker.select_temperature(november, up, down, frame, spec)
    selected = ranker.select_temperature(onset[:4], up, down, frame, spec)
    assert selected['temperature'] in (0.25, 0.5, 1, 2, 4, 8)
    assert selected['unique_seed_nll'] is None
