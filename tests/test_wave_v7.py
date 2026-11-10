import numpy as np
import pandas as pd
from src.wave_v7 import road_distances, wave_context, motion_features


def inputs():
    p = pd.DataFrame({'length_km': [2., 3., 4.], 'free_speed_kmh': [100.]*3,
                      'capacity_vph': [6000.]*3, 'lanes': [3.]*3, 'k_jam': [450.]*3})
    up = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    return up, up.T, p


def test_road_geometry_and_permutation():
    up, _, p = inputs()
    distance = road_distances(up, p)
    assert distance[0, 2] == 6.
    assert np.isinf(distance[2, 0])
    permutation = [2, 0, 1]
    np.testing.assert_allclose(road_distances(up[np.ix_(permutation, permutation)], p.iloc[permutation]), distance[np.ix_(permutation, permutation)])


def test_causal_waves_ignore_future():
    up, _, p = inputs()
    x = np.random.default_rng(7).uniform(1, 90, (25, 3, 2)).astype(np.float32)
    original = wave_context(x, up, p, causal=True)
    x[13:] = 1e6
    np.testing.assert_allclose(wave_context(x, up, p, causal=True)[:13], original[:13], equal_nan=True)


def test_missing_waves_are_missing_and_motion_shape():
    up, down, p = inputs()
    x = np.full((13, 3, 2), np.nan, dtype=np.float32)
    wave = wave_context(x, up, p)
    assert np.isnan(wave[..., :2]).all()
    assert not np.isinf(wave).any()
    x[:] = [80., 4000.]
    times = pd.date_range('2030-01-01', periods=13, freq='5min', tz='UTC')
    result = motion_features(x, np.ones((13, 3, 3)), times, up, down, p)
    assert result.shape[0] == 18
    assert not np.isinf(result).any()


def test_wave_training_replay_and_observation_preservation(tmp_path):
    import pickle
    from scripts.make_fixture import make
    from src.data import load_state, graph
    from src.wave_v7 import WaveReconstructor
    root = make(tmp_path / 'fixture')
    data = load_state(root, 'D12_I5_N', 'train', truth=True)
    up, down, _, params = graph(root, 'D12_I5_N', data.links)
    model = WaveReconstructor(rounds=3, max_rows=120).fit(data.values, data.times, up, down, params)
    masked = data.values[:100].copy()
    masked[20:30, 1] = np.nan
    prediction = model.reconstruct(masked, data.times[:100], up, down, params)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction[np.isfinite(masked)], masked[np.isfinite(masked)])
    replay = pickle.loads(pickle.dumps(model)).reconstruct(masked, data.times[:100], up, down, params)
    np.testing.assert_array_equal(prediction, replay)


def test_motion_classifier_fit_and_replay():
    import pickle
    from src.queue_events import QueueEvent
    from src.wave_v7 import MotionQueueModel
    up, down, params = inputs()
    rng = np.random.default_rng(70)
    times = pd.date_range('2030-01-01', periods=13, freq='5min', tz='UTC')
    events = []
    for condition in ('queue_onset', 'queue_ongoing'):
        for i in range(15):
            history = np.stack([rng.uniform(40, 100, (13, 3)), rng.uniform(2000, 5000, (13, 3))], axis=-1)
            events.append(QueueEvent(i*100, condition, history, np.ones((13, 3, 3)), times,
                rng.random((6, 3)) > .5, np.ones((6, 3), bool), 6))
    model = MotionQueueModel(rounds=3).fit(events, up, down, params)
    event = events[-1]
    p = model.scores(event.history, event.auxiliary, times, event.condition, up, down, params)
    assert p.shape == (6, 3)
    assert ((p >= 0) & (p <= 1)).all()
    other = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(p, other.scores(event.history, event.auxiliary, times, event.condition, up, down, params))
