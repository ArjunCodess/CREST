import numpy as np
import pandas as pd
import pytest
from src.queue_events import collect_events, visible_event, event_features, decode_queue


def series():
    times = pd.date_range("2030-01-01", periods=900, freq="5min", tz="UTC")
    values = np.ones((900, 4, 2), dtype=float) * [100., 3000.]
    for start in (80, 250, 420, 590, 760):
        values[start:start+20, :2, 0] = 30
    auxiliary = np.ones((900, 4, 3))
    return times, values, auxiliary


def test_event_sampler_covers_onset_offsets_and_respects_blocks():
    times, values, auxiliary = series()
    active = np.zeros(len(times), dtype=bool); active[40:850] = True
    offsets = set()
    for seed in range(20):
        events = collect_events(values, auxiliary, times, active, np.full(4, 60.),
            np.random.default_rng(seed), max_per_condition=20)
        for condition in ("queue_onset", "queue_ongoing"):
            selected = [e for e in events if e.condition == condition]
            origins = [e.origin for e in selected]
            assert all(abs(a-b) >= 72 for i, a in enumerate(origins) for b in origins[i+1:])
            for e in selected:
                assert active[e.origin-12:e.origin+7].all()
                assert e.truth.any()
                if condition == "queue_onset":
                    assert not (e.history[:-1, :, 0] <= 60).any()
                    offsets.add(e.first_queue_step)
    assert offsets == set(range(1, 7))


def test_origin_policy_hides_every_channel_and_preserves_labels():
    times, values, auxiliary = series()
    e = collect_events(values, auxiliary, times, np.ones(900, bool), np.full(4, 60.),
        np.random.default_rng(1))[0]
    h, a = visible_event(e, "history_only")
    assert np.isnan(h[-1]).all() and np.isnan(a[-1]).all()
    np.testing.assert_array_equal(h[:-1], e.history[:-1])
    assert np.isfinite(e.history[-1]).all()
    with pytest.raises(ValueError):
        visible_event(e, "future")


def test_event_features_match_after_hiding_origin_and_align_horizon():
    times, values, auxiliary = series()
    e = collect_events(values, auxiliary, times, np.ones(900, bool), np.full(4, 60.),
        np.random.default_rng(3))[0]
    params = pd.DataFrame({"length_km": [1.]*4, "lanes": [3.]*4,
        "free_speed_kmh": [100.]*4, "capacity_vph": [6000.]*4, "k_jam": [300.]*4})
    up = np.eye(4, k=-1); down = up.T
    h, a = visible_event(e, "history_only")
    x = event_features(h, a, e.times, up, down, params)
    e.history[-1] = 1e9; e.auxiliary[-1] = 1e9
    h2, a2 = visible_event(e, "history_only")
    np.testing.assert_allclose(x, event_features(h2, a2, e.times, up, down, params), equal_nan=True)
    assert x.shape[0] == 24
    np.testing.assert_allclose(x[:, -1], np.repeat(np.arange(1, 7)/6, 4))
    with pytest.raises(ValueError):
        event_features(h[:-1], a[:-1], e.times[:-1], up, down, params)


def test_iou_decoder_selects_probability_region_and_is_nonempty():
    p = np.array([[.95, .8, .01], [.9, .7, .01]])
    pred = decode_queue(p, "expected_iou")
    np.testing.assert_array_equal(pred, p > .5)
    assert decode_queue(np.zeros((6, 4)), "expected_iou").sum() == 1
    assert decode_queue(np.zeros((6, 4)), .5).sum() == 1


def test_released_history_cannot_read_future_or_monthly_past():
    from src.queue_events import history_arrays
    origin = pd.Timestamp("2031-03-01T01:00:00Z")
    history = pd.DataFrame({"timestamp": pd.date_range(origin-pd.Timedelta(minutes=60), periods=12, freq="5min"),
        "link_id": "a", "speed_kmh": 100., "flow_vph": 2000., "occupancy": .1,
        "pct_observed": 100., "is_score_eligible": True})
    monthly = pd.DataFrame({"timestamp": [origin-pd.Timedelta(minutes=5), origin, origin+pd.Timedelta(minutes=5)],
        "link_id": "a", "speed_kmh": [1e6, 80., 1e6], "flow_vph": 3000., "occupancy": .2,
        "pct_observed": 100., "is_score_eligible": True})
    h, aux, times = history_arrays(history, monthly, origin, ["a"], "origin_observed", np.array([285.]))
    assert h[-2, 0, 0] == 100 and h[-1, 0, 0] == 80
    np.testing.assert_allclose(aux[:, 0, 1], np.r_[np.full(12, 28.5), 57.])
    monthly.loc[monthly.timestamp != origin, ["speed_kmh", "flow_vph", "occupancy"]] = -1e9
    h2, aux2, _ = history_arrays(history, monthly, origin, ["a"], "origin_observed", np.array([285.]))
    np.testing.assert_allclose(h, h2)
    np.testing.assert_allclose(aux, aux2)
    h, aux, _ = history_arrays(history, monthly, origin, ["a"], "history_only", np.array([285.]))
    assert np.isnan(h[-1]).all() and np.isnan(aux[-1]).all()
    monthly.loc[monthly.timestamp == origin, "pct_observed"] = 50.
    h, aux, _ = history_arrays(history, monthly, origin, ["a"], "origin_observed", np.array([285.]))
    assert np.isnan(h[-1]).all() and np.isnan(aux[-1, :, :2]).all()


def test_origin_measurement_masks_preserve_released_quality():
    times, values, auxiliary = series()
    events = collect_events(values, auxiliary, times, np.ones(900, bool), np.full(4, 60.),
        np.random.default_rng(7))
    for event in events:
        history, aux = visible_event(event, "origin_observed")
        hidden = ~np.isfinite(history[-1, :, 0])
        assert np.isnan(aux[-1, hidden, :2]).all()
        np.testing.assert_array_equal(aux[-1, :, 2], event.auxiliary[-1, :, 2])
