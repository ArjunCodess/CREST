"""Location-aware causal queue forecasts, fitted only on training episodes."""
import numpy as np
from lightgbm import LGBMClassifier
from .features import neighbors
from .queue_events import EventQueueModel, event_features, visible_event


def localized_features(history, auxiliary, times, up, down, params):
    base = event_features(history, auxiliary, times, up, down, params)
    n = len(params)
    speed = history[..., 0] / params.free_speed_kmh.to_numpy()[None]
    valid = np.isfinite(speed)
    queued = valid & (speed <= .6)
    blocks = [np.stack([np.arange(n) / max(n-1, 1),
        params.length_km.to_numpy(), params.lanes.to_numpy(),
        np.sum(up > 0, axis=1), np.sum(down > 0, axis=1)], -1)]
    for width in (2, 4, 8, 13):
        count = valid[-width:].sum(0)
        occupancy = np.divide(queued[-width:].sum(0), count,
            out=np.full(n, np.nan), where=count > 0)
        mean = np.divide(np.nansum(speed[-width:], axis=0), count,
            out=np.full(n, np.nan), where=count > 0)
        blocks.extend([occupancy[:, None], mean[:, None]])
    latest = np.full(n, np.nan)
    previous = np.full(n, np.nan)
    for t in range(13):
        seen = valid[t]
        previous[seen] = latest[seen]
        latest[seen] = speed[t, seen]
    current = np.stack([latest, latest-previous, np.maximum(.6-latest, 0)], -1)
    blocks.append(current)
    for operator in (up, down):
        spatial = current[None]
        for _ in range(3):
            spatial = neighbors(spatial, operator)
            blocks.append(spatial[0])
    extra = np.tile(np.concatenate(blocks, axis=1), (6, 1))
    return np.concatenate([base, extra], axis=1).astype(np.float32)


class LocalizedQueueModel(EventQueueModel):
    def feature_rows(self, *args):
        return localized_features(*args)

    def __init__(self, policy='origin_observed', rounds=320, rows_per_window=1200):
        super().__init__(policy, rounds, rows_per_window)

    def fit(self, events, up, down, params):
        self.models = {}
        rng = np.random.default_rng(851)
        for condition in ('queue_onset', 'queue_ongoing'):
            xx, yy, weights = [], [], []
            for event in (e for e in events if e.condition == condition):
                h, a = visible_event(event, self.policy)
                x = self.feature_rows(h, a, event.times, up, down, params)
                valid, y = event.valid.ravel(), event.truth.ravel()
                if condition == 'queue_onset':
                    valid = valid.copy()
                    valid[:-len(params)] = False
                selected, weight = [], []
                for label in (True, False):
                    available = np.flatnonzero(valid & (y == label))
                    count = min(len(available), self.rows_per_window // 2)
                    if count:
                        selected.extend(rng.choice(available, count, replace=False))
                        weight.extend([len(available) / count / valid.sum()] * count)
                if selected:
                    xx.append(x[selected]); yy.append(y[selected]); weights.extend(weight)
            x, y, weight = np.concatenate(xx), np.concatenate(yy), np.asarray(weights)
            weight *= len(weight) / weight.sum()
            if len(np.unique(y)) < 2:
                self.models[condition] = float(y[0])
            else:
                self.models[condition] = LGBMClassifier(n_estimators=self.rounds,
                    learning_rate=.035, num_leaves=31, min_child_samples=40,
                    reg_lambda=10, colsample_bytree=.9, subsample=.9,
                    subsample_freq=1, n_jobs=4, random_state=851, verbosity=-1
                ).fit(x, y.astype(int), sample_weight=weight)
        return self

    def scores(self, history, auxiliary, times, condition, up, down, params):
        x = self.feature_rows(history, auxiliary, times, up, down, params)
        model = self.models[condition]
        p = np.full(len(x), model) if isinstance(model, float) else model.booster_.predict(x, num_threads=4)
        result = p.reshape(6, -1)
        if condition == 'queue_onset':
            result[:-1] = 0
        return result


class BlendedQueueModel(EventQueueModel):
    def __init__(self, candidate, baseline, weight, decoder):
        super().__init__(candidate.policy)
        self.candidate, self.baseline, self.weight = candidate, baseline, weight
        self.decoders = dict(baseline.decoders, queue_ongoing=decoder)

    def scores(self, history, auxiliary, times, condition, up, down, params):
        new = self.candidate.scores(history, auxiliary, times, condition, up, down, params)
        if self.weight == 1:
            return new
        old = self.baseline.scores(history, auxiliary, times, condition, up, down, params)
        return (1-self.weight)*old+self.weight*new
