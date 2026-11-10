"""Onset localization under the organizer's published T+30-only contract."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from .queue_events import QueueEvent, event_features, visible_event, decode_queue


def onset_events(values, auxiliary, times, active, cutoff, max_events=512, seed=540):
    finite = np.isfinite(values[..., 0])
    queue = finite & (values[..., 0] <= np.asarray(cutoff)[None])
    any_queue = queue.any(axis=1)
    cumulative = np.r_[0, np.cumsum(any_queue)]
    active_sum = np.r_[0, np.cumsum(np.asarray(active, bool))]
    # A first queue at s has origin s-6. Every earlier horizon step is clear.
    starts = np.arange(18, len(values))
    starts = starts[any_queue[starts] & (cumulative[starts] - cumulative[starts-18] == 0)]
    starts = starts[active_sum[starts+1] - active_sum[starts-18] == 19]
    starts = np.asarray([s for s in starts if finite[s-18:s-6].mean() >= .7])
    rng = np.random.default_rng(seed)
    if len(starts) > max_events:
        starts = np.sort(rng.choice(starts, max_events, replace=False))
    result = []
    for s in starts:
        t = int(s-6)
        truth = np.zeros((6, values.shape[1]), bool)
        truth[-1] = queue[s]
        result.append(QueueEvent(t, "queue_onset", values[t-12:t+1].copy(),
            auxiliary[t-12:t+1].copy(), pd.DatetimeIndex(times[t-12:t+1]),
            truth, finite[t+1:t+7].copy(), 6))
    return result


def spatial_features(history, auxiliary, times, up, down, params):
    n = len(params)
    base = event_features(history, auxiliary, times, up, down, params)[-n:, :-1]
    identity = np.arange(n, dtype=np.float32) / max(n-1, 1)
    # Identity and directed degree let trees learn recurring bottlenecks.
    static = np.stack([identity, np.sum(up > 0, axis=1), np.sum(down > 0, axis=1)], -1)
    return np.concatenate([base, static], -1).astype(np.float32)


def trajectory_signature(history, params):
    scale = np.stack([
        params.free_speed_kmh.to_numpy(), params.capacity_vph.to_numpy()], -1)
    normalized = history / scale[None]
    blocks = []
    for start, end in ((0, 6), (6, 12), (10, 13)):
        x = normalized[start:end]
        count = np.isfinite(x).sum(0)
        blocks.append(np.divide(np.nansum(x, axis=0), count,
            out=np.full(x.shape[1:], np.nan), where=count > 0))
    return np.concatenate(blocks, -1).ravel().astype(np.float32)


class OnsetLocalizer:
    def __init__(self, policy="origin_observed", rounds=240):
        self.policy, self.rounds = policy, rounds
        self.tree_weight, self.prior_weight, self.neighbors = 1., 0., 12
        self.decoder = "expected_iou"

    def fit(self, events, up, down, params):
        xx, yy, weights, signatures, truths, validity = [], [], [], [], [], []
        for event in events:
            history, auxiliary = visible_event(event, self.policy)
            x = spatial_features(history, auxiliary, event.times, up, down, params)
            valid = event.valid[-1]
            xx.append(x[valid]); yy.append(event.truth[-1, valid])
            weights.append(np.full(valid.sum(), 1 / valid.sum()))
            signatures.append(trajectory_signature(history, params))
            truths.append(event.truth[-1]); validity.append(valid)
        if len(events) < 12:
            raise ValueError("At least twelve training onset episodes are required")
        x, y, weight = np.concatenate(xx), np.concatenate(yy), np.concatenate(weights)
        weight *= len(weight) / weight.sum()
        self.model = LGBMClassifier(n_estimators=self.rounds, learning_rate=.035,
            num_leaves=23, min_child_samples=35, reg_lambda=8,
            colsample_bytree=.9, n_jobs=4, verbosity=-1, random_state=540)
        self.model.fit(x, y.astype(int), sample_weight=weight)
        self.signatures = np.asarray(signatures)
        self.signature_mean = np.nanmean(self.signatures, axis=0)
        self.signature_mean = np.nan_to_num(self.signature_mean)
        self.signature_scale = np.maximum(np.nanstd(self.signatures, axis=0), .08)
        self.signature_scale = np.nan_to_num(self.signature_scale, nan=.08)
        self.signatures = np.nan_to_num((self.signatures-self.signature_mean) / self.signature_scale)
        self.truths, self.validity = np.asarray(truths), np.asarray(validity)
        self.prior = (self.truths.sum(0)+.1) / (self.validity.sum(0)+1.)
        return self

    def components(self, history, auxiliary, times, up, down, params):
        x = spatial_features(history, auxiliary, times, up, down, params)
        tree = self.model.booster_.predict(x, num_threads=4)
        signature = trajectory_signature(history, params)
        valid = np.isfinite(signature)
        standardized = np.nan_to_num((signature-self.signature_mean)/self.signature_scale)
        distance = np.mean((self.signatures[:, valid]-standardized[None, valid])**2, axis=1)
        nearest = np.argsort(distance)[:min(self.neighbors, len(distance))]
        weight = np.exp(-(distance[nearest]-distance[nearest[0]]) / max(float(np.median(distance[nearest])), .1))
        analogue = (weight @ self.truths[nearest]+.1) / (weight @ self.validity[nearest]+1.)
        return tree, analogue, self.prior

    def scores(self, history, auxiliary, times, condition, up, down, params):
        if condition != "queue_onset":
            raise ValueError("Onset model cannot forecast an ongoing window")
        tree, analogue, prior = self.components(history, auxiliary, times, up, down, params)
        p = (1-self.prior_weight)*(self.tree_weight*tree+(1-self.tree_weight)*analogue)+self.prior_weight*prior
        output = np.zeros((6, len(params)))
        output[-1] = p
        return output

    def predict(self, event, up, down, params):
        h, a = visible_event(event, self.policy)
        return decode_onset(self.scores(h, a, event.times, event.condition, up, down, params), self.decoder)


def decode_onset(probabilities, policy="expected_iou"):
    output = np.zeros_like(probabilities, dtype=bool)
    output[-1] = decode_queue(np.asarray(probabilities)[-1:], policy)[0]
    return output
