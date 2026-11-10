"""Physical-distance state waves and causal queue-motion challengers."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .features import SCALE, shift, neighbors
from .state_v4 import EnhancedReconstructor, context_rows
from .queue_v5 import LocalizedQueueModel, localized_features


def road_distances(up, params):
    lengths = params.length_km.to_numpy()
    edges = np.asarray(up).T > 0
    costs = edges * (lengths[:, None] + lengths[None, :]) / 2
    return dijkstra(csr_matrix(costs), directed=True, limit=15.)


def wave_context(values, up, params, causal=False, distance=None):
    if distance is None:
        distance = road_distances(up, params)
    output = []
    normalized = values / SCALE
    for distances, speed in ((distance.T, 80.), (distance, 15.)):
        support = np.isfinite(distances) & (distances > 0)
        lag = np.rint(np.where(support, distances, 0) / speed * 12).astype(int)
        weight = np.exp(-np.where(support, distances, 0) / 3) * support
        for sign in ((1,) if causal else (1, -1)):
            numerator = np.zeros_like(normalized)
            denominator = np.zeros_like(normalized)
            for step in range(min(12, len(values)-1)+1):
                operator = csr_matrix(weight * (lag == step))
                x = shift(normalized, sign * step)
                valid = np.isfinite(x)
                for target, source in ((numerator, np.nan_to_num(x)), (denominator, valid.astype(np.float32))):
                    flat = source.transpose(1, 0, 2).reshape(len(params), -1)
                    target += (operator @ flat).reshape(len(params), len(x), 2).transpose(1, 0, 2)
            estimate = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            output.extend([estimate, np.log1p(denominator)])
    return np.concatenate(output, axis=-1)


class WaveReconstructor(EnhancedReconstructor):
    def context(self, values, times, up, down, params, factor, row_slice):
        if not hasattr(self, 'distance'):
            self.distance = road_distances(up, params)
        x, base = context_rows(values, times, up, down, params, factor, row_slice)
        start, end = row_slice
        return np.concatenate([x, wave_context(values, up, params, distance=self.distance)[start:end]], axis=-1), base


def motion_features(history, auxiliary, times, up, down, params, distance=None):
    if len(history) != 13:
        raise ValueError('Motion features require T-60 through T only')
    base = localized_features(history, auxiliary, times, up, down, params)
    speed = history[..., 0] / params.free_speed_kmh.to_numpy()[None]
    if distance is None:
        distance = road_distances(up, params)
    blocks = []
    for width in (3, 6, 13):
        x = speed[-width:]
        observed = np.isfinite(x)
        t = np.arange(width)[:, None]
        count = observed.sum(0)
        mean_t = np.divide((t * observed).sum(0), count, out=np.zeros(speed.shape[1]), where=count > 0)
        dt = t - mean_t
        denom = (observed * dt**2).sum(0)
        slope = np.divide((np.nan_to_num(x) * dt).sum(0), denom, out=np.full(speed.shape[1], np.nan), where=denom > 0)
        blocks.extend([slope[:, None], neighbors(slope[None, :, None], up)[0], neighbors(slope[None, :, None], down)[0]])
    for lag in (0, 3, 6, 12):
        queued = np.isfinite(speed[-1-lag]) & (speed[-1-lag] <= .6)
        for distances in (distance, distance.T):
            nearest = np.min(np.where(queued[None], distances, np.inf), axis=1)
            blocks.extend([np.where(np.isfinite(nearest), nearest / 15, np.nan)[:, None], np.isfinite(nearest)[:, None]])
    blocks.append(wave_context(history, up, params, causal=True, distance=distance)[-1])
    extra = np.tile(np.concatenate(blocks, axis=-1), (6, 1))
    horizon = np.repeat(np.arange(1, 7) / 6, len(params))[:, None]
    return np.concatenate([base, extra, extra * horizon], axis=-1).astype(np.float32)


class MotionQueueModel(LocalizedQueueModel):
    def feature_rows(self, history, auxiliary, times, up, down, params):
        if not hasattr(self, 'distance'):
            self.distance = road_distances(up, params)
        return motion_features(history, auxiliary, times, up, down, params, self.distance)
