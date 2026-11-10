"""Training-only local-linear-trend queue labels and causal forecast features.

The monthly RTS smoother may build training targets. It is not a forecast
feature. Forecast filters restart on the exact 13-row T-60..T window.
Observation variance comes from an assumption-dependent semivariogram nugget.
Process variances are training-moment estimates, clipped to positive bounds.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.special import erf
from lightgbm import LGBMClassifier, LGBMRegressor

from .features import neighbors
from .queue_events import visible_event
from .queue_v5 import localized_features
from .research_diagnostics import noise_nugget

warnings.filterwarnings('ignore', message='X does not have valid feature names')

QUEUE_RATIO = 0.6
QUALITY_MIN = 75
R_MIN, R_MAX = 1e-6, 0.05
Q_LEVEL_MIN, Q_LEVEL_MAX = 1e-8, 1e-2
Q_SLOPE_MIN, Q_SLOPE_MAX = 1e-10, 1e-3
HORIZONS = (1, 2, 3, 4, 5, 6)
ASSUMPTION = (
    'Observation variance is the larger of the linear-lag and quadratic-lag '
    'semivariogram intercepts on free-flow training pairs (normalized speed > 0.8). '
    'Those intercepts are not an identified measurement-noise floor. Process variances '
    'are second-difference moments after removing the observation contribution implied '
    'by that variance, then clipped to positive bounds. A negative moment uses the floor.'
)


class NoiseModel:
    def __init__(self, observation_variance, level_process_variance, slope_process_variance,
                 sigma_linear, sigma_quadratic, observation_source, process_source, assumption=ASSUMPTION):
        self.observation_variance = np.asarray(observation_variance, float)
        self.level_process_variance = np.asarray(level_process_variance, float)
        self.slope_process_variance = np.asarray(slope_process_variance, float)
        self.sigma_linear = np.asarray(sigma_linear, float)
        self.sigma_quadratic = np.asarray(sigma_quadratic, float)
        self.observation_source = list(observation_source)
        self.process_source = list(process_source)
        self.assumption = assumption
        self._check()

    def _check(self):
        n = len(self.observation_variance)
        arrays = (self.observation_variance, self.level_process_variance, self.slope_process_variance)
        if any(len(a) != n for a in arrays):
            raise ValueError('Noise parameters must share one link axis')
        if np.any(self.observation_variance < R_MIN) or np.any(self.observation_variance > R_MAX):
            raise ValueError('Observation variance left the positive bounds')
        if np.any(self.level_process_variance < Q_LEVEL_MIN) or np.any(self.level_process_variance > Q_LEVEL_MAX):
            raise ValueError('Level process variance left the positive bounds')
        if np.any(self.slope_process_variance < Q_SLOPE_MIN) or np.any(self.slope_process_variance > Q_SLOPE_MAX):
            raise ValueError('Slope process variance left the positive bounds')


class SmoothFit:
    def __init__(self, level, variance, probability, hard):
        self.level = level
        self.variance = variance
        self.probability = probability
        self.hard = hard


class Forecast:
    def __init__(self, level_mean, level_sd, slope_mean, slope_sd, probability):
        self.level_mean = level_mean
        self.level_sd = level_sd
        self.slope_mean = slope_mean
        self.slope_sd = slope_sd
        self.probability = probability


def _clip(value, low, high):
    return float(np.minimum(np.maximum(value, low), high))


def _as_utc(stamp):
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is None:
        return stamp.tz_localize('UTC')
    return stamp.tz_convert('UTC')


def normalize_speed(values, params, auxiliary=None):
    """Normalized speed v/v_f. Quality below 75 removes the cell."""
    free = np.asarray(params.free_speed_kmh, float)
    if np.any(~np.isfinite(free) | (free <= 0)):
        raise ValueError('Free-flow speed must be positive')
    speed = np.asarray(values[..., 0], float) / free[None]
    if auxiliary is not None:
        history, _ = mask_low_quality(np.stack([
            np.asarray(values[..., 0], float), np.asarray(values[..., 1], float)], -1), auxiliary)
        speed = history[..., 0] / free[None]
    return speed


def mask_low_quality(history, auxiliary, threshold=QUALITY_MIN):
    """Copy inputs. A released quality flag below 75 removes speed, flow and occupancy."""
    history = np.array(history, dtype=float, copy=True)
    auxiliary = np.array(auxiliary, dtype=float, copy=True)
    if auxiliary.ndim == 3 and auxiliary.shape[-1] >= 3:
        bad = np.isfinite(auxiliary[..., 2]) & (auxiliary[..., 2] < threshold)
        history[bad] = np.nan
        auxiliary[bad, 0] = np.nan
        if auxiliary.shape[-1] > 1:
            auxiliary[bad, 1] = np.nan
    return history, auxiliary


def estimate_noise(normalized, stable=None):
    """Per-link noise from a training series. The caller must not pass later months."""
    y = np.asarray(normalized, float)
    if y.ndim != 2:
        raise ValueError('Normalized speed must have shape (time, links)')
    finite = np.isfinite(y)
    if stable is None:
        stable = finite & (y > 0.8)
    else:
        stable = np.asarray(stable, bool) & finite
    n = y.shape[1]
    pooled = noise_nugget(y, stable)
    observation = np.empty(n)
    level_q = np.empty(n)
    slope_q = np.empty(n)
    sigma_l = np.empty(n)
    sigma_q = np.empty(n)
    observation_source = []
    process_source = []
    for link in range(n):
        report = noise_nugget(y[:, link], stable[:, link])
        source, variance, linear, quadratic = _observation_variance(report)
        if source == 'insufficient':
            source, variance, linear, quadratic = _observation_variance(pooled)
            source = 'pooled_' + source
        observation[link] = variance
        sigma_l[link] = linear
        sigma_q[link] = quadratic
        observation_source.append(source)
        q_level, q_slope, how = _process_variance(y[:, link], observation[link])
        level_q[link] = q_level
        slope_q[link] = q_slope
        process_source.append(how)
    failed = [i for i, how in enumerate(process_source) if how == 'insufficient']
    if failed:
        donors = [i for i in range(n) if i not in failed]
        if donors:
            fill_level = float(np.median(level_q[donors]))
            fill_slope = float(np.median(slope_q[donors]))
            how = 'pooled_link_median'
        else:
            fill_level, fill_slope, how = Q_LEVEL_MIN, Q_SLOPE_MIN, 'floor'
        for i in failed:
            level_q[i] = fill_level
            slope_q[i] = fill_slope
            process_source[i] = how
    return NoiseModel(observation, level_q, slope_q, sigma_l, sigma_q, observation_source, process_source)


def _observation_variance(report):
    if report.get('sigma_linear') is None and report.get('sigma_quadratic') is None:
        return 'insufficient', R_MIN, np.nan, np.nan
    intercepts = [v for v in report.get('unclipped_variance_intercepts', []) if v is not None and np.isfinite(v)]
    positive = [v for v in intercepts if v > 0]
    linear = report.get('sigma_linear')
    quadratic = report.get('sigma_quadratic')
    linear = np.nan if linear is None else float(linear)
    quadratic = np.nan if quadratic is None else float(quadratic)
    if not positive:
        return 'floor', R_MIN, linear, quadratic
    raw = max(positive)
    clipped = _clip(raw, R_MIN, R_MAX)
    source = 'max_linear_quadratic_intercept'
    if clipped != raw:
        source += '_clipped'
    return source, clipped, linear, quadratic


def _process_variance(series, observation_variance):
    """Stationary second-difference moments of a local linear trend plus white noise.

    Var(delta2) = q_slope + 2 q_level + 6 R
    Cov(delta2_t, delta2_{t-1}) = -q_level - 4 R
    """
    y = np.asarray(series, float)
    delta = y[2:] - 2 * y[1:-1] + y[:-2]
    ok = np.isfinite(y[2:]) & np.isfinite(y[1:-1]) & np.isfinite(y[:-2])
    if int(ok.sum()) < 40:
        return Q_LEVEL_MIN, Q_SLOPE_MIN, 'insufficient'
    gamma0 = float(np.mean(delta[ok] ** 2))
    both = ok[1:] & ok[:-1]
    if int(both.sum()) < 40:
        return Q_LEVEL_MIN, Q_SLOPE_MIN, 'insufficient'
    gamma1 = float(np.mean(delta[1:][both] * delta[:-1][both]))
    q_level = _clip(-gamma1 - 4 * observation_variance, Q_LEVEL_MIN, Q_LEVEL_MAX)
    q_slope = _clip(gamma0 - 2 * q_level - 6 * observation_variance, Q_SLOPE_MIN, Q_SLOPE_MAX)
    return q_level, q_slope, 'second_difference'


def predict_step(mean, cov, q_level, q_slope):
    """One local-linear-trend transition. mean is (batch, 2), cov is (batch, 2, 2)."""
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    q_level = np.asarray(q_level, float)
    q_slope = np.asarray(q_slope, float)
    level = mean[:, 0] + mean[:, 1]
    out = np.stack([level, mean[:, 1]], axis=1)
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    nxt = np.empty_like(cov)
    nxt[:, 0, 0] = a + 2 * b + c + q_level
    nxt[:, 0, 1] = nxt[:, 1, 0] = b + c
    nxt[:, 1, 1] = c + q_slope
    return out, nxt


def update_step(mean, cov, observation, observation_variance):
    """Joseph-stabilized scalar update of the level. Missing rows are not passed here."""
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    observation = np.asarray(observation, float)
    observation_variance = np.asarray(observation_variance, float)
    innovation = observation - mean[:, 0]
    scale = cov[:, 0, 0] + observation_variance
    k0 = cov[:, 0, 0] / scale
    k1 = cov[:, 1, 0] / scale
    updated = np.stack([mean[:, 0] + k0 * innovation, mean[:, 1] + k1 * innovation], axis=1)
    batch = len(observation)
    gain_map = np.zeros((batch, 2, 2))
    gain_map[:, 0, 0] = 1 - k0
    gain_map[:, 1, 0] = -k1
    gain_map[:, 1, 1] = 1
    cov_u = (gain_map @ cov) @ np.transpose(gain_map, (0, 2, 1))
    cov_u[:, 0, 0] += observation_variance * k0 * k0
    cov_u[:, 1, 1] += observation_variance * k1 * k1
    cov_u[:, 0, 1] += observation_variance * k0 * k1
    cov_u[:, 1, 0] = cov_u[:, 0, 1]
    return updated, _project_psd(cov_u)


def _project_psd(cov, floor=1e-12):
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    trace = cov[:, 0, 0] + cov[:, 1, 1]
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    root = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))
    smallest = 0.5 * (trace - root)
    shift = np.maximum(floor - smallest, 0)
    cov = cov.copy()
    cov[:, 0, 0] += shift
    cov[:, 1, 1] += shift
    return cov


def _filter_batch(series, observation_variance, q_level, q_slope):
    series = np.asarray(series, float)
    steps, batch = series.shape
    means = np.empty((steps, batch, 2))
    covs = np.empty((steps, batch, 2, 2))
    mean = np.zeros((batch, 2))
    mean[:, 0] = 1
    cov = np.zeros((batch, 2, 2))
    cov[:, 0, 0] = 1
    cov[:, 1, 1] = 0.25
    finite = np.isfinite(series)
    for t in range(steps):
        mean, cov = predict_step(mean, cov, q_level, q_slope)
        seen = finite[t]
        if seen.any():
            updated_mean, updated_cov = update_step(mean[seen], cov[seen], series[t, seen], observation_variance[seen])
            mean = mean.copy()
            cov = cov.copy()
            mean[seen] = updated_mean
            cov[seen] = updated_cov
        means[t] = mean
        covs[t] = cov
    return means, covs


def _smooth_batch(means, covs, q_level, q_slope):
    steps, batch, _ = means.shape
    level = np.empty((steps, batch))
    variance = np.empty((steps, batch))
    slope = np.empty((steps, batch))
    cov_path = np.empty_like(covs)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    mean_s = means[-1].copy()
    cov_s = covs[-1].copy()
    level[-1] = mean_s[:, 0]
    variance[-1] = cov_s[:, 0, 0]
    slope[-1] = mean_s[:, 1]
    cov_path[-1] = cov_s
    for t in range(steps - 2, -1, -1):
        mean_p, cov_p = predict_step(means[t], covs[t], q_level, q_slope)
        cov_p = cov_p + np.eye(2) * 1e-12
        cross = np.einsum('ij,bjk->bik', transition, covs[t])
        gain = np.transpose(np.linalg.solve(cov_p, cross), (0, 2, 1))
        mean_s = means[t] + np.einsum('bij,bj->bi', gain, mean_s - mean_p)
        gap = cov_s - cov_p
        cov_s = covs[t] + (gain @ gap) @ np.transpose(gain, (0, 2, 1))
        cov_s = _project_psd(cov_s)
        level[t] = mean_s[:, 0]
        variance[t] = cov_s[:, 0, 0]
        slope[t] = mean_s[:, 1]
        cov_path[t] = cov_s
    return level, variance, slope, cov_path


def _link_batches(steps, links):
    width = max(1, min(links, int(2_000_000 / max(steps, 1))))
    for start in range(0, links, width):
        yield start, min(links, start + width)


def kalman_filter(normalized, noise):
    """Causal filter. State at t does not depend on observations after t."""
    series = np.asarray(normalized, float)
    _check_noise_width(series.shape[1], noise)
    steps, links = series.shape
    means = np.empty((steps, links, 2))
    covs = np.empty((steps, links, 2, 2))
    for start, stop in _link_batches(steps, links):
        means[:, start:stop], covs[:, start:stop] = _filter_batch(
            series[:, start:stop], noise.observation_variance[start:stop],
            noise.level_process_variance[start:stop], noise.slope_process_variance[start:stop])
    return means, covs


def smooth_series(normalized, noise):
    """RTS smoother. Future observations inside this array move earlier states.

    Do not pass a series that crosses a fit/evaluation boundary. Forecast code
    does not call this function.
    """
    series = np.asarray(normalized, float)
    _check_noise_width(series.shape[1], noise)
    steps, links = series.shape
    level = np.empty((steps, links))
    variance = np.empty((steps, links))
    for start, stop in _link_batches(steps, links):
        means, covs = _filter_batch(
            series[:, start:stop], noise.observation_variance[start:stop],
            noise.level_process_variance[start:stop], noise.slope_process_variance[start:stop])
        level[:, start:stop], variance[:, start:stop], _, _ = _smooth_batch(
            means, covs, noise.level_process_variance[start:stop], noise.slope_process_variance[start:stop])
    sd = np.sqrt(np.maximum(variance, 1e-12))
    probability = _queue_probability(level, sd)
    hard = level <= QUEUE_RATIO
    return SmoothFit(level, variance, probability, hard)


def smooth_training_series(normalized, noise, times, boundary):
    """Smoother for training targets. Rejects a series that reaches the boundary."""
    times = pd.DatetimeIndex(times)
    if len(times) != len(normalized):
        raise ValueError('Smoother times must match the series')
    if times.max() >= _as_utc(boundary):
        raise ValueError('Training smoother includes the evaluation boundary or a later time')
    return smooth_series(normalized, noise)


def _check_noise_width(links, noise):
    if len(noise.observation_variance) != links:
        raise ValueError('Noise parameters do not match the link axis')


def _queue_probability(level, sd):
    z = (QUEUE_RATIO - np.asarray(level, float)) / np.maximum(np.asarray(sd, float), 1e-12)
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def propagate_trend(mean, cov, q_level, q_slope, horizons=HORIZONS):
    """Full h-step local-linear-trend covariance.

    F^h = [[1, h], [0, 1]]. Process noise is sum_{j=0}^{h-1} F^j Q (F^j)^T,
    which is quadratic in h when slope noise is present. Adding h*q_level to the
    current level variance drops both the h^2 slope uncertainty and that sum.
    """
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    single = mean.ndim == 1
    if single:
        mean = mean[None]
        cov = np.asarray(cov, float)[None]
        q_level = np.atleast_1d(np.asarray(q_level, float))
        q_slope = np.atleast_1d(np.asarray(q_slope, float))
    else:
        q_level = np.asarray(q_level, float)
        q_slope = np.asarray(q_slope, float)
    count = len(horizons)
    batch = mean.shape[0]
    level = np.empty((count, batch))
    level_var = np.empty((count, batch))
    slope = np.empty((count, batch))
    slope_var = np.empty((count, batch))
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    for i, horizon in enumerate(horizons):
        h = float(horizon)
        level[i] = mean[:, 0] + h * mean[:, 1]
        slope[i] = mean[:, 1]
        accumulated = h * q_level + q_slope * (h - 1) * h * (2 * h - 1) / 6
        level_var[i] = a + 2 * h * b + (h ** 2) * c + accumulated
        slope_var[i] = c + h * q_slope
    if single:
        return level[:, 0], level_var[:, 0], slope[:, 0], slope_var[:, 0]
    return level, level_var, slope, slope_var


def forecast_window(normalized, noise):
    """Restart a filter on one T-60..T window and propagate six horizons."""
    series = np.asarray(normalized, float)
    if series.ndim != 2 or len(series) != 13:
        raise ValueError('Forecast filter accepts only the exact 13-row T-60..T window and restarts inside it')
    means, covs = kalman_filter(series, noise)
    level, level_var, slope, slope_var = propagate_trend(
        means[-1], covs[-1], noise.level_process_variance, noise.slope_process_variance)
    sd = np.sqrt(np.maximum(level_var, 1e-12))
    slope_sd = np.sqrt(np.maximum(slope_var, 1e-12))
    return Forecast(level, sd, slope, slope_sd, _queue_probability(level, sd))


def linear_shortcut_variance(cov, q_level, horizons=HORIZONS):
    """The dimensionally incomplete h*q_level accumulation, kept for tests."""
    base = np.asarray(cov, float)[..., 0, 0]
    q_level = np.asarray(q_level, float)
    rows = [base + float(h) * q_level for h in horizons]
    return np.stack(rows, axis=0)


def directed_neighbor_signals(level, sd, slope, probability, up, down):
    """Directed operator signals. Link-index order is not a neighbor relation."""
    packed = np.stack([level, sd, slope, probability], axis=-1).astype(np.float32)
    return neighbors(packed, up), neighbors(packed, down)


def queue_v6_features(history, auxiliary, times, up, down, params, noise):
    """Existing causal event features plus this window's restarted trend forecast."""
    history, auxiliary = mask_low_quality(history, auxiliary)
    base = localized_features(history, auxiliary, times, up, down, params)
    free = np.asarray(params.free_speed_kmh, float)
    forecast = forecast_window(history[..., 0] / free[None], noise)
    up_signal, down_signal = directed_neighbor_signals(
        forecast.level_mean, forecast.level_sd, forecast.slope_mean, forecast.probability, up, down)
    local = np.stack([
        forecast.level_mean, forecast.level_sd, forecast.slope_mean, forecast.slope_sd, forecast.probability
    ], axis=-1)
    extra = np.concatenate([local, up_signal, down_signal], axis=-1)
    extra = extra.reshape(extra.shape[0] * extra.shape[1], extra.shape[2])
    if len(extra) != len(base):
        raise ValueError('Forecast feature rows do not match the causal event layout')
    return np.concatenate([base, extra.astype(np.float32)], axis=1)


def assert_events_before(events, times, boundary):
    """Every horizon label used for training must end strictly before evaluation."""
    boundary = _as_utc(boundary)
    times = pd.DatetimeIndex(times)
    for event in events:
        if event.origin + 6 >= len(times):
            raise ValueError('Event horizon is outside the training time index')
        if times[event.origin + 6] >= boundary:
            raise ValueError('Horizon target crosses the evaluation boundary')
        if event.times[-1] != times[event.origin]:
            raise ValueError('Event origin is not aligned with the training index')


def horizon_targets(event, probability, hard):
    """Slice T+5..T+30 labels. probability/hard are training-block smoother outputs."""
    start, stop = event.origin + 1, event.origin + 7
    if stop > len(probability):
        raise ValueError('Smoother output does not cover the event horizon')
    soft = np.asarray(probability[start:stop], float)
    discrete = np.asarray(hard[start:stop], bool).astype(float)
    if soft.shape != event.truth.shape or discrete.shape != event.truth.shape:
        raise ValueError('Smoothed targets must have the event horizon shape')
    return {'smoothed_soft': soft, 'smoothed_hard': discrete, 'raw': event.truth.astype(float)}


def origin_counts(queue, finite, active):
    """Count spaced candidate origins under a boolean queue field.

    This is a label-definition count, not the sampled evaluation set. `queue`
    is True where that definition calls a link-time queued. History completeness
    still uses finite speed observations, so the gate is comparable across labels.
    """
    queue = np.asarray(queue, bool)
    finite = np.asarray(finite, bool)
    active = np.asarray(active, bool)
    steps, _ = queue.shape
    any_queue = queue.any(axis=1)
    active_sum = np.r_[0, np.cumsum(active)]
    origins = np.arange(12, steps - 6)
    origins = origins[active_sum[origins + 7] - active_sum[origins - 12] == 19]
    chosen = {'queue_onset': [], 'queue_ongoing': []}
    taken = []
    for origin in origins:
        if finite[origin - 12:origin].mean() < .70:
            continue
        if not any_queue[origin + 1:origin + 7].any():
            continue
        if any(abs(int(origin) - old) < 72 for old in taken):
            continue
        taken.append(int(origin))
        condition = 'queue_ongoing' if any_queue[origin - 12:origin].any() else 'queue_onset'
        chosen[condition].append(int(origin))
    return {
        'queue_onset': len(chosen['queue_onset']),
        'queue_ongoing': len(chosen['queue_ongoing']),
        'total': len(taken),
        'onset_origins': chosen['queue_onset'],
        'ongoing_origins': chosen['queue_ongoing'],
    }


class QueueArmModel:
    """One of raw, smoothed_hard, smoothed_soft. Sampling follows the raw proxy mask."""

    def __init__(self, arm='raw', policy='origin_observed', rounds=40, rows_per_window=200,
                 min_child_samples=20, seed=606):
        if arm not in ('raw', 'smoothed_hard', 'smoothed_soft'):
            raise ValueError('Unknown queue arm')
        self.arm = arm
        self.policy = policy
        self.rounds = rounds
        self.rows_per_window = rows_per_window
        self.min_child_samples = min_child_samples
        self.seed = seed
        self.models = {}
        self.training_rows = {}

    def fit(self, events, up, down, params, noise, targets):
        self.noise = noise
        self.models = {}
        self.training_rows = {}
        grouped = {condition: [] for condition in ('queue_onset', 'queue_ongoing')}
        for event in events:
            grouped[event.condition].append(event)
        for condition, chosen in grouped.items():
            if not chosen:
                continue
            self.models[condition] = self._fit_condition(chosen, up, down, params, targets, condition)
        if not self.models:
            raise ValueError('Queue arm received no events')
        return self

    def _fit_condition(self, events, up, down, params, targets, condition):
        rng = np.random.default_rng(self.seed)
        xx, yy, weights = [], [], []
        links = len(params)
        for event in sorted(events, key=lambda item: item.origin):
            key = (event.origin, event.condition)
            if key not in targets:
                raise ValueError('Missing training target for an event')
            target = np.asarray(targets[key], float)
            if target.shape != event.truth.shape:
                raise ValueError('Training target shape drifted from the raw event')
            history, auxiliary = visible_event(event, self.policy)
            features = queue_v6_features(history, auxiliary, event.times, up, down, params, self.noise)
            valid = event.valid.ravel().copy()
            raw = event.truth.ravel()
            observed_target = np.isfinite(target.ravel())
            if condition == 'queue_onset':
                valid[:-links] = False
            usable = valid & observed_target
            selected, sample_weight = [], []
            for label in (True, False):
                available = np.flatnonzero(usable & (raw == label))
                count = min(len(available), self.rows_per_window // 2)
                if count:
                    picked = rng.choice(available, count, replace=False)
                    selected.extend(picked)
                    sample_weight.extend([len(available) / count / max(usable.sum(), 1)] * count)
            if selected:
                xx.append(features[selected])
                yy.append(target.ravel()[selected])
                weights.extend(sample_weight)
        if not xx:
            raise ValueError(f'No usable rows for {self.arm}/{condition}')
        x = np.concatenate(xx)
        y = np.concatenate(yy)
        weight = np.asarray(weights, float)
        weight *= len(weight) / weight.sum()
        self.training_rows[condition] = int(len(y))
        if self.arm == 'smoothed_soft':
            y = np.clip(y, 0, 1)
            if np.allclose(y, y[0]):
                return float(y[0])
            model = LGBMRegressor(objective='cross_entropy', n_estimators=self.rounds, learning_rate=.05,
                num_leaves=15, min_child_samples=self.min_child_samples, reg_lambda=6,
                colsample_bytree=.9, n_jobs=1, verbosity=-1, random_state=self.seed)
            return model.fit(x, y, sample_weight=weight)
        y_int = y.astype(int)
        if len(np.unique(y_int)) < 2:
            return float(y_int[0])
        model = LGBMClassifier(n_estimators=self.rounds, learning_rate=.05, num_leaves=15,
            min_child_samples=self.min_child_samples, reg_lambda=6, colsample_bytree=.9,
            n_jobs=1, verbosity=-1, random_state=self.seed)
        return model.fit(x, y_int, sample_weight=weight)

    def scores(self, history, auxiliary, times, condition, up, down, params):
        if condition not in self.models:
            raise ValueError(f'{self.arm} has no model for {condition}')
        features = queue_v6_features(history, auxiliary, times, up, down, params, self.noise)
        model = self.models[condition]
        if isinstance(model, float):
            probability = np.full(len(features), model, dtype=float)
        else:
            probability = np.asarray(model.booster_.predict(features, num_threads=1), float)
        probability = np.clip(probability, 0, 1).reshape(6, -1)
        if condition == 'queue_onset':
            probability[:-1] = 0
        return probability

    def probabilities(self, event, up, down, params):
        history, auxiliary = visible_event(event, self.policy)
        return self.scores(history, auxiliary, event.times, event.condition, up, down, params)
