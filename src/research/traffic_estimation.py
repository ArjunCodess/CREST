"""Offline state reconstruction and causal queue predictions share observed inputs."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .observations import Window
from .sparse_traffic import Constraints, Estimate, Interval, conditional_interval, conservation_system, robust_fit
from .traffic_simulator import Simulation


@dataclass(frozen=True)
class Reconstruction:
    values: np.ndarray
    status: str
    method: str


def historical_level(training: list[Window], channel: str) -> float:
    values = np.concatenate([window.inputs(channel).ravel() for window in training])
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"Training has no measured {channel} values")
    return float(np.median(finite))


def persistence(values: np.ndarray, level: float) -> np.ndarray:
    result = np.empty_like(values)
    previous = np.full(values.shape[1], level)
    for i, row in enumerate(values):
        previous = np.where(np.isfinite(row), row, previous)
        result[i] = previous
    return result


def graph_features(values: np.ndarray, level: float, *, offline: bool) -> np.ndarray:
    steps, cells = values.shape
    padded = np.pad(values, ((2, int(offline)), (1, 1)), constant_values=np.nan)
    spatial = [padded[2:2 + steps, i:i + cells] for i in (0, 1, 2)]
    lagged = [padded[i:i + steps, 1:1 + cells] for i in (0, 1)]
    if offline:
        lagged.append(padded[3:3 + steps, 1:1 + cells])
    features = [*spatial, *lagged, persistence(values, level)]
    # Missing values can occupy an entire feature in outage-only training.
    return np.nan_to_num(np.stack(features, axis=-1).reshape(steps * cells, -1), nan=-1.)


def train_tree(training: list[Window], channel: str, level: float, *, offline: bool) -> HistGradientBoostingRegressor:
    datasets, labels = [], []
    for window in training:
        observed = window.inputs(channel)
        for mask in outage_masks(observed.shape):
            eligible = mask & np.isfinite(observed)
            predictors = graph_features(np.where(mask, np.nan, observed), level, offline=offline)
            datasets.append(predictors[eligible.ravel()])
            labels.append(observed[eligible])
    if not sum(map(len, labels)):
        raise ValueError(f"No eligible training targets for {channel}")
    model = HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=15, min_samples_leaf=8, random_state=17)
    model.fit(np.concatenate(datasets), np.concatenate(labels))
    return model


def tree_predict(model: HistGradientBoostingRegressor, values: np.ndarray, level: float, *, offline: bool) -> np.ndarray:
    return np.where(np.isfinite(values), values, model.predict(graph_features(values, level, offline=offline)).reshape(values.shape))


def outage_masks(shape: tuple[int, int], duration: int | None = None) -> list[np.ndarray]:
    steps, cells = shape
    width = max(1, min(steps - 1, duration or steps // 4))
    masks = []
    for phase in (0, 1, 2):
        mask = np.zeros(shape, dtype=bool)
        start = min(steps - width, max(1, (phase + 1) * steps // 5))
        link = phase * (cells - 1) // 2
        mask[start:start + width, link:link + max(1, cells // 3)] = True
        masks.append(mask)
    return masks


def graph_reconstruct(
    values: np.ndarray, prior: np.ndarray, graph: sparse.spmatrix, scale: float,
    *, graph_weight: float = 0.35, time_weight: float = 0.6,
) -> Reconstruction:
    """Robust offline graph smoother for channels without observed storage."""
    steps, cells = values.shape
    n = values.size
    edges = graph.tocoo()
    edge_rows = np.arange(len(edges.row))
    difference = sparse.csc_matrix(
        (np.r_[-np.ones(len(edges.row)), np.ones(len(edges.row))],
         (np.r_[edge_rows, edge_rows], np.r_[edges.col, edges.row])), shape=(len(edge_rows), cells),
    )
    time = sparse.diags((-np.ones(steps - 1), np.ones(steps - 1)), (0, 1), shape=(steps - 1, steps))
    smooth = sparse.vstack((graph_weight * sparse.kron(sparse.eye(steps), difference),
                            time_weight * sparse.kron(time, sparse.eye(cells))), format="csc")
    system = Constraints(sparse.csc_matrix((0, n)), np.empty(0), np.zeros(n), np.full(n, np.inf))
    observed = np.flatnonzero(np.isfinite(values))
    estimate = robust_fit(system, prior.ravel() / scale, observed, values.ravel()[observed] / scale,
                          prior_weight=0.55, huber_delta=0.7, smoothness=smooth)
    if estimate.values is None:
        return Reconstruction(prior.copy(), estimate.status, "fallback")
    return Reconstruction(estimate.values.reshape(values.shape) * scale, estimate.status, "graph_huber")


def physics_reconstruct(
    simulation: Simulation, storage: np.ndarray, boundary_prior: np.ndarray,
    *, storage_level: float, tolerance: float = 4., timed_limit: float = 10.,
) -> tuple[Estimate, Interval, np.ndarray]:
    steps, channels = simulation.flux.shape
    dt = simulation.window.cadence_ms / 3_600_000
    upper = np.broadcast_to(np.max(simulation.flow_upper, axis=0), (steps, channels))
    return fit_storage(simulation.incidence, simulation.storage[0], simulation.capacity,
                       np.zeros_like(upper), upper, storage, boundary_prior, dt,
                       simulation.boundary_channels[0], storage_level=storage_level,
                       tolerance=tolerance, time_limit=timed_limit)


def fit_storage(
    incidence: np.ndarray, initial: np.ndarray, capacity: np.ndarray,
    flux_lower: np.ndarray, flux_upper: np.ndarray, storage: np.ndarray, flux_prior: np.ndarray,
    dt: float, inlet_channel: int, *, storage_level: float, tolerance: float = 4., time_limit: float = 10.,
) -> tuple[Estimate, Interval, np.ndarray]:
    steps, channels = flux_lower.shape
    cells = incidence.shape[0]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Storage reconstruction requires a positive interval length")
    # Solve in vehicles per interval so storage and flux variables have comparable scales.
    system = conservation_system(incidence, initial, capacity, flux_lower * dt, flux_upper * dt, 1.)
    n_storage = (steps + 1) * cells
    baseline = persistence(storage, storage_level)
    prior_storage = np.vstack((baseline, baseline[-1]))
    prior = np.r_[prior_storage.ravel(), np.broadcast_to(flux_prior, (steps, channels)).ravel() * dt]
    indices = np.flatnonzero(np.isfinite(storage))
    observed = storage.ravel()[indices]
    weights = np.r_[np.full(n_storage, 0.45), np.full(steps * channels, 1e-5 / dt**2)]
    fit = robust_fit(system, prior, indices, observed, prior_weight=weights, huber_delta=5, time_limit=time_limit)
    objective = np.zeros_like(prior)
    objective[n_storage + np.arange(steps) * channels + inlet_channel] = 1.
    interval = conditional_interval(system, objective, indices, observed, tolerance, time_limit=time_limit)
    if fit.values is not None:
        fit = replace(fit, values=np.r_[fit.values[:n_storage], fit.values[n_storage:] / dt])
    return fit, interval, baseline


def congested(speed: np.ndarray, threshold: float) -> np.ndarray:
    slow = np.isfinite(speed) & (speed < threshold)
    adjacent = np.zeros_like(slow)
    adjacent[..., 1:] |= slow[..., :-1]
    adjacent[..., :-1] |= slow[..., 1:]
    return slow & adjacent


def queue_features(values: np.ndarray, level: float) -> np.ndarray:
    last = values[-1]
    filled = persistence(values, level)
    state = filled[-1]
    lag = filled[max(0, len(values) - 3)]
    upstream = np.r_[state[0], state[:-1]]
    downstream = np.r_[state[1:], state[-1]]
    following = np.logical_or.accumulate(np.isfinite(values[::-1]), axis=0)
    age = (~following).sum(axis=0)
    return np.column_stack((state, state - lag, upstream, downstream, np.isfinite(last), age))


@dataclass(frozen=True)
class QueuePrediction:
    origin_ms: int
    horizon: int
    probability: np.ndarray
    onset_probability: np.ndarray
    available_cells: np.ndarray
    current_congestion: np.ndarray


class QueueHead:
    def __init__(self, training: list[Window], *, horizon: int = 3, threshold: float = 50.,
                 processing_lag_ms: int = 0) -> None:
        self.level = historical_level(training, "speed")
        self.horizon = horizon
        self.threshold = threshold
        features, targets, onset_targets = [], [], []
        for window in training:
            speed = window.inputs("speed")
            truth = congested(speed, threshold)
            hidden = outage_masks(speed.shape)[1]
            for i in range(2, len(speed) - horizon):
                origin = int(window.time_ms[i] + window.cadence_ms + processing_lag_ms)
                prefix = window.inputs("speed", hidden=hidden, as_of_ms=origin)[max(0, i - 5):i + 1]
                if not np.isfinite(prefix).any():
                    continue
                available = np.isfinite(speed[i + horizon])
                features.append(queue_features(prefix, self.level)[available])
                targets.append(truth[i + horizon, available])
                onset_targets.append((truth[i + horizon] & ~truth[i])[available])
        self.membership = self._fit(features, targets)
        self.onset = self._fit(features, onset_targets)

    @staticmethod
    def _fit(features: list[np.ndarray], labels: list[np.ndarray]) -> LogisticRegression | float:
        if not labels or not sum(map(len, labels)):
            raise ValueError("Queue training has no eligible examples")
        y = np.concatenate(labels)
        if len(np.unique(y)) == 1:
            return float(y[0])
        model = LogisticRegression(max_iter=300)
        return model.fit(np.concatenate(features), y)

    def predict(self, window: Window, index: int, hidden: np.ndarray | None = None, *,
                processing_lag_ms: int = 0) -> QueuePrediction:
        origin = int(window.time_ms[index] + window.cadence_ms + processing_lag_ms)
        prefix = window.inputs("speed", hidden=hidden, as_of_ms=origin)[max(0, index - 5):index + 1]
        feature = queue_features(prefix, self.level)
        available = np.isfinite(prefix[-2:]).any(axis=0)
        probabilities = [model.predict_proba(feature)[:, 1] if isinstance(model, LogisticRegression)
                         else np.full(len(feature), model) for model in (self.membership, self.onset)]
        return QueuePrediction(origin, self.horizon, *probabilities, available, congested(prefix[-1], self.threshold))
