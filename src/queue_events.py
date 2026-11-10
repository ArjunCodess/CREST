"""Condition-aware queue experiments using causal, train-only event proxies.

Official queue labels are unavailable. These windows label observed training
speeds and must never be presented as official latent-state validation.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from .features import features, neighbors
from .evaluation import proxy_queue_iou

CONDITIONS = ("queue_onset", "queue_ongoing")


@dataclass
class QueueEvent:
    origin: int
    condition: str
    history: np.ndarray
    auxiliary: np.ndarray
    times: pd.DatetimeIndex
    truth: np.ndarray
    valid: np.ndarray
    first_queue_step: int


def load_auxiliary(root, panel, times, links):
    """Read only train occupancy/density/quality, aligned to the state reader."""
    result = np.full((len(times), len(links), 3), np.nan, dtype=np.float32)
    folder = Path(root) / "corridors" / panel / "train/mainline_states"
    columns = ["timestamp", "link_id", "occupancy", "density_occ_linear_vehpkm", "pct_observed"]
    import pyarrow.parquet as pq
    for path in sorted(folder.rglob("*.parquet")):
        schema = pq.ParquetFile(path).schema_arrow.names
        raw = pd.read_parquet(path, columns=[c for c in columns if c in schema])
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        raw = raw[raw.timestamp.between(times[0], times[-1])].copy()
        if raw.empty:
            continue
        raw["link_id"] = raw.link_id.astype(str)
        eligible = raw.pct_observed >= 75
        for channel in ("occupancy", "density_occ_linear_vehpkm"):
            if channel in raw:
                raw.loc[~eligible, channel] = np.nan
        stamps = pd.DatetimeIndex(sorted(raw.timestamp.unique()))
        for c, name in enumerate(columns[2:]):
            if name in raw:
                frame = raw.pivot_table(index="timestamp", columns="link_id", values=name)
                result[times.get_indexer(stamps), :, c] = frame.reindex(index=stamps, columns=links).to_numpy()
    return result


def collect_events(values, auxiliary, times, active, cutoff, rng,
                   max_per_condition=96, spacing_steps=72, sampler="random"):
    """Sample origins in random order, then enforce spacing within a condition.

    Labels use T+5..T+30 and the condition uses T-60..T-5. All 19 grid
    positions must belong to the requested block, including the input context.
    Balanced sampling is a separate robustness check, not an official sampler.
    """
    if sampler not in ("random", "balanced"):
        raise ValueError("sampler must be random or balanced")
    active = np.asarray(active, dtype=bool)
    if values.shape[:2] != auxiliary.shape[:2] or len(active) != len(times):
        raise ValueError("Event inputs must share a time/link grid")
    n = len(times)
    if n < 19:
        return []
    finite = np.isfinite(values[..., 0])
    queue = finite & (values[..., 0] <= np.asarray(cutoff)[None])
    any_queue = queue.any(axis=1)
    candidates = {c: [] for c in CONDITIONS}
    origins = np.arange(12, n - 6)
    active_sum = np.r_[0, np.cumsum(active)]
    origins = origins[active_sum[origins + 7] - active_sum[origins - 12] == 19]
    queue_sum = np.r_[0, np.cumsum(any_queue)]
    future_has_queue = queue_sum[origins + 7] - queue_sum[origins + 1] > 0
    for t in origins[future_has_queue]:
        if finite[t-12:t].mean() < .70:
            continue
        condition = "queue_ongoing" if any_queue[t-12:t].any() else "queue_onset"
        first = int(np.flatnonzero(any_queue[t+1:t+7])[0] + 1)
        candidates[condition].append((int(t), first))
    result = []
    for condition, choices in candidates.items():
        if not choices:
            continue
        order = rng.permutation(len(choices))
        if sampler == "balanced":
            buckets = [list(order[[choices[i][1] == step for i in order]]) for step in range(1, 7)]
            order = [bucket[j] for j in range(max(map(len, buckets))) for bucket in buckets if j < len(bucket)]
        selected = []
        for i in order:
            t, first = choices[i]
            if any(abs(t - old) < spacing_steps for old in selected):
                continue
            selected.append(t)
            result.append(QueueEvent(t, condition, values[t-12:t+1].copy(),
                auxiliary[t-12:t+1].copy(), pd.DatetimeIndex(times[t-12:t+1]),
                queue[t+1:t+7].copy(), finite[t+1:t+7].copy(), first))
            if len(selected) >= max_per_condition:
                break
    return sorted(result, key=lambda w: w.origin)


def visible_event(event, policy, seed=2026):
    """Match origin availability for both state and auxiliary features."""
    if policy not in ("history_only", "origin_observed"):
        raise ValueError("Unknown origin availability policy")
    history = event.history.copy()
    auxiliary = event.auxiliary.copy()
    if policy == "history_only":
        hidden = np.ones(history.shape[1], dtype=bool)
    else:
        rng = np.random.default_rng(seed + event.origin)
        hidden = rng.random(history.shape[1]) < rng.choice([.2, .3, .5])
    history[-1, hidden] = np.nan
    if policy == "history_only":
        auxiliary[-1] = np.nan
    else:
        # Task 1 masks measurements at T, but quality flags remain released.
        auxiliary[-1, hidden, :2] = np.nan
    return history, auxiliary


def event_features(history, auxiliary, times, up, down, params):
    """Only a 13-step history is accepted; no future values can enter features."""
    if len(history) != 13 or len(auxiliary) != 13 or len(times) != 13:
        raise ValueError("Queue input must be the 13-row grid T-60..T")
    vf = params.free_speed_kmh.to_numpy()
    cap = params.capacity_vph.to_numpy()
    normalized = history / np.stack([vf, cap], axis=-1)[None]
    blocks = [features(history, times, up, down, params, False)[-1]]
    for width in (3, 6, 12):
        x = normalized[-width:]
        ok = np.isfinite(x)
        mean = np.divide(np.nansum(x, axis=0), ok.sum(axis=0),
            out=np.full(x.shape[1:], np.nan), where=ok.sum(axis=0) > 0)
        latest = np.full(x.shape[1:], np.nan)
        age = np.full(x.shape[1:], width, dtype=float)
        for lag in range(len(x)):
            take = ~np.isfinite(latest) & np.isfinite(x[-1-lag])
            latest[take] = x[-1-lag][take]
            age[take] = lag
        blocks.extend([mean, latest, latest - mean, age / 12])
    aux = auxiliary / np.array([1., 100., 100.])
    for lag in (0, 1, 3, 6, 12):
        blocks.extend([aux[-1-lag], np.isfinite(aux[-1-lag]).astype(float)])
    recent = normalized[-6:]
    valid = np.isfinite(recent)
    mean = np.divide(np.nansum(recent, axis=(0, 1)), valid.sum(axis=(0, 1)),
        out=np.full(2, np.nan), where=valid.sum(axis=(0, 1)) > 0)
    global_context = np.r_[mean, np.mean(valid),
        np.mean(np.isfinite(recent[..., 0]) & (recent[..., 0] <= .75))]
    blocks.append(np.broadcast_to(global_context, (history.shape[1], len(global_context))))
    # Two-hop spatial evidence complements the existing immediate-neighbor lags.
    for operator in (up, down):
        first = neighbors(normalized[-3:], operator)
        second = neighbors(first, operator)
        blocks.append(second[-1])
    base = np.concatenate(blocks, axis=1).astype(np.float32)
    return np.concatenate([np.tile(base, (6, 1)),
        np.repeat(np.arange(1, 7, dtype=np.float32) / 6, len(base))[:, None]], axis=1)


def decode_queue(probabilities, policy):
    """Choose a nonempty region by calibrated threshold or an IoU surrogate."""
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p.ravel(), kind="stable")
    if policy == "expected_iou":
        cumulative = np.cumsum(p.ravel()[order])
        size = np.arange(1, p.size + 1)
        # Ratio of expected intersection/union; not exact expected IoU.
        utility = cumulative / np.maximum(size + p.sum() - cumulative, 1e-9)
        count = int(np.argmax(utility) + 1)
        output = np.zeros(p.size, dtype=bool)
        output[order[:count]] = True
        return output.reshape(p.shape)
    threshold = float(policy)
    output = p >= threshold
    if not output.any():
        output.ravel()[order[0]] = True
    return output


class EventQueueModel:
    """Separate condition heads trained on matched event-window distributions."""
    def __init__(self, policy="history_only", rounds=180, rows_per_window=600):
        self.policy = policy
        self.rounds = rounds
        self.rows_per_window = rows_per_window
        self.decoders = {c: .3 for c in CONDITIONS}

    def fit(self, events, up, down, params):
        self.models = {}
        rng = np.random.default_rng(318)
        for condition in CONDITIONS:
            xx, yy, weights = [], [], []
            for event in (w for w in events if w.condition == condition):
                history, auxiliary = visible_event(event, self.policy)
                x = event_features(history, auxiliary, event.times, up, down, params)
                valid = event.valid.ravel()
                y = event.truth.ravel()
                selected, sample_weights = [], []
                for label in (True, False):
                    available = np.flatnonzero(valid & (y == label))
                    count = min(len(available), self.rows_per_window // 2)
                    if count:
                        ix = rng.choice(available, count, replace=False)
                        selected.extend(ix)
                        # Inverse sampling fraction and equal window weight.
                        sample_weights.extend([len(available) / count / valid.sum()] * count)
                if selected:
                    xx.append(x[selected]); yy.append(y[selected]); weights.extend(sample_weights)
            if not xx:
                raise ValueError(f"No training proxy events for {condition}")
            x, y = np.concatenate(xx), np.concatenate(yy)
            weight = np.asarray(weights); weight *= len(weight) / weight.sum()
            if len(np.unique(y)) == 1:
                self.models[condition] = float(y[0])
            else:
                model = LGBMClassifier(n_estimators=self.rounds, learning_rate=.04,
                    num_leaves=15, min_child_samples=60, reg_lambda=6,
                    colsample_bytree=.85, n_jobs=4, verbosity=-1, random_state=318)
                model.fit(x, y.astype(int), sample_weight=weight)
                self.models[condition] = model
        return self

    def scores(self, history, auxiliary, times, condition, up, down, params):
        x = event_features(history, auxiliary, times, up, down, params)
        model = self.models[condition]
        scores = np.full(len(x), model) if isinstance(model, float) else model.booster_.predict(x, num_threads=4)
        return scores.reshape(6, -1)

    def probabilities(self, event, up, down, params):
        history, auxiliary = visible_event(event, self.policy)
        return self.scores(history, auxiliary, event.times, event.condition, up, down, params)

    def select_decoders(self, events, up, down, params):
        for condition in CONDITIONS:
            group = [(w, self.probabilities(w, up, down, params)) for w in events if w.condition == condition]
            if not group:
                raise ValueError(f"No selection events for {condition}")
            policies = (.05, .1, .2, .3, .4, .5, .6, "expected_iou")
            scores = [np.mean([proxy_queue_iou(w.truth, decode_queue(p, policy), w.valid)
                for w, p in group]) for policy in policies]
            self.decoders[condition] = policies[int(np.argmax(scores))]
        return self

    def predict(self, event, up, down, params):
        return decode_queue(self.probabilities(event, up, down, params), self.decoders[event.condition])


def calibrate_occupancy_density(auxiliary):
    """Fit the released linear occupancy conversion on fitting rows only."""
    ratio = np.divide(auxiliary[..., 1], auxiliary[..., 0],
        out=np.full(auxiliary.shape[:2], np.nan), where=auxiliary[..., 0] > 0)
    scale = np.nanmedian(ratio, axis=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Training occupancy calibration is missing for a link")
    return scale


def history_arrays(history_rows, origin_rows, origin, links, policy, density_scale):
    """Own the release boundary: filter time/quality before constructing features.

    Monthly observations supply only the permitted row at T. All other inputs
    come from this window's released history. Returned arrays cannot contain
    a measurement after T or a monthly observation from an earlier time.
    """
    if policy not in ("history_only", "origin_observed"):
        raise ValueError("Unknown release history policy")
    h = history_rows[(history_rows.timestamp >= origin - pd.Timedelta(minutes=60)) &
                     (history_rows.timestamp < origin)].copy()
    if policy == "origin_observed":
        h = pd.concat([h, origin_rows[origin_rows.timestamp.eq(origin)]], ignore_index=True)
    if "pct_observed" in h:
        h.loc[h.pct_observed < 75, [c for c in
            ("speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm") if c in h]] = np.nan
    if "is_score_eligible" in h:
        h.loc[~h.is_score_eligible.fillna(False).astype(bool), [c for c in
            ("speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm") if c in h]] = np.nan
    times = pd.date_range(origin - pd.Timedelta(minutes=60), origin, freq="5min")
    arrays = []
    for column in ("speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm", "pct_observed"):
        if column in h:
            x = h.pivot_table(index="timestamp", columns="link_id", values=column)
            arrays.append(x.reindex(index=times, columns=links).to_numpy(dtype=np.float32))
        else:
            arrays.append(np.full((13, len(links)), np.nan, dtype=np.float32))
    auxiliary = np.stack(arrays[2:], axis=-1)
    converted = auxiliary[..., 0] * np.asarray(density_scale)[None]
    auxiliary[..., 1] = np.where(np.isfinite(auxiliary[..., 1]), auxiliary[..., 1], converted)
    return np.stack(arrays[:2], axis=-1), auxiliary, times


def predict_release_queue(root, panel, split, payload):
    """Export only official Task 2 keys using a fitted event-model payload."""
    root = Path(root)
    folder = root / "task2" / panel / split
    windows = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
    history = pd.read_parquet(folder / "window_history.parquet")
    history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
    history["link_id"] = history.link_id.astype(str)
    history["window_id"] = history.window_id.astype(str)
    template = pd.read_csv(folder / "sample_submission_queue.csv", dtype={"window_id": str, "link_id": str})
    template = template[template.window_id.isin(windows.window_id)].copy()
    model = payload["model"]
    outputs, origin_cache = [], {}
    for row in windows.itertuples(index=False):
        target = template[template.window_id.eq(row.window_id)].copy()
        target_times = pd.to_datetime(target.timestamp, utc=True)
        origin = target_times.min() - pd.Timedelta(minutes=5)
        expected = pd.date_range(origin + pd.Timedelta(minutes=5), periods=6, freq="5min")
        if not pd.DatetimeIndex(sorted(target_times.unique())).equals(expected):
            raise ValueError("Queue template must cover T+5 through T+30")
        at_origin = history.iloc[:0].copy()
        if model.policy == "origin_observed":
            date = origin.strftime("%Y_%m_%d")
            if date not in origin_cache:
                parts = []
                for path in (root / "corridors" / panel / split / "mainline_states_masked").rglob(f"*{date}.parquet"):
                    columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy",
                        "density_occ_linear_vehpkm", "pct_observed", "is_score_eligible"]
                    raw = pd.read_parquet(path, columns=columns)
                    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
                    raw["link_id"] = raw.link_id.astype(str)
                    parts.append(raw)
                if not parts:
                    raise FileNotFoundError(f"No released origin observations for {panel}/{split}/{date}")
                origin_cache[date] = pd.concat(parts, ignore_index=True)
            at_origin = origin_cache[date]
        h, auxiliary, times = history_arrays(history[history.window_id.eq(row.window_id)],
            at_origin, origin, payload["links"], model.policy, payload["density_scale"])
        p = model.scores(h, auxiliary, times, str(row.condition), payload["up"], payload["down"], payload["params"])
        predicted = decode_queue(p, model.decoders[str(row.condition)])
        ti = expected.get_indexer(target_times)
        ni = pd.Index(payload["links"]).get_indexer(target.link_id)
        if (ti < 0).any() or (ni < 0).any():
            raise ValueError("Unknown queue target time or link")
        target["queue_pred"] = predicted[ti, ni].astype(int)
        outputs.append(target)
    return pd.concat(outputs, ignore_index=True)
