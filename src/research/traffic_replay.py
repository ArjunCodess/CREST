"""Versioned, inspectable corridor replay and queue advisory output."""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .observations import Window
from .pems import atomic_write
from .traffic_estimation import QueuePrediction, congested, graph_reconstruct, persistence

REPLAY_VERSION = 1


@dataclass(frozen=True)
class Advisory:
    observation_ms: int
    expires_ms: int
    direction: str
    available: bool
    tail_interval_m: tuple[float, float] | None
    reason: str
    queue_intervals_m: tuple[tuple[float, float], ...] = ()


def queue_advisory(
    prediction: QueuePrediction, locations: np.ndarray, *, now_ms: int, cadence_ms: int, direction: str,
    minimum_probability: float = 0.60, maximum_width_m: float = 2500., uncertainty_cells: int = 1,
) -> Advisory:
    expires = prediction.origin_ms + min(prediction.horizon, 2) * cadence_ms
    if now_ms < prediction.origin_ms or now_ms >= expires:
        return Advisory(prediction.origin_ms, expires, direction, False, None, "expired_or_future_observation")
    anchors = prediction.available_cells & (
        prediction.current_congestion | (prediction.onset_probability >= minimum_probability)
    )
    nearby = anchors.copy()
    nearby[1:] |= anchors[:-1]
    nearby[:-1] |= anchors[1:]
    support = np.flatnonzero((prediction.probability >= minimum_probability) & nearby)
    if not len(support):
        return Advisory(prediction.origin_ms, expires, direction, False, None, "no_confident_queue")
    groups = np.split(support, np.flatnonzero(np.diff(support) > 1) + 1)
    supported = [group for group in groups if prediction.available_cells[group].any()]
    if not supported:
        return Advisory(prediction.origin_ms, expires, direction, False, None, "stale_or_wide_location")
    tail = supported[0][0]
    lo = max(tail - uncertainty_cells, 0)
    hi = min(tail + uncertainty_cells, len(locations) - 1)
    if not prediction.available_cells[lo:hi + 1].any() or locations[hi] - locations[lo] > maximum_width_m:
        return Advisory(prediction.origin_ms, expires, direction, False, None, "stale_or_wide_location")
    end = locations[-1] + (locations[-1] - locations[-2]) if len(locations) > 1 else locations[-1]
    boundaries = np.r_[locations, end]
    intervals = tuple((float(locations[group[0]]), float(boundaries[group[-1] + 1])) for group in supported)
    return Advisory(prediction.origin_ms, expires, direction, True,
                    (float(locations[lo]), float(locations[hi])), "queue", intervals)


def outage_triage(values: np.ndarray, levels: float, target: int, threshold: float = 50.) -> dict:
    if values.ndim != 2 or not 0 < target < values.shape[1] - 1:
        raise ValueError("Triage needs an interior detector and a time-by-cell input")
    observed = values[-1]
    ages = np.where(np.isfinite(values[:, target]))[0]
    state = persistence(values, levels)[-1]
    queue = congested(observed, threshold)
    neighbors = observed[[target - 1, target + 1]]
    if queue[target] or (np.isfinite(neighbors).all() and np.all(neighbors < threshold)):
        action = "corroborate_queue"
    elif not np.isfinite(observed[target]) and np.isfinite(neighbors).all() and np.all(neighbors >= threshold):
        action = "inspect_detector"
    elif not np.isfinite(observed[target]) or not np.isfinite(neighbors).all():
        action = "insufficient_evidence"
    else:
        action = "continue_monitoring"
    return {"action": action, "last_good_value_kmh": float(values[ages[-1], target]) if len(ages) else None,
            "last_good_age_steps": int(len(values) - 1 - ages[-1]) if len(ages) else None,
            "estimate_kmh": float(state[target]), "velocity_source": "detector" if np.isfinite(observed[target]) else "persistence"}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Replay:
    """Each time partition is immutable. Manifest corrections create another version."""

    def __init__(self, root: Path, graph, topology_digest: str, level: float, scale: float, *, history: int = 3,
                 lookahead: int = 0) -> None:
        if history < 0 or lookahead < 0 or not np.isfinite(level) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid replay settings")
        self.root = root
        self.graph = graph
        self.level = level
        self.scale = scale
        self.history = history
        self.lookahead = lookahead
        adjacency = graph.tocoo()
        self.settings = {"version": REPLAY_VERSION, "topology": topology_digest, "level": level,
                         "scale": scale, "history": history, "lookahead": lookahead,
                         "graph": {"shape": list(graph.shape), "row": adjacency.row.tolist(),
                                   "col": adjacency.col.tolist(), "values": adjacency.data.tolist()},
                         "estimator_sha256": _digest(Path(__file__).with_name("traffic_estimation.py").read_bytes()),
                         "solver_sha256": _digest(Path(__file__).with_name("sparse_traffic.py").read_bytes())}

    @classmethod
    def from_manifest(cls, root: Path, manifest: str) -> Replay:
        from scipy.sparse import coo_matrix

        payload = cls._manifest_payload(root, manifest)
        settings = payload["settings"]
        adjacency = settings["graph"]
        graph = coo_matrix((adjacency["values"], (adjacency["row"], adjacency["col"])), shape=adjacency["shape"])
        replay = cls(root, graph, settings["topology"], settings["level"], settings["scale"],
                     history=settings["history"], lookahead=settings["lookahead"])
        if replay.settings != settings:
            raise ValueError("The stored estimator version differs from the installed one")
        return replay

    def ingest(self, values: np.ndarray, times_ms: np.ndarray) -> str:
        if (values.ndim != 2 or values.shape != (len(times_ms), self.graph.shape[0]) or not len(values)
                or np.isinf(values).any() or np.any(np.diff(times_ms) <= 0)):
            raise ValueError("Replay input must be a sorted, unique time grid")
        return self._manifest({int(stamp): self._write_partition(int(stamp), row) for stamp, row in zip(times_ms, values)})

    def _manifest(self, partitions: dict[int, str]) -> str:
        content = json.dumps({"settings": self.settings, "partitions": partitions}, sort_keys=True).encode()
        digest = _digest(content)
        atomic_write(self.root / "manifests" / f"{digest}.json", content)
        return digest

    @staticmethod
    def _manifest_payload(root: Path, manifest: str) -> dict:
        if len(manifest) != 64 or any(c not in "0123456789abcdef" for c in manifest):
            raise ValueError("Invalid replay manifest digest")
        content = (root / "manifests" / f"{manifest}.json").read_bytes()
        if _digest(content) != manifest:
            raise ValueError("Replay manifest digest changed")
        return json.loads(content)

    def _partitions(self, manifest: str) -> dict[int, str]:
        payload = self._manifest_payload(self.root, manifest)
        if payload["settings"] != self.settings:
            raise ValueError("Manifest digest or replay settings mismatch")
        return {int(time): digest for time, digest in sorted(payload["partitions"].items(), key=lambda item: int(item[0]))}

    def _write_partition(self, stamp: int, row: np.ndarray) -> str:
        buffer = io.BytesIO()
        pq.write_table(pa.table({"time_ms": [stamp] * len(row), "cell": list(range(len(row))),
                                 "speed_kmh": pa.array(row, from_pandas=True)}), buffer)
        body = buffer.getvalue()
        digest = _digest(body)
        if not (self.root / "events" / f"{digest}.parquet").exists():
            atomic_write(self.root / "events" / f"{digest}.parquet", body)
        return digest

    def _read_partition(self, digest: str) -> np.ndarray:
        body = (self.root / "events" / f"{digest}.parquet").read_bytes()
        if _digest(body) != digest:
            raise ValueError("Replay partition checksum changed")
        return pq.read_table(io.BytesIO(body), columns=["speed_kmh"]).column("speed_kmh").to_numpy()

    def correct(self, manifest: str, stamp: int, cell: int, speed: float | None) -> str:
        partitions = self._partitions(manifest)
        if stamp not in partitions or not 0 <= cell < self.graph.shape[0] or (
            speed is not None and (not np.isfinite(speed) or speed < 0)
        ):
            raise ValueError("Invalid detector correction")
        row = self._read_partition(partitions[stamp]).copy()
        row[cell] = np.nan if speed is None else speed
        partitions[stamp] = self._write_partition(stamp, row)
        return self._manifest(partitions)

    def run(self, manifest: str, *, full: bool = False, previous: str | None = None) -> dict:
        started = time.perf_counter()
        partitions = self._partitions(manifest)
        prior_partitions = self._partitions(previous) if previous is not None else None
        times = list(partitions)
        output = {}
        recomputed = 0
        loaded = {}
        for i, stamp in enumerate(times):
            start, end = max(0, i - self.history), min(len(times), i + self.lookahead + 1)
            dependencies = {time: partitions[time] for time in times[start:end]}
            key = _digest(json.dumps([self.settings, stamp, dependencies], sort_keys=True).encode())
            cached = self.root / "results" / f"{key}.json"
            unchanged = prior_partitions is None or all(prior_partitions.get(t) == d for t, d in dependencies.items())
            if cached.exists() and not full and unchanged:
                output[stamp] = json.loads(cached.read_text(encoding="utf-8"))
                continue
            rows = []
            for digest in dependencies.values():
                if digest not in loaded:
                    loaded[digest] = self._read_partition(digest)
                rows.append(loaded[digest])
            inputs = np.vstack(rows)
            prior = persistence(inputs, self.level)
            prediction = graph_reconstruct(inputs, prior, self.graph, self.scale)
            status = {"estimates": prediction.values[i - start].tolist(), "solver_status": prediction.status,
                      "dependencies": dependencies, "output_key": key}
            # JSON keys must have the same representation on cache hits and misses.
            body = json.dumps(status, allow_nan=False).encode()
            atomic_write(cached, body)
            output[stamp] = json.loads(body)
            recomputed += 1
        return {"manifest": manifest, "outputs": output, "recomputed": recomputed,
                "elapsed_seconds": time.perf_counter() - started}

    def compare(self, before: str, after: str) -> dict:
        previous = self.run(before)["outputs"]
        edited = self.run(after, previous=before)
        outputs = edited.pop("outputs")
        changed = [
            {"time_ms": stamp, "before": previous.get(stamp), "after": outputs.get(stamp)}
            for stamp in sorted(previous.keys() | outputs.keys())
            if previous.get(stamp) != outputs.get(stamp)
        ]
        return {**edited, "previous_manifest": before, "affected_windows": changed,
                "unaffected_windows": len(previous.keys() & outputs.keys()) -
                sum(row["before"] is not None and row["after"] is not None for row in changed)}


def replay_case(window: Window, root: Path, target: int) -> dict:
    values = window.inputs("speed")
    if len(values) < 3 or not np.isfinite(values).any():
        raise ValueError("Replay requires three windows and a supported detector")
    level = float(np.nanmedian(values[0]))
    if not np.isfinite(level):
        raise ValueError("Replay requires an initial calibration reading")
    replay = Replay(root, window.graph, ":".join(window.topology_sha256), level, 25.)
    baseline = replay.ingest(values, window.time_ms)
    warm = replay.run(baseline, full=True)
    candidates = np.flatnonzero(np.isfinite(values[:, target]) & (np.arange(len(values)) > 0))
    if not len(candidates):
        raise ValueError("No observed target readings available for outage replay")
    i = int(candidates[len(candidates) // 2])
    stamp = int(window.time_ms[i])
    revised = replay.correct(baseline, stamp, target, None)
    incremental = replay.run(revised, previous=baseline)
    full = replay.run(revised, full=True)
    hidden = values[:i + 1].copy()
    hidden[-1, target] = np.nan
    notice = outage_triage(hidden, level, target)
    notice["robust_estimate_kmh"] = incremental["outputs"][stamp]["estimates"][target]
    notice["reconstruction_abs_error_kmh"] = abs(notice["robust_estimate_kmh"] - values[i, target])
    notice["baseline_abs_error_kmh"] = abs(notice["estimate_kmh"] - values[i, target])
    queue_cases = np.flatnonzero(congested(values, 50.)[:, target])
    congestion = None
    if len(queue_cases):
        q = int(queue_cases[0])
        congestion = {"time_ms": int(window.time_ms[q]), **outage_triage(values[:q + 1], level, target)}
    return {"source": window.source, "site": window.site, "target": window.cell_ids[target],
            "raw_sha256": window.raw_sha256, "topology_sha256": window.topology_sha256,
            "outage": {"time_ms": stamp, "injected": True, **notice}, "congestion": congestion,
            "baseline_manifest": baseline, "corrected_manifest": revised,
            "matches_full_recompute": incremental["outputs"] == full["outputs"],
            "unchanged_outputs": sum(warm["outputs"][k] == full["outputs"][k] for k in full["outputs"]),
            "window_count": len(values), "recomputed": incremental["recomputed"],
            "incremental_seconds": incremental["elapsed_seconds"], "full_seconds": full["elapsed_seconds"]}
