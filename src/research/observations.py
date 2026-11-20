"""Normalize measured feeds and prepare windows with a strict visibility cutoff."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse

from .pems import AmbiguousFeedTime, CADENCE_MS, PEMS_TIMEZONE, compact_rows, local_interval

UNITS = {"speed": "km/h", "flow": "veh/h", "count": "veh", "occupancy": "fraction", "storage": "veh"}


class NoSnapshots(ValueError):
    pass


@dataclass(frozen=True)
class Window:
    source: str
    site: str
    time_ms: np.ndarray
    arrival_ms: np.ndarray
    cell_ids: tuple[str, ...]
    positions_m: np.ndarray
    values: dict[str, np.ndarray]
    raw_sha256: tuple[str, ...]
    topology_sha256: tuple[str, ...]
    quality: np.ndarray
    cadence_ms: int

    def __post_init__(self) -> None:
        shape = (len(self.time_ms), len(self.cell_ids))
        if (self.cadence_ms <= 0 or not all(shape) or len(set(self.cell_ids)) != shape[1]
                or self.positions_m.shape != (shape[1],) or not np.isfinite(self.positions_m).all()
                or np.any(np.diff(self.positions_m) <= 0)
                or np.any(np.diff(self.time_ms) != self.cadence_ms) or self.arrival_ms.shape != (shape[0],)
                or self.quality.shape != shape or np.any(self.arrival_ms < self.time_ms + self.cadence_ms)
                or not np.isfinite(self.quality).all() or np.any(self.quality < 0) or np.any(self.quality > 1)
                or any(channel not in UNITS or values.shape != shape or np.isinf(values).any()
                       for channel, values in self.values.items())):
            raise ValueError("Traffic window has invalid topology, cadence, timing, or observations")

    @property
    def graph(self) -> sparse.csc_matrix:
        n = len(self.cell_ids)
        return sparse.diags(np.ones(n - 1), -1, shape=(n, n), format="csc")

    def inputs(
        self, channel: str, *, hidden: np.ndarray | None = None, as_of_ms: int | None = None,
        min_quality: float = 1.0,
    ) -> np.ndarray:
        allowed = self.quality >= min_quality
        if hidden is not None:
            allowed &= ~hidden
        if as_of_ms is not None:
            allowed &= self.arrival_ms[:, None] <= as_of_ms
            allowed &= self.time_ms[:, None] + self.cadence_ms <= as_of_ms
        return np.where(allowed, self.values[channel], np.nan)


def _optional_number(value: str) -> float:
    return float(value) if value else np.nan


def load_pems_corridor(
    root: Path, district: int, freeway: int, direction: str, *, cells: int = 8, after_ms: int = 0,
    first_station: str | None = None, max_gap_m: float = 5000, before_ms: int | None = None,
    available_by_ms: int | None = None,
) -> Window:
    """Build a regular station grid from immutable snapshots of the compact feed."""
    directory = root / f"D{district}"
    zone = ZoneInfo(PEMS_TIMEZONE)
    lower_day = datetime.fromtimestamp(after_ms / 1000, zone).strftime("%Y%m%d") if after_ms else ""
    end_ms = before_ms if available_by_ms is None else min(before_ms or available_by_ms, available_by_ms)
    upper_day = datetime.fromtimestamp((end_ms - 1) / 1000, zone).strftime("%Y%m%d") if end_ms is not None else "99999999"
    receipts = []
    for path in sorted(directory.glob("*.json")):
        if not lower_day <= path.stem[:8] <= upper_day:
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        try:
            stamp = int(local_interval(receipt["feed_local_time"]).timestamp() * 1000)
        except AmbiguousFeedTime:
            continue
        if stamp < after_ms or (end_ms is not None and stamp >= end_ms):
            continue
        arrival = max(stamp + CADENCE_MS, int(datetime.fromisoformat(receipt["arrival"]).timestamp() * 1000))
        if available_by_ms is None or arrival <= available_by_ms:
            receipts.append((stamp, arrival, receipt))
    if not receipts:
        raise NoSnapshots(f"No PeMS snapshots in the requested range in {directory}")
    topology = tuple(sorted({r["topology_sha256"] for _, _, r in receipts}))
    station_maps = []
    for digest in topology:
        metadata = gzip.decompress((root / "station-metadata" / f"{digest}.xml.gz").read_bytes())
        if hashlib.sha256(metadata).hexdigest() != digest:
            raise ValueError("PeMS station metadata digest changed")
        stations = ET.fromstring(metadata).findall(f".//district[@id='{district}']//vds")
        station_maps.append({
            s.attrib["id"]: float(s.attrib["abs_pm"]) * (1 if direction in ("N", "E") else -1)
            for s in stations if s.get("type") == "ML" and s.get("freeway_id") == str(freeway)
            and s.get("freeway_dir") == direction and s.get("abs_pm") is not None
        })
    stable = {key: value for key, value in station_maps[0].items()
              if all(other.get(key) == value for other in station_maps[1:])}
    ordered = sorted((position, key) for key, position in stable.items())
    if first_station is not None:
        ordered = ordered[next((i for i, (_, key) in enumerate(ordered) if key == first_station), len(ordered)):]
    for start in range(len(ordered) - cells + 1):
        sequence = ordered[start:start + cells]
        raw_positions, ids = zip(*sequence)
        positions = (np.asarray(raw_positions) - raw_positions[0]) * 1609.344
        if np.all((np.diff(positions) > 0) & (np.diff(positions) <= max_gap_m)):
            if first_station is None or ids[0] == first_station:
                break
    else:
        raise ValueError("No contiguous, unambiguous mainline corridor in the metadata")
    indices = {station: i for i, station in enumerate(ids)}
    grid = {}
    hashes = []
    for key, arrival, receipt in receipts:
        raw = gzip.decompress((directory / receipt["raw_file"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != receipt["raw_sha256"]:
            raise ValueError("PeMS snapshot digest changed")
        stamp, rows = compact_rows(raw)
        if int(stamp.timestamp() * 1000) != key:
            raise ValueError("PeMS snapshot timestamp disagrees with its receipt")
        values = np.full((4, cells), np.nan)
        quality = np.zeros(cells)
        for row in rows:
            if row[0] in indices:
                i = indices[row[0]]
                count, occ, speed = (_optional_number(row[j]) for j in (1, 2, 3))
                observed = _optional_number(row[10]) / 100
                if ((np.isfinite(count) and count < 0) or (np.isfinite(speed) and speed < 0)
                        or (np.isfinite(occ) and not 0 <= occ <= 1) or (np.isfinite(observed) and not 0 <= observed <= 1)):
                    raise ValueError("PeMS count, speed, occupancy, or percent observed is outside its domain")
                values[:, i] = [speed * 1.609344, count * 12, count, occ]
                quality[i] = observed if np.isfinite(observed) else 0
        grid[key] = arrival, values, quality
        hashes.append(receipt["raw_sha256"])
    if not grid:
        raise ValueError("No PeMS snapshots in the requested period")
    cadence = CADENCE_MS
    times = np.arange(after_ms or min(grid), before_ms or max(grid) + cadence, cadence)
    arrivals = np.empty(len(times), dtype=np.int64)
    arrays = np.full((len(times), 4, cells), np.nan)
    quality = np.zeros((len(times), cells))
    for t, stamp in enumerate(times):
        arrivals[t] = stamp + cadence
        if stamp in grid:
            arrivals[t], arrays[t], quality[t] = grid[stamp]
    return Window("PeMS", f"D{district}-{freeway}{direction}-{ids[0]}", times, arrivals, ids, positions,
                  dict(zip(("speed", "flow", "count", "occupancy"), arrays.transpose(1, 0, 2))),
                  tuple(hashes), topology, quality, cadence)


def recording_partition(directory: Path, entry: dict) -> bytes:
    filename = Path(entry["filename"])
    if filename.name != entry["filename"] or filename.suffix != ".gz":
        raise ValueError(f"Invalid trajectory partition filename at {directory}")
    path = directory / filename
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["gzip_sha256"]:
        raise ValueError(f"Trajectory partition checksum changed for {path}")
    data = gzip.decompress(payload)
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Trajectory partition content checksum changed for {path}")
    return data


def read_recording(directory: Path, index: int) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    manifest = json.loads((directory / "recordings.json").read_text(encoding="utf-8"))
    period = manifest["periods"][index]
    frames = []
    hashes = []
    for entry in period["files"]:
        data = recording_partition(directory, entry)
        frames.append(pd.read_csv(io.BytesIO(data), usecols=["vehicle_id", "global_time", "local_y", "v_vel"]))
        hashes.append(entry["sha256"])
    return pd.concat(frames, ignore_index=True), tuple(hashes), f"{manifest['site']}-{period['first']}"


def ngsim_window(
    tracks: pd.DataFrame, hashes: tuple[str, ...], site: str, *,
    cadence_ms: int = 60_000, cells: int = 4, max_gap_ms: int = 200,
) -> tuple[Window, np.ndarray]:
    """Edie integrals with explicit tracked storage and counted boundary crossings."""
    if cadence_ms <= 0 or cells < 2 or tracks.empty:
        raise ValueError("Invalid trajectory grid or coordinates")
    x = tracks["local_y"].to_numpy(dtype=float) * 0.3048
    if not np.isfinite(x).all():
        raise ValueError("Invalid trajectory grid or coordinates")
    edges = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), cells + 1)
    del x
    if edges[-1] <= edges[0]:
        raise ValueError("NGSIM recording has no usable spatial span")
    if not tracks["global_time"].is_monotonic_increasing:
        tracks = tracks.sort_values("global_time", kind="stable")
    recording_times = tracks["global_time"].to_numpy(dtype=np.int64)
    starts = np.arange(recording_times[0] + cadence_ms, recording_times[-1] - cadence_ms + 1, cadence_ms)
    q, speed, storage, counts, downstream = np.full((5, len(starts), cells), np.nan)
    for k, start in enumerate(starts):
        end = start + cadence_ms
        # Include the rows needed to reconstruct segments that cross either time boundary.
        lo, hi = np.searchsorted(recording_times, [start - max_gap_ms, end + max_gap_ms])
        rows = tracks.iloc[lo:hi][["vehicle_id", "global_time", "local_y"]]
        rows = rows.sort_values(["vehicle_id", "global_time"])
        t = rows["global_time"].to_numpy(dtype=np.int64)
        pos = rows["local_y"].to_numpy(dtype=float) * 0.3048
        vehicle = rows["vehicle_id"].to_numpy()
        dt = np.diff(t)
        adjacent = (vehicle[:-1] == vehicle[1:]) & (dt > 0) & (dt <= max_gap_ms)
        a, b, x0, x1 = [v[adjacent] for v in (t[:-1], t[1:], pos[:-1], pos[1:])]
        dx = x1 - x0
        active = (a < end) & (b > start)
        aa, bb, xx, dd = a[active], b[active], x0[active], dx[active]
        dtime = bb - aa
        t_lo = np.maximum(0, (start - aa) / dtime)
        t_hi = np.minimum(1, (end - aa) / dtime)
        snapshot = (a <= start) & (b > start)
        positions = x0[snapshot] + dx[snapshot] * (start - a[snapshot]) / (b[snapshot] - a[snapshot])
        for i, (left, right) in enumerate(zip(edges, edges[1:])):
            moving = dd != 0
            left_alpha = np.divide(left - xx, dd, out=np.zeros_like(xx), where=moving)
            right_alpha = np.divide(right - xx, dd, out=np.ones_like(xx), where=moving)
            part = np.maximum(0, np.minimum(t_hi, np.maximum(left_alpha, right_alpha))
                              - np.maximum(t_lo, np.minimum(left_alpha, right_alpha)))
            part[~moving & ((xx < left) | (xx >= right))] = 0
            distance = float(np.dot(part, np.abs(dd)))
            travel_ms = float(np.dot(part, dtime))
            q[k, i] = distance / ((right - left) * cadence_ms) * 3_600_000 if travel_ms else np.nan
            speed[k, i] = distance / travel_ms * 3600 if travel_ms else np.nan
            storage[k, i] = np.sum((left <= positions) & (positions < right))
            counts[k, i] = np.sum((dd > 0) & (left_alpha >= t_lo) & (left_alpha < t_hi) & (left_alpha < 1))
            downstream[k, i] = np.sum((dd > 0) & (right_alpha >= t_lo) & (right_alpha < t_hi) & (right_alpha < 1))
    topology = hashlib.sha256(edges.tobytes()).hexdigest()
    timing = starts + cadence_ms
    return Window("NGSIM", site, starts, timing, tuple(f"cell-{i}" for i in range(cells)), edges[:-1],
                  {"speed": speed, "flow": q, "storage": storage, "count": counts},
                  hashes, (topology,), np.ones(speed.shape), cadence_ms), downstream
