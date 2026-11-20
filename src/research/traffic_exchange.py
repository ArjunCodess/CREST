"""Export scorer-ready synthetic fixtures and score estimates without leaking targets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .observations import ngsim_window, read_recording
from .pems import atomic_write
from .traffic_estimation import outage_masks, persistence
from .traffic_simulator import simulate


def _archive(path: Path, **arrays: np.ndarray) -> str:
    content = io.BytesIO()
    np.savez_compressed(content, **arrays)
    payload = content.getvalue()
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _export_case(
    output: Path, scenario: str, target: np.ndarray, mask: np.ndarray, **geometry: np.ndarray,
) -> dict:
    observed = np.isfinite(target)
    visible = observed & ~mask[None]
    inputs = output / "inputs" / f"{scenario}.npz"
    scoring = output / "scorer-only" / f"{scenario}.npz"
    return {
        "id": scenario,
        "input_sha256": _archive(inputs, visible=visible, observed=np.where(visible, target, np.nan), **geometry),
        "scorer_sha256": _archive(scoring, target=target, eligible=observed & mask[None]),
    }


def export_suite(output: Path, splits: dict[str, tuple[int, ...]]) -> dict:
    cases = []
    for split, seeds in splits.items():
        for seed in seeds:
            layout = "junction" if split == "evaluation" else "chain"
            simulation = simulate(seed, layout)
            window = simulation.window
            target = np.stack([window.values[channel] for channel in ("speed", "flow", "storage")])
            for index, mask in enumerate(outage_masks(window.quality.shape)):
                scenario = f"{split}-{seed}-{index}"
                case = _export_case(output, scenario, target, mask, time_ms=window.time_ms,
                                    cell_position_m=window.positions_m, incidence=simulation.incidence,
                                    capacity_vehicles=simulation.capacity)
                cases.append({**case, "split": split, "seed_cluster": seed, "layout": layout})
    manifest = {"type": "synthetic", "channels": ["speed_kmh", "flow_vph", "storage_veh"],
                "schema_version": 1, "cases": cases}
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def export_trajectory_suite(output: Path, root: Path) -> dict:
    """Export derived FHWA observations with a complete recording in each split."""
    license_url = "https://creativecommons.org/licenses/by-sa/3.0/"
    cases = []
    recordings = []
    for site in ("i-80", "us-101"):
        directory = root / site
        manifest = json.loads((directory / "recordings.json").read_text(encoding="utf-8"))
        for index, period in enumerate(manifest["periods"]):
            split = "evaluation" if site == "us-101" else "train" if index == 0 else "calibration"
            tracks, hashes, recording = read_recording(directory, index)
            window, _ = ngsim_window(tracks, hashes, recording)
            target = np.stack([window.values[key] for key in ("speed", "flow", "storage")])
            cells = len(window.cell_ids)
            incidence = np.eye(cells, cells + 1) - np.eye(cells + 1)[1:]
            recordings.append({
                "recording": recording, "split": split, "first_ms": period["first"], "last_ms": period["last"],
                "rows": sum(part["rows"] for part in period["files"]), "source_hashes": hashes,
                "topology_sha256": window.topology_sha256,
            })
            for index, mask in enumerate(outage_masks(window.quality.shape)):
                scenario = f"{split}-{recording}-{index}"
                case = _export_case(output, scenario, target, mask, time_ms=window.time_ms,
                                    cell_position_m=window.positions_m, incidence=incidence)
                cases.append({**case, "split": split, "recording_cluster": recording, "layout": "chain"})
    manifest = {
        "type": "ngsim_derived", "schema_version": 1, "channels": ["speed_kmh", "flow_vph", "storage_veh"],
        "source": "FHWA Next Generation Simulation Vehicle Trajectories and Supporting Data",
        "source_url": "https://data.transportation.gov/d/8ect-6jqj",
        "source_license": "CC-BY-SA-3.0", "derived_data_license": "CC-BY-SA-3.0", "license_url": license_url,
        "transform": "One-minute Edie speed/flow, tracked storage, mainline segment aggregation and outage masking",
        "recordings": recordings, "cases": cases,
    }
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    atomic_write(output / "ATTRIBUTION.txt", (
        "Derived from FHWA Next Generation Simulation Vehicle Trajectories and Supporting Data.\n"
        f"Source: {manifest['source_url']}\n"
        f"Changes: {manifest['transform']}. The package omits raw trajectories and vehicle identifiers.\n"
        f"Derived data are distributed under Creative Commons Attribution-ShareAlike 3.0 Unported.\n{license_url}\n"
    ).encode())
    return manifest


def score_submission(suite: Path, scenario_id: str, submission: Path) -> dict:
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    matching = next((entry for entry in manifest["cases"] if entry["id"] == scenario_id), None)
    if matching is None:
        raise ValueError("Unknown benchmark scenario")
    with np.load(submission, allow_pickle=False) as output:
        point = output["point"]
        lower = output["lower"] if "lower" in output else None
        upper = output["upper"] if "upper" in output else None
        fallback = output["fallback"] if "fallback" in output else np.zeros(point.shape, dtype=bool)
    label_file = (suite / "scorer-only" / f"{scenario_id}.npz").read_bytes()
    if hashlib.sha256(label_file).hexdigest() != matching["scorer_sha256"]:
        raise ValueError("Scorer targets changed since export")
    with np.load(io.BytesIO(label_file), allow_pickle=False) as labels:
        target, eligible = labels["target"], labels["eligible"]
    if point.shape != target.shape or np.isinf(point).any() or fallback.shape != target.shape or fallback.dtype != bool:
        raise ValueError("Provide a shape-matched point array and an optional boolean fallback mask")
    if lower is not None or upper is not None:
        if (lower is None or upper is None or lower.shape != target.shape or upper.shape != target.shape
                or np.isinf(lower).any() or np.isinf(upper).any()
                or np.any(np.isfinite(lower) != np.isfinite(upper)) or np.any(lower > upper)):
            raise ValueError("Provide paired, ordered lower and upper arrays or omit both")
    results = {}
    for i, channel in enumerate(manifest["channels"]):
        y, prediction = target[i], point[i]
        mask = eligible[i] & np.isfinite(y)
        scored = mask & np.isfinite(prediction)
        errors = np.abs(y[scored] - prediction[scored])
        result = {"eligible": int(mask.sum()), "point_scored": int(scored.sum()), "abstained": int((mask & ~scored).sum()),
                  "fallbacks_scored": int((scored & fallback[i]).sum()),
                  "point_mae": float(errors.mean()) if errors.size else None}
        if lower is not None:
            bounds = mask & np.isfinite(lower[i])
            n = int(bounds.sum())
            result["interval_scored"] = n
            result["interval_coverage"] = float(np.mean((lower[i][bounds] <= y[bounds]) & (y[bounds] <= upper[i][bounds]))) if n else None
            result["mean_interval_width"] = float(np.mean(upper[i][bounds] - lower[i][bounds])) if n else None
        results[channel] = result
    return {"scenario": scenario_id, "channels": results}


def evaluate_export(suite: Path) -> list[dict]:
    """Exercise the release scorer on baselines that can only read participant inputs."""
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    training = []
    evaluation = []
    fields = {"observed", "visible", "time_ms", "cell_position_m", "incidence"}
    if manifest["type"] == "synthetic":
        fields.add("capacity_vehicles")
    for entry in manifest["cases"]:
        payload = (suite / "inputs" / f"{entry['id']}.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["input_sha256"]:
            raise ValueError(f"Participant input checksum changed in {entry['id']}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as inputs:
            observed, visible = inputs["observed"], inputs["visible"]
            if (
                set(inputs.files) != fields
                or not np.array_equal(np.isfinite(observed), visible)
            ):
                raise ValueError(f"Unexpected participant input visibility in {entry['id']}")
        if entry["split"] == "train":
            training.append(observed.reshape(observed.shape[0], -1))
        elif entry["split"] == "evaluation":
            evaluation.append((entry, observed))
    levels = np.nanmedian(np.concatenate(training, axis=1), axis=1)
    rows = []
    with TemporaryDirectory() as directory:
        prediction = Path(directory) / "prediction.npz"
        for entry, observed in evaluation:
            history = np.where(np.isfinite(observed), observed, levels[:, None, None])
            latest = np.stack([persistence(channel, level) for channel, level in zip(observed, levels)])
            interpolated = history.copy()
            time = np.arange(observed.shape[1])
            for channel, cells in enumerate(observed.transpose(0, 2, 1)):
                for cell, values in enumerate(cells):
                    seen = np.isfinite(values)
                    if seen.any():
                        interpolated[channel, :, cell] = np.interp(time, time[seen], values[seen])
            for method, point in (
                ("historical_median", history), ("persistence", latest), ("linear_interpolation", interpolated),
                ("observed_only", observed),
            ):
                np.savez(prediction, point=point)
                results = score_submission(suite, entry["id"], prediction)
                cluster = {key: entry[key] for key in ("seed_cluster", "recording_cluster") if key in entry}
                rows.append({"method": method, **cluster, **results})
    return rows
