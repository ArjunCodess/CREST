"""Export scorer-ready synthetic fixtures and score estimates without leaking targets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .pems import atomic_write
from .traffic_estimation import outage_masks
from .traffic_simulator import simulate


def _archive(path: Path, **arrays: np.ndarray) -> str:
    content = io.BytesIO()
    np.savez_compressed(content, **arrays)
    payload = content.getvalue()
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


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
                visible = ~mask[None] & np.isfinite(target)
                masked = np.where(visible, target, np.nan)
                inputs = output / "inputs" / f"{scenario}.npz"
                scoring = output / "scorer-only" / f"{scenario}.npz"
                input_sha = _archive(inputs, visible=visible, observed=masked, time_ms=window.time_ms,
                                     cell_position_m=window.positions_m, incidence=simulation.incidence,
                                     capacity_vehicles=simulation.capacity)
                target_sha = _archive(scoring, target=target, eligible=np.broadcast_to(mask, target.shape).copy())
                cases.append({"id": scenario, "split": split, "seed_cluster": seed, "layout": layout,
                              "input_sha256": input_sha, "scorer_sha256": target_sha})
    manifest = {"type": "synthetic", "channels": ["speed_kmh", "flow_vph", "storage_veh"],
                "schema_version": 1, "cases": cases}
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2).encode())
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
