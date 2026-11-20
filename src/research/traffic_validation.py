"""Development experiments for operational outages and incremental replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .observations import Window, ngsim_window, read_recording
from .pems import PEMS_TIMEZONE
from .traffic_decisions import evaluate_queues
from .traffic_estimation import QueueHead, congested, graph_reconstruct, persistence
from .traffic_replay import Replay, outage_triage
from .traffic_simulator import simulate


def operational_outages(window: Window) -> list[dict]:
    """Replay fixed blocks on measured speed, scoring only available reference readings."""
    zone = ZoneInfo(PEMS_TIMEZONE)
    days = np.array([datetime.fromtimestamp(t / 1000, zone).date().isoformat() for t in window.time_ms])
    target = 3
    truth = window.inputs("speed")
    graph = window.graph
    neighborhood = truth[:, target - 1:target + 2]
    usable = np.isfinite(neighborhood).all(axis=1)
    reference_queue = congested(truth, 50.)[:, target]
    results = []
    for day in sorted(set(days)):
        period = np.flatnonzero(days == day)
        for duration in (6, 12):
            # Fixed candidates on the archive's time grid. Keep distinct episodes per duration.
            for start in range(int(period[0]) + 6, int(period[-1]) - duration, duration + 3):
                stop = start + duration
                if not usable[start - 1:stop + 1].all():
                    continue
                origins = window.time_ms[start:min(stop + 3, period[-1] + 1)] + 2 * window.cadence_ms
                if window.arrival_ms[start - 1] > origins[0]:
                    continue
                for fault, delay_steps in (("blackout", 0), ("delay", 3)):
                    methods = {name: Counter() for name in ("triage_graph", "latest_reading")}
                    actions = Counter()
                    recovery = None
                    decision_recovery = dict.fromkeys(methods)
                    for i, origin in enumerate(origins, start):
                        history = slice(max(0, i - 11), i + 1)
                        inputs = np.where(
                            window.arrival_ms[history, None] <= origin, truth[history], np.nan,
                        )
                        affected = (np.arange(history.start, i + 1) >= start) & (np.arange(history.start, i + 1) < stop)
                        delayed = window.arrival_ms[history] + delay_steps * window.cadence_ms > origin
                        inputs[affected & (delayed if delay_steps else True), target] = np.nan
                        if i >= stop and np.isfinite(inputs[-1, target]) and recovery is None:
                            recovery = int(origin - (window.time_ms[stop] + window.cadence_ms))
                        if not np.isfinite(inputs).any():
                            if i < stop:
                                for metrics in methods.values():
                                    metrics["eligible"] += 1
                                    metrics["queue_proxy"] += bool(reference_queue[i])
                            continue
                        level = float(np.median(inputs[np.isfinite(inputs)]))
                        baseline = persistence(inputs, level)
                        triage = outage_triage(inputs, level, target)
                        robust = graph_reconstruct(inputs, baseline, graph, 25.)
                        if i < stop:
                            actions[triage["action"]] += 1
                        for name, value, alert, fallback in (
                            ("triage_graph", robust.values[-1, target], triage["action"] == "corroborate_queue",
                             robust.status != "optimal"),
                            ("latest_reading", baseline[-1, target], bool(congested(baseline[-1], 50.)[target]),
                             not np.isfinite(inputs[:, target]).any()),
                        ):
                            if i >= stop:
                                if (usable[i] and np.isfinite(inputs[-1, target]) and alert == reference_queue[i]
                                        and decision_recovery[name] is None):
                                    decision_recovery[name] = int(origin - (window.time_ms[stop] + window.cadence_ms))
                                continue
                            metrics = methods[name]
                            metrics["eligible"] += 1
                            metrics["scored"] += 1
                            metrics["abs_error_kmh"] += float(abs(value - truth[i, target]))
                            metrics["queue_proxy"] += bool(reference_queue[i])
                            metrics["false_queue_checks"] += bool(alert and not reference_queue[i])
                            metrics["missed_queues"] += bool(reference_queue[i] and not alert)
                            metrics["fallbacks"] += fallback
                    results.append({
                        "day_pacific": day, "start_ms": int(window.time_ms[start]), "site": window.site,
                        "fault": fault, "fault_minutes": duration * window.cadence_ms / 60_000,
                        "added_delay_ms": delay_steps * window.cadence_ms,
                        "fresh_reading_recovery_ms": recovery, "decision_recovery_ms": decision_recovery,
                        "actions": dict(actions),
                        "methods": {name: dict(metrics) for name, metrics in methods.items()},
                    })
    return results


def replay_benchmark(windows: list[Window], output: Path) -> list[dict]:
    """Compare the same late edits with a warm cache and a forced full rebuild."""
    long_synthetic = simulate(811, "junction", cells=8, steps=576).window
    rows = []
    for window in [*windows, long_synthetic]:
        speed = window.inputs("speed")
        level = float(np.median(speed[np.isfinite(speed)]))
        target = 3
        candidates = np.flatnonzero(np.isfinite(speed[:, target]))
        if len(candidates) < 12:
            continue
        for lookahead in (0, 2):
            root = output / window.site / f"lookahead-{lookahead}"
            replay = Replay(root, window.graph, "+".join(window.topology_sha256), level, 25., lookahead=lookahead)
            baseline = replay.ingest(speed, window.time_ms)
            before = replay.run(baseline, full=True)
            edits = [int(window.time_ms[candidates[i]]) for i in (len(candidates) * 3 // 4, len(candidates) // 4)]
            revised = replay.correct(baseline, edits[0], target, None)
            revised = replay.correct(revised, edits[1], target, None)
            chronological = replay.correct(replay.correct(baseline, edits[1], target, None), edits[0], target, None)
            duplicate = replay.correct(revised, edits[1], target, None)
            # Reopen from the manifest before comparing the outputs, as in the CLI.
            replay = Replay.from_manifest(root, revised)
            incremental = replay.run(revised, previous=baseline)
            full = replay.run(revised, full=True)
            changed = sum(old != incremental["outputs"][stamp] for stamp, old in before["outputs"].items())
            matches = incremental["outputs"] == full["outputs"]
            if not matches or revised != chronological or revised != duplicate or changed != incremental["recomputed"]:
                raise RuntimeError(f"Incremental correction violated replay invariants for {window.site}")
            rows.append({
                "site": window.site, "source": window.source, "windows": len(speed), "lookahead": lookahead,
                "baseline_manifest": baseline, "corrected_manifest": revised, "edited_times_ms": edits,
                "recomputed": changed, "unchanged": len(speed) - changed, "matches_full": matches,
                "independent_of_edit_order": revised == chronological, "duplicate_idempotent": revised == duplicate,
                "cold_full_seconds": before["elapsed_seconds"], "full_seconds": full["elapsed_seconds"],
                "incremental_seconds": incremental["elapsed_seconds"],
            })
    return rows


def trajectory_advisory_sweep(root: Path) -> dict:
    """Hold out US-101 and vary delivery conditions on fixed, disjoint time blocks."""
    cadence = 15_000
    block = 300_000
    training = []
    training_dir = root / "i-80"
    period_count = len(json.loads((training_dir / "recordings.json").read_text(encoding="utf-8"))["periods"])
    for index in range(period_count):
        tracks, hashes, site = read_recording(training_dir, index)
        first, last = int(tracks["global_time"].min()), int(tracks["global_time"].max())
        for start in range(first, last - block + 1, block):
            sample = tracks.loc[tracks["global_time"].between(start, start + block, inclusive="left")]
            training.append(ngsim_window(sample, hashes, site, cadence_ms=cadence)[0])
    model = QueueHead(training, horizon=2)
    test_dir = root / "us-101"
    period_count = len(json.loads((test_dir / "recordings.json").read_text(encoding="utf-8"))["periods"])
    rows = []
    for index in range(period_count):
        tracks, hashes, site = read_recording(test_dir, index)
        first, last = int(tracks["global_time"].min()), int(tracks["global_time"].max())
        for start in range(first, last - block + 1, block):
            sample = tracks.loc[tracks["global_time"].between(start, start + block, inclusive="left")]
            window, _ = ngsim_window(sample, hashes, site, cadence_ms=cadence)
            for penetration in (.15, .35, .70):
                for delay in (0, 7_500, 30_000):
                    result = evaluate_queues(
                        model, window, tracks=sample, direction="US-101 southbound",
                        probe_penetration=penetration, communication_delay_ms=delay,
                    )
                    del result["event_rows"]
                    rows.append({"recording": site, "start_ms": start, "stop_ms": start + block, **result})
    return {"training_blocks": len(training), "test_windows": rows, "scope": "trajectory_development_transfer"}


def episode_profile(windows: list[Window]) -> dict:
    """Describe observed gaps and low-speed episodes in elapsed minutes."""
    unavailable, congestion = [], []
    for window in windows:
        speed = window.inputs("speed")
        for mask, episodes in ((~np.isfinite(speed), unavailable), (congested(speed, 50.), congestion)):
            for cell in mask.T:
                edges = np.flatnonzero(np.diff(np.r_[False, cell, False])).reshape(-1, 2)
                episodes.extend((edges[:, 1] - edges[:, 0]) * window.cadence_ms / 60_000)
    return {
        "missing_speed_runs": len(unavailable),
        "missing_speed_p50_p90_minutes": np.quantile(unavailable, (.5, .9)).tolist() if unavailable else None,
        "congestion_proxy_runs": len(congestion),
        "congestion_proxy_p50_p90_minutes": np.quantile(congestion, (.5, .9)).tolist() if congestion else None,
    }


def detector_queue_transfer(windows: list[Window]) -> dict:
    """Train on a source corridor's first day and test a later day at the other site."""
    zone = ZoneInfo(PEMS_TIMEZONE)
    earlier, later = windows
    origin = datetime.fromtimestamp(earlier.time_ms[0] / 1000, zone).date()
    days = [[datetime.fromtimestamp(t / 1000, zone).date() for t in w.time_ms] for w in windows]
    selected = [
        (earlier, np.flatnonzero([day == origin for day in days[0]])),
        (later, np.flatnonzero([day > origin for day in days[1]])),
    ]
    if any(len(rows) < 12 for _, rows in selected):
        return {"status": "insufficient_time_blocked_windows"}
    training, evaluation = [
        replace(w, time_ms=w.time_ms[rows], arrival_ms=w.arrival_ms[rows], quality=w.quality[rows],
                values={channel: values[rows] for channel, values in w.values.items()})
        for w, rows in selected
    ]
    lag = training.cadence_ms
    head = QueueHead([training], processing_lag_ms=lag)
    result = evaluate_queues(head, evaluation, processing_lag_ms=lag)
    return {
        "status": "scored", "label": "detector_low_speed_proxy", "processing_lag_ms": lag,
        "training": {"site": training.site, "start_ms": int(training.time_ms[0]), "stop_ms": int(training.time_ms[-1])},
        "evaluation": {"site": evaluation.site, "start_ms": int(evaluation.time_ms[0]), "stop_ms": int(evaluation.time_ms[-1])},
        **result,
    }
