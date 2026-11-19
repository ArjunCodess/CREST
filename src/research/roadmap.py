"""CREST outage replay, measured-data pilots, and frozen benchmark runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .observations import NoSnapshots, Window, load_pems_corridor, ngsim_window, read_recording
from .pems import atomic_write
from .protocol import registry_file_lock
from .provenance import research_code_content_hash
from .traffic_benchmark import StateBenchmark, aggregate, physics_suite, storage_scaling
from .traffic_decisions import evaluate_queues, route_suite
from .traffic_estimation import QueueHead, fit_storage, graph_reconstruct, historical_level, outage_masks, persistence
from .traffic_exchange import export_suite, score_submission
from .traffic_replay import Replay, outage_triage, replay_case
from .traffic_simulator import simulate

DEVELOP = (11, 12, 13, 14)
CALIBRATE = (101, 102, 103)
FIXTURE = (241, 242, 243, 244)
PEMS = ((4, 880, "S", "401557"), (7, 10, "W", "717258"))


def json_write(path: Path, payload: dict) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def measured_pems(root: Path) -> list[Window]:
    return [load_pems_corridor(root, d, f, direction, first_station=first) for d, f, direction, first in PEMS]


def _cohort_signature(window: Window) -> dict:
    return {"source": window.source, "site": window.site, "start_ms": int(window.time_ms[0]),
            "end_ms": int(window.time_ms[-1] + window.cadence_ms), "topology": window.topology_sha256,
            "source_hashes": window.raw_sha256}


def shadow_cycle(pems_root: Path, output: Path, as_of_ms: int) -> dict:
    """Publish a read-only operator snapshot and idempotent per-window decisions."""
    start = time.perf_counter()
    sites = []
    for district, freeway, direction, first in PEMS:
        try:
            window = load_pems_corridor(pems_root, district, freeway, direction, first_station=first,
                                        after_ms=as_of_ms // 300_000 * 300_000 - 3_600_000)
        except NoSnapshots:
            sites.append({"site": f"D{district}-{freeway}{direction}-{first}", "status": "no_recent_feed", "advisory": None})
            continue
        matured = np.flatnonzero(window.time_ms + window.cadence_ms <= as_of_ms)
        last = int(matured[-1]) if len(matured) else -1
        inputs = window.inputs("speed", as_of_ms=as_of_ms)[:last + 1]
        recent = np.flatnonzero(np.isfinite(inputs).any(axis=1))
        if not len(recent) or as_of_ms - window.time_ms[recent[-1]] - window.cadence_ms >= 2 * window.cadence_ms:
            sites.append({"site": window.site, "status": "stale_feed", "advisory": None})
            continue
        if not np.isfinite(inputs[0]).any():
            inputs = inputs[recent[0]:]
        level = float(np.nanmedian(inputs[0]))
        decision = outage_triage(inputs, level, 3)
        estimate = graph_reconstruct(inputs, persistence(inputs, level), window.graph, 25.)
        decision["robust_estimate_kmh"] = float(estimate.values[-1, 3])
        entry = {"site": window.site, "status": "current", "interval_ms": int(window.time_ms[last]),
                 "input_sha256": window.raw_sha256, "topology_sha256": window.topology_sha256,
                 "decision": decision, "estimation_status": estimate.status}
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        json_write(output / "incidents" / window.site / f"{entry['interval_ms']}-{digest}.json", entry)
        sites.append(entry)
    return {"as_of_ms": as_of_ms, "sites": sites, "elapsed_seconds": time.perf_counter() - start}


def run_state(
    train: list[Window], calibration: list[tuple[str, Window]], test: list[tuple[str, Window]], *,
    durations: tuple[int | None, ...] = (None,),
) -> dict:
    output = {}
    for channel in ("speed", "flow"):
        benchmark = StateBenchmark(train, channel)
        calibration_rows, errors = benchmark.evaluate(calibration, duration=durations[0])
        control = aggregate([row for row in calibration_rows if row["pattern"].startswith("block")])["methods"]
        comparators = [name for name in control if name != "graph_huber" and control[name]["mae"] is not None]
        reference = min(comparators, key=lambda name: control[name]["mae"]) if comparators else "persistence"
        radius = float(np.quantile(errors, 0.9, method="higher")) if errors else None
        output[channel] = {"calibration_radius": radius, "calibration_target": 0.9,
                           "development_baselines": control, "outages": {}}
        for duration in durations:
            rows, _ = benchmark.evaluate(test, duration=duration, radius=radius)
            block = aggregate([row for row in rows if row["pattern"].startswith("block")], reference)
            random = aggregate([row for row in rows if row["pattern"] == "random"], reference)
            output[channel]["outages"][str(duration)] = {"contiguous": block, "random_mask": random, "rows": rows}
    return output


def run_fixture(output: Path) -> None:
    training = [simulate(seed, "chain").window for seed in DEVELOP]
    calibration = [(str(seed), simulate(seed, "chain").window) for seed in CALIBRATE]
    test = [(str(seed), simulate(seed, "junction").window) for seed in FIXTURE]
    benchmark = run_state(training, calibration, test)
    model = QueueHead(training)
    result = {"mode": "synthetic_fixture", "code_hash": research_code_content_hash(),
              "splits": {"train": DEVELOP, "calibration": CALIBRATE, "evaluation": FIXTURE,
                         "evaluation_layout": "junction", "training_layout": "chain"},
              "state": benchmark, "physics": physics_suite(list(DEVELOP), list(FIXTURE)),
              "storage_scaling": storage_scaling(),
              "queue": evaluate_queues(model, test[0][1]), "routes": route_suite(),
              "replay": replay_case(test[0][1], output / "event-log", 3)}
    json_write(output / "fixture.json", result)


def run_measured(output: Path, pems_root: Path, ngsim_root: Path) -> None:
    loop = measured_pems(pems_root)
    receipts = [replay_case(window, output / "event-log" / window.site, 3) for window in loop]
    periods: dict[str, list[Window]] = {}
    balance = {}
    track_probes = {}
    probe_training = []
    boundary_sources = {}
    tracked_residuals = {}
    for site in ("i-80", "us-101"):
        directory = ngsim_root / site
        if not (directory / "recordings.json").exists():
            continue
        manifest = json.loads((directory / "recordings.json").read_text(encoding="utf-8"))
        periods[site] = []
        for index in range(len(manifest["periods"])):
            tracks, hashes, recording = read_recording(directory, index)
            window, outbound = ngsim_window(tracks, hashes, recording)
            periods[site].append(window)
            boundary_sources[recording] = outbound
            residual = (np.diff(window.values["storage"], axis=0) -
                        (window.values["count"][:-1] - outbound[:-1]))
            tracked_residuals[recording] = residual
            balance[recording] = {"mean_absolute_vehicles": float(np.mean(np.abs(residual))),
                                  "max_absolute_vehicles": float(np.max(np.abs(residual))),
                                  "scope": "tracked crossings against independent tracked storage snapshots"}
            if site == "i-80":
                first, last = [int(f(tracks["global_time"])) for f in (min, max)]
                for start in (first, (first + last - 300_000) // 2, last - 300_000):
                    subset = tracks.loc[tracks["global_time"].between(start, start + 300_000)]
                    probe_training.append(ngsim_window(subset, hashes, recording, cadence_ms=15_000)[0])
            elif index == 0:
                start = int(tracks["global_time"].min()) + 120_000
                subset = tracks.loc[tracks["global_time"].between(start, start + 300_000)].copy()
                track_probes[site] = (ngsim_window(subset, hashes, recording, cadence_ms=15_000)[0], subset)
    analysis = {"mode": "measured_development", "code_hash": research_code_content_hash(),
                "pems": {"replays": receipts, "inputs": [_cohort_signature(w) for w in loop]},
                "ngsim": {"recordings": {site: [_cohort_signature(w) for w in windows]
                                         for site, windows in periods.items()}, "tracked_balance": balance},
                "submissions": {
                    "ite": "prototype_demo" if any(r["congestion"] is not None for r in receipts) else "outage_only",
                    "sigmod": "prototype_demo" if all(r["matches_full_recompute"] and r["recomputed"] < r["window_count"]
                                                      for r in receipts) else "needs_incremental_parity",
                    "birmingham": "needs_independent_measured_shadow_pilot",
                    "ieee_iv_workshop": "needs_matching_call_and_vehicle_evidence",
                    "itsc": "needs_frozen_measured_evaluation",
                    "kdd": "needs_confirmed_cycle_and_release_permissions",
                }}
    if len(periods.get("i-80", [])) >= 2 and periods.get("us-101"):
        training, calibration = periods["i-80"][:-1], periods["i-80"][-1]
        test = [(w.site, w) for w in periods["us-101"]]
        analysis["ngsim"]["cross_site_state"] = run_state(training, [(calibration.site, calibration)], test)
        analysis["ngsim"]["queue"] = evaluate_queues(QueueHead(training), test[0][1])
        destination, trajectories = track_probes["us-101"]
        analysis["ngsim"]["trajectory_advisory"] = evaluate_queues(
            QueueHead(probe_training, horizon=2), destination, tracks=trajectories, direction="US-101 southbound",
        )
        reference = np.median(np.concatenate([
            np.c_[w.values["count"], boundary_sources[w.site][:, -1]] for w in training
        ]), axis=0)
        discrepancy_cap = np.quantile(np.abs(np.concatenate([tracked_residuals[w.site] for w in training])),
                                      0.9, method="higher")
        measured_fit = []
        for _, window in test:
            truth = window.values["storage"]
            steps, cells = truth.shape
            incidence = np.zeros((cells, cells + 1))
            incidence[np.arange(cells), np.arange(cells)] = 1
            incidence[np.arange(cells), np.arange(1, cells + 1)] = -1
            cadence = window.cadence_ms / 3_600_000
            hidden = outage_masks(truth.shape)[1]
            observations = window.inputs("storage", hidden=hidden)
            # This conditioning experiment knows the inlet/outlet counters but masks interior storage.
            counts = np.c_[window.values["count"][:, 0], boundary_sources[window.site][:, -1]]
            upper = np.full((steps, cells + 1), 18_000.)
            lower = np.zeros_like(upper)
            lower[:, [0, -1]] = np.maximum(counts - 2, 0) / cadence
            upper[:, [0, -1]] = (counts + 2) / cadence
            signed_discrepancy = np.c_[np.eye(cells), -np.eye(cells)]
            augmented = (np.c_[incidence, signed_discrepancy], np.c_[lower, np.zeros((steps, 2 * cells))],
                         np.c_[upper, np.full((steps, 2 * cells), discrepancy_cap / cadence)],
                         np.r_[reference / cadence, np.zeros(2 * cells)])
            storage_level = historical_level(training, "storage")
            for mode, (graph, lo, hi, prior) in (
                ("strict_tracked_chain", (incidence, lower, upper, reference / cadence)),
                ("bounded_track_discrepancy", augmented),
            ):
                estimate, bound, fallback = fit_storage(graph, truth[0], np.full(cells, 240.), lo, hi,
                                                        observations, prior, cadence, 0, storage_level=storage_level)
                point = fallback if estimate.values is None else estimate.values[:(steps + 1) * cells].reshape(-1, cells)[:-1]
                evidence = "fallback" if estimate.values is None else "conditional" if bound.status == "optimal" else "degraded"
                measured_fit.append({"site": window.site, "mode": mode, "status": estimate.status, "bounds": asdict(bound),
                                     "solver_message": estimate.solver_message, "point_evidence": evidence,
                                     "track_discrepancy_cap_per_cell": float(discrepancy_cap) if mode == "bounded_track_discrepancy" else 0.,
                                     "missing_storage_mae": float(np.mean(np.abs(point[hidden] - truth[hidden]))),
                                     "persistence_mae": float(np.mean(np.abs(fallback[hidden] - truth[hidden]))),
                                     "balance_solver_residual": estimate.balance_error,
                                     "scope": "tracked accounting development test, not true external vehicle sources"})
        analysis["ngsim"]["cross_site_storage"] = measured_fit
        analysis["benchmark_profile"] = {}
        cohorts = {"synthetic": [simulate(seed).window for seed in FIXTURE], "pems": loop,
                   "ngsim": [w for windows in periods.values() for w in windows]}
        for name, windows in cohorts.items():
            profile = {"cadence_seconds": sorted({w.cadence_ms / 1000 for w in windows})}
            for channel in ("speed", "flow"):
                measurements = np.concatenate([w.inputs(channel).ravel() for w in windows])
                finite = np.isfinite(measurements)
                profile[channel] = {"scored": int(finite.sum()), "grid_cells": len(measurements),
                                    "percentiles_10_50_90": np.quantile(measurements[finite], (0.1, 0.5, 0.9)).tolist()}
            analysis["benchmark_profile"][name] = profile
    json_write(output / "measured-development.json", analysis)


def _select_cohorts(config: dict, pems_root: Path) -> dict[str, list[tuple[str, Window]]]:
    role_specs = config["cohorts"]
    if set(role_specs) != {"train", "calibration", "evaluation"}:
        raise ValueError("Declare exactly train, calibration and evaluation cohorts")
    result = {}
    zone = ZoneInfo("America/Los_Angeles")
    for role, specs in role_specs.items():
        windows = []
        keys = set()
        for row in specs:
            if not row["days"] or len(set(row["days"])) != len(row["days"]):
                raise ValueError("Declare a nonempty list of unique local calendar days")
            for day in row["days"]:
                start = datetime.fromisoformat(day).replace(tzinfo=zone)
                stop = start + timedelta(days=1)
                window = load_pems_corridor(pems_root, row["district"], row["freeway"], row["direction"],
                                            first_station=row["first_station"], after_ms=int(start.timestamp() * 1000),
                                            before_ms=int(stop.timestamp() * 1000))
                if np.isfinite(window.inputs("speed")).sum() < 100 or np.isfinite(window.inputs("flow")).sum() < 100:
                    raise ValueError(f"Insufficient measured coverage for {window.site} on {day}")
                window_keys = {(station, day) for station in window.cell_ids}
                if keys & window_keys:
                    raise ValueError(f"Duplicate station-day in {role} on {day}")
                keys.update(window_keys)
                windows.append((day, window))
        if not windows:
            raise ValueError(f"Declare at least one {role} cohort")
        result[role] = windows
    days = {role: {key for key, _ in rows} for role, rows in result.items()}
    if not max(days["train"]) < min(days["calibration"]) <= max(days["calibration"]) < min(days["evaluation"]):
        raise ValueError("Development, calibration and evaluation days must be disjoint and ordered")
    development = {station for role in ("train", "calibration") for _, w in result[role] for station in w.cell_ids}
    reserve = {station for _, w in result["evaluation"] for station in w.cell_ids}
    if development & reserve or len(days["evaluation"]) < 10:
        raise ValueError("Require held-out sites and at least ten independent evaluation days")
    if any(not np.isfinite(config[key]) or config[key] <= 0 for key in
           ("flow_noninferiority_veh_per_hour", "random_mask_noninferiority_kmh")):
        raise ValueError("Declare positive, development-derived noninferiority margins")
    return result


def reserve_evaluation(root: Path, station_days: set[tuple[str, str]], manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    registry = root / "registry.json"
    pairs = sorted(station_days)
    run_id = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    with registry_file_lock(registry):
        for path in root.glob("*/started.json"):
            attempt = json.loads(path.read_text(encoding="utf-8"))
            if station_days & {tuple(pair) for pair in attempt["station_days"]}:
                raise ValueError("An evaluation station-day was already reserved")
        attempt = root / run_id
        attempt.mkdir()
        json_write(attempt / "started.json", {"manifest": manifest, "station_days": pairs,
                                               "started_at": datetime.now(UTC).isoformat()})
    return attempt


def freeze_run(config_path: Path, pems_root: Path, receipt: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohorts = _select_cohorts(config, pems_root)
    manifest = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "code_hash": research_code_content_hash(),
                "cohorts": {role: [_cohort_signature(w) for _, w in rows] for role, rows in cohorts.items()}}
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


def run_final(config_path: Path, pems_root: Path, receipt: Path, output: Path) -> None:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    cohorts = _select_cohorts(config, pems_root)
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    supplied = {role: [_cohort_signature(w) for _, w in rows] for role, rows in cohorts.items()}
    if (manifest["code_hash"] != research_code_content_hash()
            or manifest["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or manifest["cohorts"] != json.loads(json.dumps(supplied))):
        raise ValueError("The frozen code, config or measured input changed")
    station_days = {(station, day) for day, window in cohorts["evaluation"] for station in window.cell_ids}
    registry = reserve_evaluation(pems_root / "final-runs", station_days, manifest)
    run_id = registry.name
    started = time.perf_counter()
    results = run_state([w for _, w in cohorts["train"]], cohorts["calibration"], cohorts["evaluation"],
                        durations=(6, 12))
    intervals = [results["speed"]["outages"][str(width)]["contiguous"]["graph_huber_minus_reference_95ci"]
                 for width in (6, 12)]
    flow = [results["flow"]["outages"][str(width)]["contiguous"]["graph_huber_minus_reference_95ci"] for width in (6, 12)]
    normal = [results["speed"]["outages"][str(width)]["random_mask"]["graph_huber_minus_reference_95ci"] for width in (6, 12)]
    speed_gate = all(interval is not None and interval[1] < 0 for interval in intervals)
    flow_gate = all(interval is not None and interval[1] <= config["flow_noninferiority_veh_per_hour"] for interval in flow)
    normal_gate = all(interval is not None and interval[1] <= config["random_mask_noninferiority_kmh"] for interval in normal)
    result = {"mode": "measured_final", "run_id": run_id, "frozen": manifest, "state": results,
              "speed_gate": speed_gate, "flow_gate": flow_gate, "random_mask_gate": normal_gate,
              "elapsed_seconds": time.perf_counter() - started}
    json_write(registry / "completed.json", result)
    json_write(output / "frozen-evaluation.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pems", type=Path, default=Path("data/crest/pems"))
    parser.add_argument("--ngsim", type=Path, default=Path("data/crest/ngsim"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/crest"))
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("fixture")
    sub.add_parser("measured")
    sub.add_parser("export-benchmark")
    score = sub.add_parser("score-benchmark")
    score.add_argument("--scenario", required=True)
    score.add_argument("--prediction", type=Path, required=True)
    shadow = sub.add_parser("shadow")
    shadow.add_argument("--poll-seconds", type=int, default=0)
    for label in ("freeze", "final"):
        action = sub.add_parser(label)
        action.add_argument("--config", type=Path, required=True)
        action.add_argument("--receipt", type=Path, required=True)
    inspect = sub.add_parser("replay")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--site-index", type=int, choices=(0, 1), default=0)
    inspect.add_argument("--correct", nargs=3, metavar=("TIME_MS", "CELL", "SPEED_KMH_OR_MISSING"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.mode == "fixture":
            run_fixture(args.output)
        elif args.mode == "measured":
            run_measured(args.output, args.pems, args.ngsim)
        elif args.mode == "export-benchmark":
            export_suite(args.output / "benchmark", {"train": DEVELOP, "calibration": CALIBRATE, "evaluation": FIXTURE})
        elif args.mode == "score-benchmark":
            print(json.dumps(score_submission(args.output / "benchmark", args.scenario, args.prediction), indent=2))
        elif args.mode == "shadow":
            if args.poll_seconds and args.poll_seconds < 60:
                parser.error("Use at least 60 seconds between shadow runs")
            while True:
                cycle = shadow_cycle(args.pems, args.output, int(time.time() * 1000))
                json_write(args.output / "shadow.json", cycle)
                if not args.poll_seconds:
                    break
                time.sleep(args.poll_seconds)
        elif args.mode == "freeze":
            freeze_run(args.config, args.pems, args.receipt)
        elif args.mode == "final":
            run_final(args.config, args.pems, args.receipt, args.output)
        else:
            district, freeway, direction, first = PEMS[args.site_index]
            site = f"D{district}-{freeway}{direction}-{first}"
            replay = Replay.from_manifest(args.output / "event-log" / site, args.manifest)
            manifest = args.manifest
            if args.correct:
                stamp, cell, speed = args.correct
                manifest = replay.correct(manifest, int(stamp), int(cell), None if speed == "missing" else float(speed))
            print(json.dumps(replay.run(manifest), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
