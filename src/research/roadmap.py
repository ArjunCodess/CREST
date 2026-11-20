"""CREST outage replay, measured-data pilots, and frozen benchmark runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .observations import NoSnapshots, Window, load_pems_corridor, ngsim_window, read_recording
from .pems import CADENCE_MS, PEMS_TIMEZONE, atomic_write
from .protocol import registry_file_lock
from .provenance import research_code_content_hash
from .traffic_ablation import matched_mask_controls, storage_ablation, summarize_mask_controls, wave_ablation
from .traffic_benchmark import StateBenchmark, aggregate, benchmark_masks, physics_suite, storage_scaling
from .traffic_decisions import evaluate_queues, route_suite
from .traffic_estimation import QueueHead, fit_storage, graph_reconstruct, historical_level, outage_masks, persistence
from .traffic_exchange import evaluate_export, export_suite, export_trajectory_suite, score_submission
from .traffic_replay import Replay, outage_triage, replay_case
from .traffic_simulator import simulate
from .traffic_validation import (
    detector_queue_transfer, episode_profile, operational_outages, replay_benchmark, trajectory_advisory_sweep,
)

DEVELOP = (11, 12, 13, 14)
CALIBRATE = (101, 102, 103)
FIXTURE = (241, 242, 243, 244)
PEMS = ((4, 880, "S", "401557"), (7, 10, "W", "717258"))
FINAL_DURATIONS = (6, 12)
MIN_EVALUATION_DAYS = 10


def json_write(path: Path, payload: dict) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def measured_pems(root: Path, *, available_by_ms: int | None = None) -> list[Window]:
    return [load_pems_corridor(root, d, f, direction, first_station=first, available_by_ms=available_by_ms)
            for d, f, direction, first in PEMS]


def _cohort_signature(window: Window) -> dict:
    return {"source": window.source, "site": window.site, "start_ms": int(window.time_ms[0]),
            "end_ms": int(window.time_ms[-1] + window.cadence_ms), "topology": window.topology_sha256,
            "source_hashes": window.raw_sha256}


def shadow_cycle(pems_root: Path, output: Path, as_of_ms: int, *, code_hash: str | None = None) -> dict:
    """Publish a read-only operator snapshot and idempotent per-window decisions."""
    start = time.perf_counter()
    sites = []
    for district, freeway, direction, first in PEMS:
        try:
            window = load_pems_corridor(pems_root, district, freeway, direction, first_station=first,
                                        after_ms=as_of_ms // CADENCE_MS * CADENCE_MS - 3_600_000,
                                        available_by_ms=as_of_ms)
        except NoSnapshots:
            sites.append({"site": f"D{district}-{freeway}{direction}-{first}", "status": "no_recent_feed", "advisory": None})
            continue
        age_ms = as_of_ms - int(window.time_ms[-1]) - window.cadence_ms
        entry = {"site": window.site, "interval_ms": int(window.time_ms[-1]), "age_ms": age_ms,
                 "capture_lag_ms": int(window.arrival_ms[-1]) - int(window.time_ms[-1]) - window.cadence_ms,
                 "input_sha256": window.raw_sha256, "topology_sha256": window.topology_sha256}
        if age_ms >= 2 * window.cadence_ms:
            sites.append({**entry, "status": "stale_feed", "advisory": None})
            continue
        inputs = window.inputs("speed")
        usable = np.isfinite(inputs).any(axis=1)
        if not usable[-1]:
            sites.append({**entry, "status": "no_usable_readings", "advisory": None})
            continue
        inputs = inputs[np.argmax(usable):]
        level = float(np.nanmedian(inputs[0]))
        decision = outage_triage(inputs, level, 3)
        estimate = graph_reconstruct(inputs, persistence(inputs, level), window.graph, 25.)
        decision["robust_estimate_kmh"] = float(estimate.values[-1, 3])
        entry.update(status="current", decision=decision, estimation_status=estimate.status)
        incident = {key: value for key, value in entry.items() if key != "age_ms"}
        digest = hashlib.sha256(json.dumps(incident, sort_keys=True).encode()).hexdigest()
        json_write(output / "incidents" / window.site / f"{entry['interval_ms']}-{digest}.json", incident)
        sites.append(entry)
    cycle = {"as_of_ms": as_of_ms, "code_hash": code_hash or research_code_content_hash(),
             "sites": sites, "elapsed_seconds": time.perf_counter() - start}
    json_write(output / "shadow-history" / f"{as_of_ms}.json", cycle)
    return cycle


def summarize_shadow(output: Path) -> dict:
    zone = ZoneInfo(PEMS_TIMEZONE)
    sites = {}
    codes = set()
    cycle_count = 0
    elapsed = []
    for path in sorted((output / "shadow-history").glob("*.json")):
        cycle = json.loads(path.read_text(encoding="utf-8"))
        cycle_count += 1
        codes.add(cycle["code_hash"])
        if "elapsed_seconds" in cycle:
            elapsed.append(cycle["elapsed_seconds"])
        day = datetime.fromtimestamp(cycle["as_of_ms"] / 1000, zone).date().isoformat()
        for entry in cycle["sites"]:
            daily = sites.setdefault(entry["site"], {}).setdefault(day, {"statuses": {}, "windows": {}})
            status = entry["status"]
            daily["statuses"][status] = daily["statuses"].get(status, 0) + 1
            if status == "current":
                daily["windows"].setdefault(entry["interval_ms"], entry["capture_lag_ms"])
    for days in sites.values():
        for daily in days.values():
            latency = list(daily.pop("windows").values())
            daily["distinct_current_windows"] = len(latency)
            daily["capture_lag_p95_ms"] = float(np.quantile(latency, .95)) if latency else None
    return {"scope": "prospective_shadow_cycles", "cycles": cycle_count, "code_hashes": sorted(codes), "sites": sites,
            "cycle_timing": {"samples": len(elapsed),
                             "p95_seconds": float(np.quantile(elapsed, .95)) if elapsed else None}}


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
            profile = {"cadence_seconds": sorted({w.cadence_ms / 1000 for w in windows}), **episode_profile(windows)}
            for channel in ("speed", "flow"):
                measurements = np.concatenate([w.inputs(channel).ravel() for w in windows])
                finite = np.isfinite(measurements)
                profile[channel] = {"scored": int(finite.sum()), "grid_cells": len(measurements),
                                    "percentiles_10_50_90": np.quantile(measurements[finite], (0.1, 0.5, 0.9)).tolist()}
            analysis["benchmark_profile"][name] = profile
    json_write(output / "measured-development.json", analysis)


def inspect_cohorts(config: dict, pems_root: Path) -> tuple[dict[str, list[tuple[str, Window]]], dict]:
    role_specs = config["cohorts"]
    if set(role_specs) != {"train", "calibration", "evaluation"}:
        raise ValueError("Declare exactly train, calibration and evaluation cohorts")
    result = {}
    zone = ZoneInfo(PEMS_TIMEZONE)
    today = datetime.now(zone).date()
    days = {role: set() for role in role_specs}
    stations = {role: set() for role in role_specs}
    entries = []
    for role, specs in role_specs.items():
        windows = []
        keys = set()
        for row in specs:
            if not row["days"] or len(set(row["days"])) != len(row["days"]):
                raise ValueError("Declare a nonempty list of unique local calendar days")
            for day in row["days"]:
                local_day = date.fromisoformat(day)
                if local_day.isoformat() != day:
                    raise ValueError("Use YYYY-MM-DD for cohort days")
                days[role].add(day)
                entry = {"role": role, "day": day, "site": f"D{row['district']}-{row['freeway']}"
                         f"{row['direction']}-{row['first_station']}"}
                entries.append(entry)
                if local_day >= today:
                    entry["status"] = "day_not_complete"
                    continue
                start = datetime.combine(local_day, datetime.min.time(), tzinfo=zone)
                stop = start + timedelta(days=1)
                try:
                    window = load_pems_corridor(pems_root, row["district"], row["freeway"], row["direction"],
                                                first_station=row["first_station"], after_ms=int(start.timestamp() * 1000),
                                                before_ms=int(stop.timestamp() * 1000))
                except NoSnapshots:
                    entry["status"] = "no_snapshots"
                    continue
                window_keys = {(station, day) for station in window.cell_ids}
                if keys & window_keys:
                    raise ValueError(f"Duplicate station-day in {role} on {day}")
                keys.update(window_keys)
                stations[role].update(window.cell_ids)
                visible = {channel: np.isfinite(window.inputs(channel)) for channel in ("speed", "flow")}
                entry["observed_cells"] = {channel: int(mask.sum()) for channel, mask in visible.items()}
                if min(entry["observed_cells"].values()) < 100:
                    entry["status"] = "insufficient_observations"
                    continue
                if role != "train":
                    durations = FINAL_DURATIONS if role == "evaluation" else FINAL_DURATIONS[:1]
                    entry["unscorable_masks"] = []
                    for duration in durations:
                        *blocks, (_, random) = benchmark_masks(window.quality.shape, duration)
                        for channel, mask in visible.items():
                            if not any(mask[block].all() for _, block in blocks):
                                entry["unscorable_masks"].append(f"{channel}:{duration}:contiguous")
                            if role == "evaluation" and not (mask & random).any():
                                entry["unscorable_masks"].append(f"{channel}:{duration}:random")
                    if entry["unscorable_masks"]:
                        entry["status"] = "insufficient_mask_truth"
                        continue
                entry["status"] = "ready"
                windows.append((day, window))
        if not days[role]:
            raise ValueError(f"Declare at least one {role} cohort")
        result[role] = windows
    if not max(days["train"]) < min(days["calibration"]) <= max(days["calibration"]) < min(days["evaluation"]):
        raise ValueError("Development, calibration and evaluation days must be disjoint and ordered")
    if (stations["train"] | stations["calibration"]) & stations["evaluation"]:
        raise ValueError("Require held-out evaluation stations")
    if any(not np.isfinite(config[key]) or config[key] <= 0 for key in
           ("flow_noninferiority_veh_per_hour", "random_mask_noninferiority_kmh")):
        raise ValueError("Declare positive, development-derived noninferiority margins")
    ready_days = len({day for day, _ in result["evaluation"]})
    return result, {"ready": ready_days >= MIN_EVALUATION_DAYS and all(row["status"] == "ready" for row in entries),
                    "ready_evaluation_days": ready_days, "required_evaluation_days": MIN_EVALUATION_DAYS,
                    "cohort_days": entries}


def _select_cohorts(config: dict, pems_root: Path) -> dict[str, list[tuple[str, Window]]]:
    cohorts, readiness = inspect_cohorts(config, pems_root)
    if not readiness["ready"]:
        raise ValueError("Cohorts are incomplete or lack ten usable evaluation days. Run readiness for details.")
    return cohorts


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
                        durations=FINAL_DURATIONS)
    gates = {}
    for name, channel, mask, margin in (
        ("speed_gate", "speed", "contiguous", 0.0),
        ("flow_gate", "flow", "contiguous", config["flow_noninferiority_veh_per_hour"]),
        ("random_mask_gate", "speed", "random_mask", config["random_mask_noninferiority_kmh"]),
    ):
        groups = [results[channel]["outages"][str(width)][mask] for width in FINAL_DURATIONS]
        gates[name] = all(
            group["paired_clusters"] >= MIN_EVALUATION_DAYS
            and group["graph_huber_minus_reference_95ci"] is not None
            and (group["graph_huber_minus_reference_95ci"][1] < 0 if name == "speed_gate"
                 else group["graph_huber_minus_reference_95ci"][1] <= margin)
            for group in groups
        )
    result = {"mode": "measured_final", "run_id": run_id, "frozen": manifest, "state": results,
              **gates,
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
    operational = sub.add_parser("operational-validation")
    operational.add_argument("--available-by-ms", type=int, help="Freeze archive visibility at a UTC Unix timestamp")
    sub.add_parser("ablate")
    sub.add_parser("demand-transfer")
    sub.add_parser("advisory-sweep")
    export = sub.add_parser("export-benchmark")
    export.add_argument("--source", choices=("synthetic", "ngsim"), default="synthetic")
    verify = sub.add_parser("verify-benchmark")
    verify.add_argument("--source", choices=("synthetic", "ngsim"), default="synthetic")
    score = sub.add_parser("score-benchmark")
    score.add_argument("--source", choices=("synthetic", "ngsim"), default="synthetic")
    score.add_argument("--scenario", required=True)
    score.add_argument("--prediction", type=Path, required=True)
    shadow = sub.add_parser("shadow")
    shadow.add_argument("--poll-seconds", type=int, default=0)
    sub.add_parser("shadow-report")
    readiness = sub.add_parser("readiness")
    readiness.add_argument("--config", type=Path, required=True)
    for label in ("freeze", "final"):
        action = sub.add_parser(label)
        action.add_argument("--config", type=Path, required=True)
        action.add_argument("--receipt", type=Path, required=True)
    inspect = sub.add_parser("replay")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--site-index", type=int, choices=(0, 1), default=0)
    inspect.add_argument("--log-dir", type=Path, help="Open an explicit replay directory, including a long replay")
    inspect.add_argument("--correct", nargs=3, metavar=("TIME_MS", "CELL", "SPEED_KMH_OR_MISSING"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.mode == "fixture":
            run_fixture(args.output)
        elif args.mode == "measured":
            run_measured(args.output, args.pems, args.ngsim)
        elif args.mode == "operational-validation":
            windows = measured_pems(args.pems, available_by_ms=args.available_by_ms)
            json_write(args.output / "operational-validation.json", {
                "scope": "retrospective_measured_development_and_synthetic_replay",
                "code_hash": research_code_content_hash(),
                "available_by_ms": args.available_by_ms,
                "inputs": [_cohort_signature(w) for w in windows],
                "outages": [row for window in windows for row in operational_outages(window)],
                "detector_queue_transfer": detector_queue_transfer(windows),
                "replay": replay_benchmark(windows, args.output / "long-event-log"),
            })
        elif args.mode == "ablate":
            development = [simulate(seed, "chain").window for seed in DEVELOP]
            evaluation = [(seed, simulate(seed, "junction").window) for seed in FIXTURE]
            rows = matched_mask_controls(development, evaluation)
            output = {
                "scope": "development_only", "code_hash": research_code_content_hash(),
                "splits": {"training_seeds": DEVELOP, "synthetic_evaluation": FIXTURE},
                "synthetic_wave": wave_ablation(development, [window for _, window in evaluation]),
                "storage": storage_ablation(DEVELOP, FIXTURE),
                "mask_controls": {"rows": rows, "summary": summarize_mask_controls(rows)},
            }
            recordings = []
            for site in ("i-80", "us-101"):
                directory = args.ngsim / site
                if not (directory / "recordings.json").exists():
                    break
                tracks, digests, name = read_recording(directory, 0)
                recordings.append(ngsim_window(tracks, digests, name)[0])
            if len(recordings) == 2:
                output["measured_wave"] = wave_ablation(recordings[:1], recordings[1:])
                output["measured_inputs"] = [_cohort_signature(window) for window in recordings]
            json_write(args.output / "ablations.json", output)
        elif args.mode == "demand-transfer":
            training = [simulate(seed, "chain").window for seed in DEVELOP]
            calibration = [(str(seed), simulate(seed, "chain").window) for seed in CALIBRATE]
            results = {}
            for regime, seeds in (("light", (1201, 1202, 1203, 1204)), ("surge", (1301, 1302, 1303, 1304))):
                evaluation = [(str(seed), simulate(seed, "junction", demand_regime=regime).window) for seed in seeds]
                results[regime] = {"seeds": seeds, "state": run_state(training, calibration, evaluation)}
            json_write(args.output / "demand-transfer.json", {
                "scope": "synthetic_topology_and_demand_transfer", "code_hash": research_code_content_hash(),
                "training_seeds": DEVELOP, "calibration_seeds": CALIBRATE, "evaluation": results,
            })
        elif args.mode == "advisory-sweep":
            json_write(args.output / "advisory-sweep.json", {
                "code_hash": research_code_content_hash(), **trajectory_advisory_sweep(args.ngsim),
            })
        elif args.mode == "export-benchmark":
            if args.source == "ngsim":
                export_trajectory_suite(args.output / "benchmark-ngsim", args.ngsim)
            else:
                export_suite(args.output / "benchmark", {"train": DEVELOP, "calibration": CALIBRATE, "evaluation": FIXTURE})
        elif args.mode == "verify-benchmark":
            folder = "benchmark-ngsim" if args.source == "ngsim" else "benchmark"
            json_write(args.output / f"{folder}-baselines.json", {
                "scope": f"{args.source}_export", "code_hash": research_code_content_hash(),
                "scores": evaluate_export(args.output / folder),
            })
        elif args.mode == "score-benchmark":
            folder = "benchmark-ngsim" if args.source == "ngsim" else "benchmark"
            print(json.dumps(score_submission(args.output / folder, args.scenario, args.prediction), indent=2))
        elif args.mode == "readiness":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            _, report = inspect_cohorts(config, args.pems)
            print(json.dumps(report, indent=2))
        elif args.mode == "shadow-report":
            report = summarize_shadow(args.output)
            json_write(args.output / "shadow-report.json", report)
            print(json.dumps(report, indent=2))
        elif args.mode == "shadow":
            if args.poll_seconds and args.poll_seconds < 60:
                parser.error("Use at least 60 seconds between shadow runs")
            code_hash = research_code_content_hash()
            while True:
                cycle = shadow_cycle(args.pems, args.output, int(time.time() * 1000), code_hash=code_hash)
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
            replay = Replay.from_manifest(args.log_dir or args.output / "event-log" / site, args.manifest)
            manifest = args.manifest
            if args.correct:
                stamp, cell, speed = args.correct
                manifest = replay.correct(manifest, int(stamp), int(cell), None if speed == "missing" else float(speed))
            print(json.dumps(replay.run(manifest), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
