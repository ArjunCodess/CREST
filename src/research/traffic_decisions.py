"""Evaluate causal queue and route decisions with explicit evidence status."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.od_v6 import solve_weighted_od

from .baselines import od_lp_bounds_baseline
from .observations import Window
from .traffic_estimation import QueueHead, congested, outage_masks
from .traffic_replay import queue_advisory


def evaluate_queues(
    model: QueueHead, window: Window, *, processing_lag_ms: int = 0, seed: int = 19,
    probe_penetration: float = 0.35, communication_delay_ms: int | None = None,
    tracks: pd.DataFrame | None = None, direction: str = "along-corridor",
) -> dict:
    """Score fixed-duration forecast episodes and simulated moving probes."""
    speed = window.inputs("speed")
    truth = congested(speed, model.threshold)
    mask = outage_masks(speed.shape)[1]
    delay = window.cadence_ms // 2 if communication_delay_ms is None else communication_delay_ms
    random = np.random.default_rng(seed)
    outputs = {name: Counter() for name in ("gated_infrastructure", "ungated_infrastructure", "probe_speed")}
    leads: dict[str, list[float]] = {name: [] for name in outputs}
    event_rows = []
    probe_at = {}
    if tracks is not None:
        samples = tracks.loc[tracks["global_time"].isin(window.time_ms + window.cadence_ms),
                             ["global_time", "vehicle_id", "local_y", "v_vel"]]
        for stamp, batch in samples.drop_duplicates(["global_time", "vehicle_id"]).groupby("global_time"):
            probe_at[int(stamp)] = batch.set_index("vehicle_id")
    for index in range(2, len(speed) - model.horizon):
        future = index + model.horizon
        prediction = model.predict(window, index, mask, processing_lag_ms=processing_lag_ms)
        now = prediction.origin_ms + delay
        future_available = np.isfinite(speed[future])
        future_queue = truth[future]
        previous_queue = truth[future - 1]
        peak = float(prediction.probability.max())
        advisory = queue_advisory(prediction, window.positions_m, now_ms=now, cadence_ms=window.cadence_ms,
                                  direction=direction)
        event_rows.append({
            "origin_ms": prediction.origin_ms, "eligible_cells": int(future_available.sum()),
            "true_queue_cells": int(future_queue.sum()), "peak_queue_probability": peak,
            "true_onset_cells": int(np.sum(future_queue & ~previous_queue)),
            "forecast_onset_cells": int(np.sum(prediction.onset_probability >= 0.6)),
            "advisory_reason": advisory.reason,
        })
        if tracks is None:
            trips = [(cell - 1, cell, speed[index, cell - 1], window.positions_m[cell - 1])
                     for cell in range(1, speed.shape[1])]
        else:
            positions = window.positions_m
            end = positions[-1] + (positions[1] - positions[0])
            before = probe_at.get(int(window.time_ms[index] + window.cadence_ms))
            after = probe_at.get(int(window.time_ms[future] + window.cadence_ms))
            trips = []
            if before is not None and after is not None:
                for record in before.join(after[["local_y"]], how="inner", rsuffix="_future").itertuples():
                    p0, p1 = record.local_y * .3048, record.local_y_future * .3048
                    if positions[0] <= p0 <= p1 < end:
                        trips.append((int(np.searchsorted(positions, p0, side="right") - 1),
                                      int(np.searchsorted(positions, p1, side="right") - 1),
                                      record.v_vel * 1.09728, p0))
        for first, cell, probe_speed, position in trips:
            if not future_available[cell] or not np.isfinite(speed[index, first]):
                continue
            encounter = future_queue[cell] and not truth[index, first]
            probe_reports = random.random() < probe_penetration and np.isfinite(probe_speed) and probe_speed >= 0
            horizon_seconds = max(0, int(window.time_ms[future] + window.cadence_ms) - prediction.origin_ms) / 1000
            position_usable = np.isfinite(probe_speed) and probe_speed >= 0
            projected = position + probe_speed / 3.6 * horizon_seconds if position_usable else position
            positions = window.positions_m
            endpoint = np.searchsorted(positions, projected, side="right")
            covers = advisory.available and position_usable and probe_speed >= model.threshold and any(
                position < stop and projected >= start for start, stop in advisory.queue_intervals_m
            )
            decisions = {
                "gated_infrastructure": (bool(covers), not position_usable or advisory.reason in (
                    "stale_or_wide_location", "expired_or_future_observation")),
                "ungated_infrastructure": (
                    bool(position_usable and np.any(prediction.probability[first:endpoint] >= 0.60)),
                    not position_usable,
                ),
                "probe_speed": (bool(probe_reports and probe_speed < model.threshold), not probe_reports),
            }
            for name, (alarm, unavailable) in decisions.items():
                score = outputs[name]
                score["trips"] += 1
                score["encounters"] += bool(encounter)
                score["withheld"] += unavailable
                score["false_warnings"] += bool(alarm and not encounter)
                score["detected_encounters"] += bool(alarm and encounter)
                if alarm and encounter:
                    leads[name].append(max(0, int(window.time_ms[future] + window.cadence_ms) - now) / 1000)
    summary = {}
    for name, counts in outputs.items():
        summary[name] = {key: counts[key] for key in
                         ("trips", "encounters", "withheld", "false_warnings", "detected_encounters")}
        summary[name]["false_warnings_per_trip"] = counts["false_warnings"] / counts["trips"] if counts["trips"] else None
        summary[name]["median_lead_seconds"] = float(np.median(leads[name])) if leads[name] else None
    return {"site": window.site, "source": window.source,
            "probe_model": "sampled_ngsim_tracks" if tracks is not None else "simulated_one_cell_advance",
            "warning_position_basis": "origin position and speed projected to the forecast horizon",
            "probe_penetration": probe_penetration, "communication_delay_ms": delay,
            "horizon_steps": model.horizon, "event_rows": event_rows, "methods": summary,
            "scope": "detector_proxy" if window.source == "PeMS" else "trajectory_or_simulator_speed_proxy"}


def route_suite() -> dict:
    """A common detector sees two routes. A second detector resolves one branch."""
    incidence = np.array([[1., 1.], [1., 0.], [0., 1.]])
    truth = np.array([70., 30.])
    counts = incidence @ truth
    prior = np.array([45., 45.])
    output = []
    for mask in (np.array([True, False, False]), np.array([True, True, False])):
        case = od_lp_bounds_baseline(incidence, counts, counts, np.array([1., 0.]), prior, observed_mask=mask)
        flows, _ = solve_weighted_od(incidence, np.where(mask, counts, np.nan), prior, regularization=0.01)
        output.append({"observed_rows": np.flatnonzero(mask).tolist(), "nnls": case["flows_point"].tolist(),
                       "dual": flows.tolist(), "route_0_bounds": case["aggregate_bounds"]})
    return {"evidence": "synthetic", "truth": truth.tolist(), "cases": output}
