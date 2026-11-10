"""Refit frozen queue policies before a specified training-month assessment."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state, graph
from src.queue_events import (CONDITIONS, EventQueueModel, collect_events, load_auxiliary,
    visible_event, calibrate_occupancy_density)
from src.queue_model import QueueClassifier, QueueSpeedRegressor, onset_slowdown
from src.evaluation import proxy_queue_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--promotion", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--assessment-month", default="2030-12")
    ap.add_argument("--panel", action="append")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    decisions = json.loads(args.promotion.read_text())["decisions"]
    assessment_start = pd.Timestamp(args.assessment_month + "-01T00:00:00Z")
    assessment_end = assessment_start + pd.offsets.MonthBegin(1)
    fit_end = assessment_start - pd.Timedelta(minutes=90)
    manifest = {"policy_selected_on": "October", "assessment_month": args.assessment_month,
        "fit_end_exclusive": str(fit_end), "assessment_end_exclusive": str(assessment_end),
        "source_sha256": __import__("hashlib").sha256(Path("src/queue_events.py").read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary, rows = [], []
    for decision in decisions:
        if not decision["candidate_for_confirmation"] or (args.panel and decision["panel"] not in args.panel):
            continue
        started = time.monotonic(); panel = decision["panel"]
        source = Path(decision["source_root"])
        with (source / f"{panel}_event.pkl").open("rb") as f:
            selected = pickle.load(f)["model"]
        old = next(r for r in json.loads((source / "pilot_summary.json").read_text()) if r["panel"] == panel)
        print(f"Confirm {panel}: frozen origin={selected.policy}, decoder={selected.decoders}", flush=True)
        d = load_state(args.release_root, panel, "train", True,
            start="2030-06-01T00:00:00Z", end=str(assessment_end))
        auxiliary = load_auxiliary(args.release_root, panel, d.times, d.links)
        up, down, _, params = graph(args.release_root, panel, d.links)
        fit = np.asarray(d.times < fit_end)
        test = np.asarray(d.times >= assessment_start)
        density_scale = calibrate_occupancy_density(auxiliary[fit])
        auxiliary[..., 1] = auxiliary[..., 0] * density_scale[None]
        cutoff = .6 * params.free_speed_kmh.to_numpy()
        training = collect_events(d.values, auxiliary, d.times, fit, cutoff,
            np.random.default_rng(318), max_per_condition=128)
        classifier = QueueClassifier().fit(d.values[fit], d.times[fit], up, down, params)
        regressor = QueueSpeedRegressor().fit(d.values[fit], d.times[fit], up, down, params)
        model = EventQueueModel(policy=selected.policy, rounds=selected.rounds).fit(training, up, down, params)
        model.decoders = selected.decoders.copy()
        with (args.output / f"{panel}_event.pkl").open("wb") as f:
            pickle.dump({"model": model, "links": d.links, "up": up, "down": down, "params": params,
                "density_scale": density_scale, "fit_end": str(d.times[fit][-1]),
                "policy_selected_on": "October", "assessment": args.assessment_month + " not used for fitting or policy selection"}, f)
        for sampler in ("random", "balanced"):
            events = collect_events(d.values, auxiliary, d.times, test, cutoff,
                np.random.default_rng(321), max_per_condition=48, sampler=sampler)
            detail = []
            for w in events:
                h, _ = visible_event(w, "history_only")
                policy = old["baseline_policies"][w.condition]
                if policy == "speed":
                    speed = regressor.scores(h, w.times, up, down, params)
                    baseline = speed <= cutoff
                    if w.condition == "queue_onset" and not baseline.any():
                        baseline = onset_slowdown(speed, h[..., 0])
                else:
                    classifier.thresholds[w.condition], classifier.force_nonempty = policy
                    baseline = classifier.classify(classifier.scores(h, w.times, up, down, params), w.condition)
                pred = model.predict(w, up, down, params)
                detail.append({"panel": panel, "sampler": sampler, "origin": str(d.times[w.origin]),
                    "condition": w.condition, "baseline": proxy_queue_iou(w.truth, baseline, w.valid),
                    "event": proxy_queue_iou(w.truth, pred, w.valid), "predicted_cells": int(pred.sum()),
                    "true_proxy_cells": int(w.truth.sum()), "first_queue_step": w.first_queue_step})
            frame = pd.DataFrame(detail)
            scores = frame.groupby("condition")[["baseline", "event"]].mean()
            if len(scores) != 2:
                raise ValueError("Missing assessment condition")
            record = {"panel": panel, "sampler": sampler, "events": len(events),
                "chosen_origin_policy": model.policy, "condition_scores": scores.to_dict("index"),
                "baseline_macro": float(scores.baseline.mean()), "event_macro": float(scores.event.mean()),
                "runtime_seconds": time.monotonic()-started}
            record["delta"] = record["event_macro"] - record["baseline_macro"]
            summary.append(record); rows.extend(detail)
            print(f"  {args.assessment_month} {sampler}: baseline={record['baseline_macro']:.4f}, event={record['event_macro']:.4f}, delta={record['delta']:+.4f}", flush=True)
        (args.output / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(rows).to_csv(args.output / "assessment_windows.csv", index=False)
        del d, auxiliary, classifier, regressor, model, training
        gc.collect()
    print("Confirmation completed", flush=True)


if __name__ == "__main__":
    main()
