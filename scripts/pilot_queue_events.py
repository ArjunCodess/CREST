"""Refitted queue pilot: train Jun-Sep, select Oct, assess Nov, train release only."""
import argparse
from dataclasses import asdict
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
from src.queue_events import CONDITIONS, EventQueueModel, collect_events, load_auxiliary, visible_event, calibrate_occupancy_density
from src.queue_model import QueueClassifier, QueueSpeedRegressor, onset_slowdown
from src.evaluation import proxy_queue_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=180)
    ap.add_argument("--train-events", type=int, default=128)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"training_source": "unmasked train observations only",
        "proxy_labels": "observed speed <= 0.60 free_speed; official latent labels unavailable",
        "fit": ["2030-06-01", "2030-09-30T22:30:00Z"],
        "selection": ["2030-10-01", "2030-11-01"],
        "assessment": ["2030-11-01", "2030-12-01"],
        "window_purge": "all input and target rows belong to their own block",
        "baseline": "existing QueueClassifier and QueueSpeedRegressor refitted before selection",
        "new_model": "separate condition heads; matched origin masks; occupancy/density/quality; window-IoU decoder",
        "panels": args.panel, "rounds": args.rounds,
        "origin_observed_mask": "independent 20/30/50 percent mask on T; approximate official availability",
        "policy_selection": "origin policy and decoder selected on October only",
        "assessment_use": "first November report; no fitting, threshold or policy selection on assessment"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary, rows = [], []
    for panel in args.panel:
        started = time.monotonic()
        print(f"Load {panel}", flush=True)
        d = load_state(args.release_root, panel, "train", True,
            start="2030-06-01T00:00:00Z", end="2030-12-01T00:00:00Z")
        auxiliary = load_auxiliary(args.release_root, panel, d.times, d.links)
        up, down, _, params = graph(args.release_root, panel, d.links)
        cutoff = .6 * params.free_speed_kmh.to_numpy()
        fit = np.asarray(d.times < pd.Timestamp("2030-09-30T22:30:00Z"))
        select = np.asarray((d.times >= pd.Timestamp("2030-10-01T00:00:00Z")) &
                            (d.times < pd.Timestamp("2030-11-01T00:00:00Z")))
        assess = np.asarray(d.times >= pd.Timestamp("2030-11-01T00:00:00Z"))
        density_scale = calibrate_occupancy_density(auxiliary[fit])
        auxiliary[..., 1] = auxiliary[..., 0] * density_scale[None]
        training = collect_events(d.values, auxiliary, d.times, fit, cutoff,
            np.random.default_rng(318), max_per_condition=args.train_events)
        selection = collect_events(d.values, auxiliary, d.times, select, cutoff,
            np.random.default_rng(319), max_per_condition=48)
        assessments = {sampler: collect_events(d.values, auxiliary, d.times, assess, cutoff,
            np.random.default_rng(320), max_per_condition=48, sampler=sampler)
            for sampler in ("random", "balanced")}
        window_rows = []
        for block, events in [("fit", training), ("selection", selection), *assessments.items()]:
            for w in events:
                window_rows.append({"block": block, "origin": str(d.times[w.origin]),
                    "condition": w.condition, "first_queue_step": w.first_queue_step,
                    "positive_targets": int(w.truth.sum()), "eligible_targets": int(w.valid.sum())})
        pd.DataFrame(window_rows).to_csv(args.output / f"{panel}_windows.csv", index=False)
        print(f"  Events train={len(training)}, selection={len(selection)}, assessment=" +
              str({k:len(v) for k,v in assessments.items()}), flush=True)
        if any(not any(w.condition == c for w in group) for group in
               [training, selection, *assessments.values()] for c in CONDITIONS):
            raise ValueError(f"{panel}: missing condition; cannot report a balanced queue score")
        print("  Refit existing classifier and speed baseline", flush=True)
        classifier = QueueClassifier().fit(d.values[fit], d.times[fit], up, down, params)
        regressor = QueueSpeedRegressor().fit(d.values[fit], d.times[fit], up, down, params)
        baseline_policies = {}
        for condition in CONDITIONS:
            group = [w for w in selection if w.condition == condition]
            cache = []
            for w in group:
                history, _ = visible_event(w, "history_only")
                probability = classifier.scores(history, w.times, up, down, params)
                speed = regressor.scores(history, w.times, up, down, params)
                pred = speed <= cutoff
                if condition == "queue_onset" and not pred.any():
                    pred = onset_slowdown(speed, history[..., 0])
                cache.append((w, probability, pred))
            policies = ["speed"] + [(float(t), force) for t in (.1, .2, .3, .4, .5, .6, .7, .8, .9) for force in (False, True)]
            scores = []
            for policy in policies:
                score = []
                for w, probability, reg_pred in cache:
                    if policy == "speed": pred = reg_pred
                    else:
                        classifier.thresholds[condition], classifier.force_nonempty = policy
                        pred = classifier.classify(probability, condition)
                    score.append(proxy_queue_iou(w.truth, pred, w.valid))
                scores.append(np.mean(score))
            baseline_policies[condition] = policies[int(np.argmax(scores))]
        candidates, select_scores = {}, {}
        for policy in ("history_only", "origin_observed"):
            print(f"  Train event model {policy}", flush=True)
            model = EventQueueModel(policy=policy, rounds=args.rounds).fit(training, up, down, params)
            model.select_decoders(selection, up, down, params)
            score = np.mean([np.mean([proxy_queue_iou(w.truth, model.predict(w, up, down, params), w.valid)
                for w in selection if w.condition == c]) for c in CONDITIONS])
            candidates[policy] = model; select_scores[policy] = float(score)
            print(f"    Selection proxy IoU={score:.4f}, decoder={model.decoders}", flush=True)
        chosen = max(select_scores, key=select_scores.get)
        with (args.output / f"{panel}_event.pkl").open("wb") as f:
            pickle.dump({"model": candidates[chosen], "links": d.links, "up": up, "down": down,
                "params": params, "density_scale": density_scale, "fit_end": str(d.times[fit][-1]), "selected_on": "October proxies"}, f)
        for sampler, events in assessments.items():
            details = []
            for w in events:
                history, _ = visible_event(w, "history_only")
                policy = baseline_policies[w.condition]
                if policy == "speed":
                    speed = regressor.scores(history, w.times, up, down, params)
                    baseline = speed <= cutoff
                    if w.condition == "queue_onset" and not baseline.any():
                        baseline = onset_slowdown(speed, history[..., 0])
                else:
                    classifier.thresholds[w.condition], classifier.force_nonempty = policy
                    baseline = classifier.classify(classifier.scores(history, w.times, up, down, params), w.condition)
                pred = candidates[chosen].predict(w, up, down, params)
                details.append({"panel": panel, "sampler": sampler, "origin": str(d.times[w.origin]),
                    "condition": w.condition, "baseline": proxy_queue_iou(w.truth, baseline, w.valid),
                    "event": proxy_queue_iou(w.truth, pred, w.valid), "predicted_cells": int(pred.sum()),
                    "true_proxy_cells": int(w.truth.sum()), "first_queue_step": w.first_queue_step})
            frame = pd.DataFrame(details)
            by_condition = frame.groupby("condition")[["baseline", "event"]].mean()
            record = {"panel": panel, "sampler": sampler, "events": len(events),
                "chosen_origin_policy": chosen, "selection_scores": select_scores,
                "baseline_policies": baseline_policies,
                "condition_scores": by_condition.to_dict("index"),
                "baseline_macro": float(by_condition.baseline.mean()),
                "event_macro": float(by_condition.event.mean()),
                "runtime_seconds": time.monotonic() - started}
            record["delta"] = record["event_macro"] - record["baseline_macro"]
            summary.append(record); rows.extend(details)
            print(f"  Assessment {sampler}: baseline={record['baseline_macro']:.4f}, event={record['event_macro']:.4f}, delta={record['delta']:+.4f}", flush=True)
        (args.output / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(rows).to_csv(args.output / "assessment_windows.csv", index=False)
        del d, auxiliary, classifier, regressor, candidates, training, selection, assessments
        gc.collect()
    print("Pilot complete; official latent queue performance remains unmeasured.", flush=True)


if __name__ == "__main__":
    main()
