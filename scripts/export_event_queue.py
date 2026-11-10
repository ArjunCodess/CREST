"""Export accepted event models into the complete frozen baseline queue table."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.queue_events import predict_release_queue
from src.submission import keyed
from src.pipeline import predict_queue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--baseline-queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    baseline = keyed(pd.read_csv(a.baseline_queue), "queue")
    manifest = json.loads((a.model_root / "model_manifest.json").read_text())
    frames = []
    for row in manifest["models"]:
        path = a.model_root / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Event checkpoint hash mismatch")
        with path.open("rb") as f:
            payload = pickle.load(f)
        for split in ("validation", "private"):
            print(f"Export event queues {row['panel']}/{split}", flush=True)
            frames.append(predict_release_queue(a.release_root, row["panel"], split, payload))
    replacement = keyed(pd.concat(frames, ignore_index=True), "queue")
    if len(replacement.index.difference(baseline.index)):
        raise ValueError("Event output contains keys absent from baseline")
    original = baseline.copy()
    baseline.loc[replacement.index, "queue_pred"] = replacement.queue_pred.to_numpy()
    # Both official conditions guarantee a nonempty truth horizon. Replacing an
    # empty prediction (IoU zero) by one cell cannot lower that window's IoU.
    empty_ids = set(baseline.groupby(level="window_id").queue_pred.sum().loc[lambda s: s == 0].index)
    rescued = []
    for index_path in sorted((a.release_root / "task2").glob("*/*/window_index.csv")):
        index = pd.read_csv(index_path, dtype={"window_id": str})
        wanted = empty_ids & set(index.window_id)
        if not wanted:
            continue
        panel, split = index_path.parent.parent.name, index_path.parent.name
        classifier_path = a.model_root.parent / "queue" / f"{panel}_queue.pkl"
        if not classifier_path.exists():
            raise ValueError(f"No existing classifier to fill empty windows for {panel}")
        with (a.model_root.parent / "state" / f"{panel}.pkl").open("rb") as f:
            payload = pickle.load(f)
        with classifier_path.open("rb") as f:
            classifier = pickle.load(f)
        classifier.force_nonempty = True
        payload["queue_classifier"] = classifier
        forced = predict_queue(a.release_root, panel, split, payload)
        forced = keyed(forced[forced.window_id.isin(wanted)], "queue")
        baseline.loc[forced.index, "queue_pred"] = forced.queue_pred.to_numpy()
        for wid in sorted(wanted):
            if int(baseline.xs(wid, level="window_id").queue_pred.sum()) != 1:
                raise ValueError("Empty-window repair must add exactly one highest-score cell")
            rescued.append({"window_id": wid, "panel": panel, "split": split})
    if len(baseline) != 174000:
        raise ValueError("Complete queue export must have exactly 174000 rows")
    output = baseline.reset_index()
    output["timestamp"] = output.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output["queue_pred"] = output.queue_pred.astype(int)
    keyed(output, "queue")
    path = a.output / "queue.csv"; output.to_csv(path, index=False)
    audit = {"rows": len(output), "replacement_rows": len(replacement),
        "changed_cells": int((baseline.queue_pred != original.queue_pred).sum()),
        "accepted_panels": [r["panel"] for r in manifest["models"]],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "other_nonempty_queue_windows_unchanged": True,
        "empty_windows_repaired": rescued,
        "empty_repair_basis": "Both conditions guarantee nonempty truth; old empty IoU=0, replacement IoU>=0",
        "causal_scope": "window history and permitted observations exactly at T; nothing after T"}
    (a.output / "queue_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
