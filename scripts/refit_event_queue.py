"""Refit only queue models that passed both temporal confirmation stages."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import load_state, graph
from src.queue_events import (EventQueueModel, collect_events, load_auxiliary,
    calibrate_occupancy_density)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--initial-promotion", type=Path, required=True)
    p.add_argument("--confirmation-promotion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    initial = {r["panel"] for r in json.loads(a.initial_promotion.read_text())["decisions"] if r["candidate_for_confirmation"]}
    decisions = json.loads(a.confirmation_promotion.read_text())["decisions"]
    manifest = {"training_source": "all nine unmasked training months only",
        "method": "event-conditioned queue classifier with matched origin availability",
        "gate": "November development and separately refitted confirmation month gates passed",
        "models": [], "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (Path("src/queue_events.py"), Path(__file__))}}
    existing_manifest = a.output / "model_manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Refit source changed; use a separate model output directory")
        manifest["models"] = previous["models"]
    for decision in decisions:
        panel = decision["panel"]
        if panel not in initial or not decision["candidate_for_confirmation"]:
            continue
        existing = next((m for m in manifest["models"] if m["panel"] == panel), None)
        if existing:
            if hashlib.sha256((a.output / existing["file"]).read_bytes()).hexdigest() != existing["sha256"]:
                raise ValueError("Existing full-train checkpoint hash mismatch")
            print(f"Keep verified refit {panel}", flush=True)
            continue
        with (Path(decision["source_root"]) / f"{panel}_event.pkl").open("rb") as f:
            selected = pickle.load(f)["model"]
        print(f"Full train refit {panel}, {selected.policy}", flush=True)
        d = load_state(a.release_root, panel, "train", True)
        auxiliary = load_auxiliary(a.release_root, panel, d.times, d.links)
        density_scale = calibrate_occupancy_density(auxiliary)
        auxiliary[..., 1] = auxiliary[..., 0] * density_scale[None]
        up, down, _, params = graph(a.release_root, panel, d.links)
        events = collect_events(d.values, auxiliary, d.times, np.ones(len(d.times), bool),
            .6 * params.free_speed_kmh.to_numpy(), np.random.default_rng(318), max_per_condition=128)
        model = EventQueueModel(policy=selected.policy, rounds=selected.rounds).fit(events, up, down, params)
        model.decoders = selected.decoders.copy()
        path = a.output / f"{panel}_event.pkl"
        with path.open("wb") as f:
            pickle.dump({"model": model, "links": d.links, "up": up, "down": down, "params": params,
                "density_scale": density_scale, "fit_start": str(d.times[0]), "fit_end": str(d.times[-1]),
                "origin_policy": model.policy, "decoder_selection": "October training proxies",
                "feature_boundary": "released window history plus optional permitted row at T only"}, f)
        manifest["models"].append({"panel": panel, "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "origin_policy": model.policy,
            "decoders": model.decoders, "training_events": len(events)})
        (a.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        del d, auxiliary, events, model
        gc.collect()
    if not manifest["models"]:
        raise ValueError("No models cleared the promotion gates")
    print(f"Refitted {len(manifest['models'])} accepted event models", flush=True)


if __name__ == "__main__":
    main()
