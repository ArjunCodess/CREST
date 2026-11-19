"""Package development results without copying raw measured inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/crest"))
    args = parser.parse_args()
    blobs = {name: (args.artifacts / f"{name}.json").read_bytes() for name in ("fixture", "measured-development")}
    fixture, measured = map(json.loads, blobs.values())
    if fixture["code_hash"] != measured["code_hash"] or measured["mode"] != "measured_development":
        raise ValueError("Regenerate both development receipts with the same source")
    ngsim = measured["ngsim"]
    models = ("history", "persistence", "tree", "graph_recurrent", "graph_huber")
    display = ("Historical median", "Persistence", "Tree", "Graph recurrent", "Graph Huber")
    state = {
        channel: ngsim["cross_site_state"][channel]["outages"]["None"]["contiguous"]["methods"]
        for channel in ("speed", "flow")
    }
    report = {
        "mode": measured["mode"],
        "code_hash": measured["code_hash"],
        "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()},
        "block_outage_state": {
            channel: {model: {field: values[model][field] for field in ("mae", "scored", "eligible", "fallbacks")}
                      for model in models} for channel, values in state.items()
        },
        "tracked_balance": ngsim["tracked_balance"],
        "cross_site_storage": ngsim["cross_site_storage"],
        "trajectory_advisory": {
            key: ngsim["trajectory_advisory"][key] for key in
            ("scope", "warning_position_basis", "probe_penetration", "communication_delay_ms", "horizon_steps", "methods")
        },
        "detector_replay": [
            {key: row[key] for key in ("site", "window_count", "recomputed", "matches_full_recompute", "raw_sha256")}
            for row in measured["pems"]["replays"]
        ],
        "synthetic_replay": {
            key: fixture["replay"][key] for key in ("window_count", "recomputed", "matches_full_recompute")
        },
    }
    generated = Path(__file__).resolve().parents[1] / "generated"
    (generated / "roadmap_development.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    line_end = r" \\"
    processed = report["trajectory_advisory"]["methods"]["gated_infrastructure"]
    storage = {row["mode"]: row for row in ngsim["cross_site_storage"]}
    discrepancy = storage["bounded_track_discrepancy"]
    replays = report["detector_replay"]
    rows = [
        rf"\newcommand{{\CrestPilotSpeedN}}{{{state['speed']['graph_huber']['scored']}}}",
        rf"\newcommand{{\CrestPilotFlowN}}{{{state['flow']['graph_huber']['scored']}}}",
        rf"\newcommand{{\CrestPilotStorage}}{{{discrepancy['missing_storage_mae']:.2f}}}",
        rf"\newcommand{{\CrestPilotStorageBase}}{{{discrepancy['persistence_mae']:.2f}}}",
        rf"\newcommand{{\CrestPilotDiscrepancy}}{{{discrepancy['track_discrepancy_cap_per_cell']:.0f}}}",
        rf"\newcommand{{\CrestPilotQueueHits}}{{{processed['detected_encounters']}}}",
        rf"\newcommand{{\CrestPilotQueueN}}{{{processed['encounters']}}}",
        rf"\newcommand{{\CrestPilotQueueFalse}}{{{processed['false_warnings']}}}",
        rf"\newcommand{{\CrestPilotTrips}}{{{processed['trips']}}}",
        rf"\newcommand{{\CrestPilotReplayA}}{{{replays[0]['recomputed']}/{replays[0]['window_count']}}}",
        rf"\newcommand{{\CrestPilotReplayB}}{{{replays[1]['recomputed']}/{replays[1]['window_count']}}}",
        r"\newcommand{\CrestPilotTable}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Estimator & Speed MAE (km/h) & Flow MAE (veh/h)" + line_end,
        r"\midrule",
    ]
    for name, title in zip(models, display):
        rows.append(f"{title} & {state['speed'][name]['mae']:.2f} & {state['flow'][name]['mae']:.2f}" + line_end)
    rows.extend([r"\bottomrule", r"\end{tabular}}"])
    (generated / "roadmap_development.tex").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
