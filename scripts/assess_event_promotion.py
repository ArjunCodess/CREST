"""Apply predeclared queue gates with paired day-cluster bootstrap intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def assess(roots, draws=1000):
    records = []
    for root in roots:
        summary_path = root / "pilot_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        detail = pd.read_csv(root / "assessment_windows.csv")
        detail["day"] = detail.origin.str[:10]
        for row in summary:
            x = detail[(detail.panel == row["panel"]) & (detail.sampler == row["sampler"])].copy()
            days = x.day.unique()
            rng = np.random.default_rng(192)
            deltas = []
            clusters = {day: group for day, group in x.groupby("day")}
            for _ in range(draws):
                sampled = pd.concat([clusters[day] for day in rng.choice(days, len(days), replace=True)])
                means = sampled.groupby("condition")[["event", "baseline"]].mean()
                if len(means) == 2:
                    deltas.append(float((means.event - means.baseline).mean()))
            interval = np.quantile(deltas, [.025, .975]).tolist()
            ongoing = row["condition_scores"]["queue_ongoing"]
            passed = row["delta"] >= .05 and ongoing["event"] - ongoing["baseline"] >= -.02 and interval[0] > 0
            records.append({**row, "source_root": str(root), "paired_day_bootstrap_95pct": interval,
                "bootstrap_days": len(days), "passes_sampler_gate": bool(passed)})
    panels = {}
    for row in records:
        panels.setdefault(row["panel"], []).append(row)
    decisions = []
    for panel, rows in panels.items():
        passed = len(rows) == 2 and {r["sampler"] for r in rows} == {"random", "balanced"} and all(r["passes_sampler_gate"] for r in rows)
        decisions.append({"panel": panel, "candidate_for_confirmation": bool(passed),
            "source_root": rows[0]["source_root"], "chosen_origin_policy": rows[0]["chosen_origin_policy"],
            "minimum_delta": min(r["delta"] for r in rows),
            "minimum_ci_lower": min(r["paired_day_bootstrap_95pct"][0] for r in rows)})
    return {"gate": "delta>=0.05, ongoing delta>=-0.02, paired day-cluster 95% lower>0, both samplers",
        "interpretation": "observed-speed training proxies; same-month samplers are robustness checks, not independent folds",
        "sampler_results": records, "decisions": decisions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = assess(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
