"""Generate paper figures from saved validation state arrays (read-only artifacts)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"


def resolve_artifacts() -> Path:
    candidates = [
        os.environ.get("TRAFFICFLOW_ARTIFACTS_OFFICIAL"),
        PAPER.parent / "artifacts" / "official",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if list(path.glob("*_validation_states.npz")):
            return path
    raise FileNotFoundError("No *_validation_states.npz in official artifacts")


def main() -> None:
    run = resolve_artifacts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    files = sorted(run.glob("*_validation_states.npz"))
    preferred = run / "D12_I5_N_validation_states.npz"
    state_file = preferred if preferred.exists() else files[0]
    bundle = np.load(state_file)
    states = bundle["prediction"]
    v = states[..., 0]
    q = states[..., 1]
    k = q / np.maximum(v, 1.0)
    rng = np.random.default_rng(42)
    ix = rng.choice(v.size, min(12000, v.size), replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    sc = ax.scatter(
        k.ravel()[ix],
        q.ravel()[ix],
        c=v.ravel()[ix],
        s=5,
        alpha=0.6,
        cmap="viridis",
    )
    ax.set(
        xlabel="Derived density (vehicles/km, all lanes)",
        ylabel="Flow (vehicles/hour)",
        title="Reconstructed states: " + state_file.stem.replace("_states", ""),
    )
    fig.colorbar(sc, ax=ax, label="Speed (km/h)")
    fig.savefig(FIGURES / "fundamental_diagram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    width = min(len(v), 288)
    means = v.mean(axis=1)
    rolling = np.convolve(means, np.ones(width) / width, mode="valid")
    start = int(np.argmin(rolling))
    im = ax.imshow(
        v[start : start + width].T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=120,
    )
    ax.set(
        xlabel="Five-minute step within selected interval",
        ylabel="Link index (ID order)",
        title="Lowest-speed 24-hour reconstructed interval",
    )
    fig.colorbar(im, ax=ax, label="Speed (km/h)")
    fig.savefig(FIGURES / "space_time.png", dpi=180)
    plt.close(fig)

    meta = {
        "figure_source": state_file.name,
        "heatmap_interval_start": str(bundle["times"][start]),
        "artifacts_root": str(run),
    }
    (FIGURES / "figure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
