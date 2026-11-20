"""Cell-transmission truth for a chain or a freeway with ramps and multiple paths."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observations import Window


@dataclass(frozen=True)
class Simulation:
    window: Window
    storage: np.ndarray
    flux: np.ndarray
    incidence: np.ndarray
    capacity: np.ndarray
    flow_upper: np.ndarray
    boundary_channels: tuple[int, ...]


def simulate(seed: int, layout: str = "junction", cells: int = 6, steps: int = 72) -> Simulation:
    if cells < 4 or steps < 12 or layout not in ("chain", "junction"):
        raise ValueError("Need at least four cells, twelve steps, and a chain or junction")
    random = np.random.default_rng(seed)
    # Every channel has one sending and/or receiving cell. None is outside the network.
    edges = [(i, i + 1) for i in range(cells - 1)]
    internal = len(edges)
    edges += [(None, 0), (cells - 1, None)]
    if layout == "junction":
        edges += [(None, 1), (2, None)]
    incidence = np.zeros((cells, len(edges)))
    inbound = [[] for _ in range(cells)]
    outbound = [[] for _ in range(cells)]
    for channel, (up, down) in enumerate(edges):
        if up is not None:
            incidence[up, channel] = -1
            outbound[up].append(channel)
        if down is not None:
            incidence[down, channel] = 1
            inbound[down].append(channel)
    dt = 1 / 120  # Thirty seconds
    capacity = np.full(cells, 180.0)
    storage = np.zeros((steps + 1, cells))
    storage[0] = random.uniform(12, 22, cells)
    max_flow = np.full((steps, len(edges)), 4200.0)
    if layout == "junction":
        max_flow[:, internal + 2:] = 1100
    max_flow[steps // 4:steps * 3 // 4, internal - 1] = random.uniform(900, 1600)
    flux = np.zeros_like(max_flow)
    inlet_scale = random.uniform(2300, 3500)
    for t in range(steps):
        demand = np.minimum(4200, storage[t] * 45)
        supply = np.minimum(4200, (capacity - storage[t]) * 16)
        f = flux[t]
        f[internal] = inlet_scale + 450 * np.sin(t / 7) + random.normal(0, 80)
        if layout == "junction":
            f[internal + 2] = random.uniform(460, 1050)
        for i, channels in enumerate(outbound):
            if len(channels) == 1:
                f[channels[0]] = demand[i]
            else:
                f[channels] = demand[i] * np.array([0.83, 0.17])
        np.minimum(f, max_flow[t], out=f)
        for i, channels in enumerate(inbound):
            offered = f[channels].sum()
            if offered > supply[i]:
                f[channels] *= supply[i] / offered
        storage[t + 1] = storage[t] + dt * (incidence @ f)
    # Cell-average proxy for simulator speed and all-lane output rates.
    flow = np.array([flux[:, channels].sum(axis=1) for channels in outbound]).T
    speed = np.minimum(103, flow / np.maximum(storage[:-1], 1) * 2.1)
    interval = 30_000
    time = np.arange(steps, dtype=np.int64) * interval
    window = Window("synthetic_ctm", f"{layout}-{seed}", time, time + interval,
                    tuple(f"cell-{i}" for i in range(cells)), np.arange(cells) * 2100.,
                    {"speed": speed, "flow": flow, "storage": storage[:-1]},
                    (), (f"{layout}-{cells}",), np.ones_like(speed), interval)
    return Simulation(window, storage, flux, incidence, capacity, max_flow, tuple(range(internal, len(edges))))
