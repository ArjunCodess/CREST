"""Common-mask benchmark for reconstructions and conditional conservation bounds."""
from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .observations import Window
from .sparse_traffic import conservation_system
from .traffic_estimation import (
    graph_reconstruct, historical_level, outage_masks, persistence, physics_reconstruct, train_tree, tree_predict,
)
from .traffic_simulator import Simulation, simulate


class GraphRecurrentImputer(nn.Module):
    """Shared recurrent temporal encoder with directed neighboring values and masks."""

    def __init__(self, seed: int = 17) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.GRU(6, 12, bidirectional=True, batch_first=True)
            self.output = nn.Linear(24, 1)

    def forward(self, masked: torch.Tensor) -> torch.Tensor:
        filled = torch.nan_to_num(masked, nan=0.)
        mask = torch.isfinite(masked).to(filled.dtype)
        values = torch.stack((filled, mask), dim=-1)
        edge = torch.zeros_like(values[:1])
        ordered = torch.cat((torch.cat((edge, values[:-1])), values, torch.cat((values[1:], edge))), dim=-1)
        features, _ = self.encoder(ordered)
        return self.output(features).squeeze(-1)

    def fit_windows(self, windows: list[Window], channel: str, level: float, scale: float) -> None:
        self.train()
        examples = []
        for window in windows:
            values = window.inputs(channel)
            for mask in outage_masks(values.shape):
                included = mask & np.isfinite(values)
                if included.any():
                    sample = (np.where(mask, np.nan, values) - level) / scale
                    examples.append((torch.tensor(sample.T, dtype=torch.float32),
                                     torch.tensor(((values - level) / scale).T, dtype=torch.float32),
                                     torch.tensor(included.T)))
        optimizer = torch.optim.Adam(self.parameters(), lr=0.012)
        for _ in range(6):
            for masked, truth, included in examples:
                optimizer.zero_grad()
                error = nn.functional.smooth_l1_loss(self(masked)[included], truth[included])
                error.backward()
                optimizer.step()

    @torch.no_grad()
    def impute(self, values: np.ndarray, level: float, scale: float) -> np.ndarray:
        self.eval()
        sample = torch.tensor((values.T - level) / scale, dtype=torch.float32)
        prediction = self(sample).numpy().T * scale + level
        return np.where(np.isfinite(values), values, np.maximum(0, prediction))


def paired_cluster_interval(deltas: list[float], seed: int = 424242) -> list[float] | None:
    if len(deltas) < 2:
        return None
    x = np.asarray(deltas)
    random = np.random.default_rng(seed)
    draws = x[random.integers(len(x), size=(10_000, len(x)))].mean(axis=1)
    return np.quantile(draws, (0.025, 0.975)).tolist()


def benchmark_masks(shape: tuple[int, int], duration: int | None) -> list[tuple[str, np.ndarray]]:
    blocks = outage_masks(shape, duration)
    random_mask = np.zeros(shape, dtype=bool)
    random = np.random.default_rng(2600)
    random_mask.flat[random.choice(random_mask.size, int(blocks[0].sum()), replace=False)] = True
    return [(f"block-{i}", mask) for i, mask in enumerate(blocks)] + [("random", random_mask)]


class StateBenchmark:
    def __init__(self, training: list[Window], channel: str) -> None:
        self.channel = channel
        self.level = historical_level(training, channel)
        pool = np.concatenate([window.inputs(channel).ravel() for window in training])
        self.scale = max(1., float(np.nanquantile(pool, 0.9) - np.nanquantile(pool, 0.1)))
        self.tree = train_tree(training, channel, self.level, offline=True)
        self.recurrent = GraphRecurrentImputer()
        self.recurrent.fit_windows(training, channel, self.level, self.scale)

    def predict(self, window: Window, mask: np.ndarray) -> dict:
        inputs = window.inputs(self.channel, hidden=mask)
        start = time.perf_counter()
        tree = tree_predict(self.tree, inputs, self.level, offline=True)
        tree_seconds = time.perf_counter() - start
        start = time.perf_counter()
        filtered = graph_reconstruct(inputs, tree, window.graph, self.scale)
        return {
            "history": (np.where(np.isfinite(inputs), inputs, self.level), "optimal", None),
            "persistence": (persistence(inputs, self.level), "optimal", None),
            "tree": (tree, "optimal", tree_seconds),
            "graph_recurrent": (self.recurrent.impute(inputs, self.level, self.scale), "optimal", None),
            "graph_huber": (filtered.values, filtered.status, time.perf_counter() - start + tree_seconds),
        }

    def evaluate(
        self, windows: list[tuple[str, Window]], *, duration: int | None = None,
        radius: float | None = None,
    ) -> tuple[list[dict], list[float]]:
        entries, residuals = [], []
        for cluster, window in windows:
            for pattern, mask in benchmark_masks(window.quality.shape, duration):
                predictions = self.predict(window, mask)
                # Read evaluation measurements only after all methods have predicted.
                truth = window.inputs(self.channel)
                eligible = mask & np.isfinite(truth)
                for name, (prediction, status, seconds) in predictions.items():
                    covered = eligible & np.isfinite(prediction)
                    error = np.abs(prediction[covered] - truth[covered])
                    interval_hits = None
                    if name == "graph_huber" and radius is not None and status == "optimal":
                        interval_hits = int(np.sum(error <= radius))
                    if name == "graph_huber" and radius is None and status == "optimal" and pattern.startswith("block"):
                        residuals.extend(error.tolist())
                    entries.append({
                        "source": window.source, "cluster": cluster, "site": window.site, "pattern": pattern,
                        "channel": self.channel, "method": name, "status": status, "scored": len(error),
                        "eligible": int(eligible.sum()), "abs_error": float(error.sum()), "seconds": seconds,
                        "interval_hits": interval_hits, "interval_width": 2 * radius if interval_hits is not None else None,
                    })
        return entries, residuals


def aggregate(entries: list[dict], reference: str | None = None) -> dict:
    total = defaultdict(lambda: {"eligible": 0, "scored": 0, "abs_error": 0., "fallbacks": 0, "fits": 0,
                                 "fallback_scored": 0, "seconds": [], "interval_hits": 0, "interval_n": 0,
                                 "interval_width_sum": 0.})
    clustered = defaultdict(lambda: [0., 0])
    for row in entries:
        method = row["method"]
        output = total[method]
        for key in ("eligible", "scored", "abs_error"):
            output[key] += row[key]
        output["fits"] += 1
        output["fallbacks"] += row["status"] != "optimal"
        if row["status"] != "optimal":
            output["fallback_scored"] += row["scored"]
        if row["seconds"] is not None:
            output["seconds"].append(row["seconds"])
        if row["interval_hits"] is not None:
            output["interval_hits"] += row["interval_hits"]
            output["interval_n"] += row["scored"]
            output["interval_width_sum"] += row["interval_width"] * row["scored"]
        key = (row["cluster"], method)
        clustered[key][0] += row["abs_error"]
        clustered[key][1] += row["scored"]
    for row in total.values():
        row["mae"] = row["abs_error"] / row["scored"] if row["scored"] else None
        row["abstentions"] = row["eligible"] - row["scored"]
        row["p95_seconds"] = float(np.quantile(row.pop("seconds"), 0.95)) if row["seconds"] else row.pop("seconds") or None
        row["interval_coverage"] = row["interval_hits"] / row["interval_n"] if row["interval_n"] else None
        width_sum = row.pop("interval_width_sum")
        row["mean_interval_width"] = width_sum / row["interval_n"] if row["interval_n"] else None
    deltas = []
    if reference is not None:
        for key in sorted({k[0] for k in clustered}):
            ae, n = clustered[(key, "graph_huber")]
            ae_ref, n_ref = clustered[(key, reference)]
            if n and n == n_ref:
                deltas.append(ae / n - ae_ref / n_ref)
    return {"methods": dict(total), "paired_reference": reference, "paired_clusters": len(deltas),
            "graph_huber_minus_reference_95ci": paired_cluster_interval(deltas)}


def ensemble_storage_filter(sim: Simulation, storage: np.ndarray, flux_prior: np.ndarray, seed: int) -> np.ndarray:
    """An EnKF with cell-transmission supply/demand and uncertain boundary inputs."""
    random = np.random.default_rng(seed)
    members = 32
    n_cells, n_channels = sim.incidence.shape
    dt = sim.window.cadence_ms / 3_600_000
    cap = sim.capacity
    stock = np.broadcast_to(sim.storage[0], (members, n_cells)).copy()
    states = np.empty((len(storage), n_cells))
    send = np.maximum(-sim.incidence.T, 0)
    receive = np.maximum(sim.incidence.T, 0)
    send_count = send.sum(axis=0)
    outgoing_weights = send @ np.divide(1, send_count, where=send_count > 0, out=np.zeros_like(send_count))
    has_sender = send.sum(axis=1) > 0
    upper = np.max(sim.flow_upper, axis=0)
    for t, row in enumerate(storage):
        if t:
            demand = np.minimum(3900, stock * 38)
            supply = np.minimum(3900, (cap - stock) * 14)
            offered = ((demand @ send.T) * outgoing_weights)
            offered[:, ~has_sender] = flux_prior[~has_sender]
            offered = np.clip(offered + random.normal(0, 180, (members, n_channels)), 0, upper)
            for i in range(n_cells):
                incoming = receive[:, i] > 0
                total = offered[:, incoming].sum(axis=1)
                offered[:, incoming] *= np.minimum(1, supply[:, i] / np.maximum(total, 1e-9))[:, None]
            stock = np.clip(stock + dt * (offered @ sim.incidence.T), 0, cap)
        observed = np.flatnonzero(np.isfinite(row))
        if len(observed) and t:
            shifted = stock - stock.mean(axis=0)
            covariance = shifted.T @ shifted / (members - 1)
            local = covariance[np.ix_(observed, observed)] + 9 * np.eye(len(observed))
            gain = np.linalg.solve(local, covariance[observed]).T
            innovations = row[observed] + random.normal(0, 3, (members, len(observed))) - stock[:, observed]
            stock = np.clip(stock + innovations @ gain.T, 0, cap)
        states[t] = stock.mean(axis=0)
    return states


def physics_suite(development_seeds: list[int], test_seeds: list[int]) -> list[dict]:
    training = [simulate(seed, "junction") for seed in development_seeds]
    boundary = np.median(np.concatenate([s.flux for s in training]), axis=0)
    storage_level = float(np.median(np.concatenate([s.storage for s in training])))
    output = []
    for seed in test_seeds:
        sim = simulate(seed, "junction")
        for i, mask in enumerate(outage_masks(sim.window.quality.shape)):
            observed = sim.window.inputs("storage", hidden=mask)
            if i == 2:
                observed[sim.window.quality.shape[0] // 2, 0] += 200
            fit, bounds, fallback = physics_reconstruct(sim, observed, boundary, storage_level=storage_level)
            point = fallback if fit.values is None else fit.values[:sim.storage.size].reshape(sim.storage.shape)[:-1]
            filtered = ensemble_storage_filter(sim, observed, boundary, seed + i)
            truth = sim.window.values["storage"]
            true_inlet = float(sim.flux[:, sim.boundary_channels[0]].sum() * sim.window.cadence_ms / 3_600_000)
            output.append({
                "seed": seed, "stress": ("outage", "outage", "outage_and_biased_sensor")[i],
                "point_status": fit.status, "conditional_bounds": asdict(bounds), "used_fallback": fit.values is None,
                "solver_message": fit.solver_message,
                "point_evidence": "conditional" if bounds.status == "optimal" else "degraded",
                "balance_residual_vehicles": fit.balance_error,
                "missing_storage_mae_vehicles": float(np.mean(np.abs(point[mask] - truth[mask]))),
                "persistence_mae_vehicles": float(np.mean(np.abs(fallback[mask] - truth[mask]))),
                "ctm_enkf_mae_vehicles": float(np.mean(np.abs(filtered[mask] - truth[mask]))),
                "true_inlet_vehicles": true_inlet,
                "inlet_covered": (bounds.lower <= true_inlet <= bounds.upper) if bounds.status == "optimal" else None,
            })
    return output


def storage_scaling() -> list[dict]:
    output = []
    for cells, steps in ((6, 36), (12, 72), (24, 144)):
        sim = simulate(81, "junction", cells, steps)
        observations = sim.window.inputs("storage", hidden=outage_masks(sim.window.quality.shape)[1])
        prior = np.full(sim.flux.shape[1], 1800.)
        upper = np.broadcast_to(np.max(sim.flow_upper, axis=0), sim.flux.shape)
        system = conservation_system(sim.incidence, sim.storage[0], sim.capacity,
                                     np.zeros_like(upper), upper, sim.window.cadence_ms / 3_600_000)
        tracemalloc.start()
        began = time.perf_counter()
        try:
            fitted, bounds, _ = physics_reconstruct(sim, observations, prior, storage_level=35.)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        output.append({"cells": cells, "flux_channels": sim.flux.shape[1], "steps": steps,
                       "balance_nonzeros": system.equality.nnz, "variables": len(system.lower),
                       "elapsed_seconds": time.perf_counter() - began, "python_peak_bytes": peak,
                       "point_status": fitted.status, "interval_status": bounds.status})
    return output
