"""Matched development comparisons for traffic reconstruction."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .observations import Window
from .traffic_benchmark import paired_cluster_interval
from .traffic_estimation import fit_storage, historical_level, outage_masks, persistence, train_tree, tree_predict
from .traffic_simulator import simulate


def wave_ablation(training: list[Window], evaluation: list[Window]) -> list[dict]:
    """Keep the tree tuning budget and masks fixed when adding neighbor lags."""
    rows = []
    for channel in ("speed", "flow"):
        level = historical_level(training, channel)
        plain = train_tree(training, channel, level, offline=True)
        with_waves = train_tree(training, channel, level, offline=True, wave=True)
        for window in evaluation:
            for index, mask in enumerate(outage_masks(window.quality.shape)):
                inputs = window.inputs(channel, hidden=mask)
                reference = tree_predict(plain, inputs, level, offline=True)
                candidate = tree_predict(with_waves, inputs, level, offline=True, wave=True)
                truth = window.inputs(channel)
                eligible = mask & np.isfinite(truth)
                if eligible.any():
                    rows.append({
                        "source": window.source, "site": window.site, "channel": channel, "mask": index,
                        "eligible": int(eligible.sum()),
                        "tree_mae": float(np.mean(np.abs(reference[eligible] - truth[eligible]))),
                        "directed_lag_tree_mae": float(np.mean(np.abs(candidate[eligible] - truth[eligible]))),
                    })
    return rows


def storage_ablation(development_seeds: tuple[int, ...], evaluation_seeds: tuple[int, ...]) -> list[dict]:
    """Use one storage/flow prior while changing only the requested assumptions."""
    training = [simulate(seed, "junction") for seed in development_seeds]
    flux = np.concatenate([sim.flux for sim in training])
    flow_prior = np.median(flux, axis=0)
    band_lo, band_hi = np.quantile(flux, (.05, .95), axis=0)
    storage_level = float(np.median(np.concatenate([sim.storage for sim in training])))
    rows = []
    for seed in evaluation_seeds:
        sim = simulate(seed, "junction")
        window = sim.window
        dt = window.cadence_ms / 3_600_000
        bounds_lo = np.zeros_like(sim.flux)
        bounds_hi = np.broadcast_to(sim.flow_upper.max(axis=0), sim.flux.shape)
        narrowed_lo, narrowed_hi = bounds_lo.copy(), bounds_hi.copy()
        boundary = list(sim.boundary_channels)
        narrowed_lo[:, boundary] = band_lo[boundary]
        narrowed_hi[:, boundary] = band_hi[boundary]
        for name, mask in enumerate(outage_masks(window.quality.shape)):
            inputs = window.inputs("storage", hidden=mask)
            bias_cell = (len(inputs) // 2, 0)
            if name == 2:
                inputs[bias_cell] += 200
            variants = (
                ("soft_no_balance", bounds_lo, bounds_hi, False, False),
                ("soft_balance", bounds_lo, bounds_hi, True, False),
                ("soft_tighter_boundaries", narrowed_lo, narrowed_hi, True, False),
                ("hard_observations", bounds_lo, bounds_hi, True, True),
            )
            for method, lower, upper, balance, hard in variants:
                fit, interval, fallback = fit_storage(
                    sim.incidence, sim.storage[0], sim.capacity, lower, upper, inputs, flow_prior, dt, boundary[0],
                    storage_level=storage_level, balance=balance, hard_observations=hard,
                )
                point = fallback if fit.values is None else fit.values[:sim.storage.size].reshape(sim.storage.shape)[:-1]
                true_total = float(sim.flux[:, boundary[0]].sum() * dt)
                rows.append({
                    "seed": seed, "mask": name, "biased_observation": name == 2, "method": method,
                    "status": fit.status, "fallback": fit.values is None, "balance_enforced": balance,
                    "balance_error_vehicles": fit.balance_error if balance else None,
                    "storage_mae_vehicles": float(np.mean(np.abs(point[mask] - sim.storage[:-1][mask]))),
                    "eligible": int(mask.sum()), "inlet_truth_vehicles": true_total,
                    "inlet_bound": asdict(interval),
                    "width_vehicles": interval.upper - interval.lower if interval.status == "optimal" else None,
                    "inlet_covered": interval.lower <= true_total <= interval.upper if interval.status == "optimal" else None,
                })
    return rows


def matched_mask_controls(
    training: list[Window], evaluation: list[tuple[int, Window]],
) -> list[dict]:
    """Use one random mask of exactly the same cardinality per contiguous mask."""
    rows = []
    for channel in ("speed", "flow"):
        level = historical_level(training, channel)
        tree = train_tree(training, channel, level, offline=True)
        for seed, window in evaluation:
            random = np.random.default_rng(seed)
            for index, block in enumerate(outage_masks(window.quality.shape)):
                control = np.zeros_like(block)
                control.flat[random.choice(block.size, int(block.sum()), replace=False)] = True
                for regime, mask in (("contiguous", block), ("random", control)):
                    observed = window.inputs(channel, hidden=mask)
                    predictions = {"persistence": persistence(observed, level),
                                   "tree": tree_predict(tree, observed, level, offline=True)}
                    truth = window.inputs(channel)
                    for method, prediction in predictions.items():
                        rows.append({
                            "seed": seed, "mask": index, "method": method, "channel": channel, "regime": regime,
                            "eligible": int(mask.sum()), "mae": float(np.mean(np.abs(prediction[mask] - truth[mask]))),
                        })
    return rows


def summarize_mask_controls(rows: list[dict]) -> dict:
    summary = {}
    for channel in ("speed", "flow"):
        summary[channel] = {}
        for regime in ("contiguous", "random"):
            cells = [row for row in rows if row["channel"] == channel and row["regime"] == regime]
            seeds = sorted({row["seed"] for row in cells})
            methods = ("persistence", "tree")
            by_seed = {
                method: [float(np.mean([r["mae"] for r in cells if r["seed"] == seed and r["method"] == method]))
                         for seed in seeds]
                for method in methods
            }
            summary[channel][regime] = {
                "eligible_per_method": sum(row["eligible"] for row in cells if row["method"] == "tree"),
                "mean_cluster_mae": {method: float(np.mean(by_seed[method])) for method in methods},
                "tree_minus_persistence_ci95": paired_cluster_interval(
                    [tree - latest for tree, latest in zip(by_seed["tree"], by_seed["persistence"])],
                ),
            }
    return summary
