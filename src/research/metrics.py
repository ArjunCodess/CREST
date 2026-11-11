"""Truth scoring for synthetic research runs (not competition composite)."""
from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np


ScoringRegion = Literal['observed', 'missing', 'all']


def _region_mask(truth: np.ndarray, obs_mask: np.ndarray | None, region: ScoringRegion) -> np.ndarray:
    valid = np.isfinite(truth)
    if region == 'all':
        return valid
    if obs_mask is None:
        return valid if region == 'observed' else np.zeros_like(valid, dtype=bool)
    obs = np.asarray(obs_mask, dtype=bool)
    if region == 'observed':
        return valid & obs
    return valid & ~obs


def state_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
    obs_mask: np.ndarray | None = None,
    *,
    region: ScoringRegion = 'all',
) -> float:
    mask = _region_mask(truth, obs_mask, region)
    diff = (pred - truth)[mask]
    if diff.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(diff ** 2)))


def flow_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def dynamics_balance_rmse(storage: np.ndarray, B: np.ndarray, R: np.ndarray, z: np.ndarray, u: np.ndarray, dt: float) -> float:
    net = z @ B.T + u @ R.T
    resid = storage[1:] - storage[:-1] - dt * net
    return float(np.sqrt(np.mean(resid ** 2)))


def physical_recovery_errors(
    storage: np.ndarray,
    truth_storage: np.ndarray,
    z: np.ndarray,
    z_true: np.ndarray,
    u: np.ndarray,
    u_true: np.ndarray,
) -> dict[str, float]:
    return {
        'storage_vs_truth_rmse': state_rmse(storage, truth_storage, region='all'),
        'internal_flow_vs_truth_rmse': flow_rmse(z, z_true),
        'boundary_flow_vs_truth_rmse': flow_rmse(u, u_true),
    }


def bound_coverage(true_value: float, lower: float | None, upper: float | None) -> bool | None:
    if lower is None or upper is None or not np.isfinite(true_value):
        return None
    return bool(lower <= true_value <= upper)


def od_aggregate_error(estimated: float, truth_total: float) -> float:
    return float(abs(estimated - truth_total))


def timed(callable_fn, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callable_fn(*args, **kwargs)
    return result, time.perf_counter() - start
