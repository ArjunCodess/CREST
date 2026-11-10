"""Prefix-only fits and the later block that model selection never saw."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .features import SCALE, artificial_mask, prediction_blocks, sampled_features
from .models import booster
from .physics import project_fd
from .evaluation import state_metrics

def _mask_api():
    official = Path(__file__).resolve().parents[1] / 'starter' / 'src'
    if str(official) not in sys.path:
        sys.path.insert(0, str(official))
    from task1.baseline_task1_historical_mean import stable_mask
    return stable_mask

def selection_and_untouched(times):
    """Seven-day selection block, then a later seven-day block selection never sees.

    The purged 80% prefix ends in early January. Both blocks sit in February,
    after that prefix. The final seven days stay out: temporal CV already used them.
    """
    times = pd.DatetimeIndex(times)
    end = times[-1]
    selection = np.asarray((times >= end - pd.Timedelta(days=21)) & (times < end - pd.Timedelta(days=14)))
    untouched = np.asarray((times >= end - pd.Timedelta(days=14)) & (times < end - pd.Timedelta(days=7)))
    if not selection.any() or not untouched.any():
        raise ValueError('Series is too short for a selection block and a later untouched block')
    return selection, untouched

def official_cell_mask(panel, times, links):
    """Official one-regime-per-day hash, in canonical timestamp text."""
    stable_mask = _mask_api()
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize('UTC')
    dates = times.strftime('%Y-%m-%d')
    stamps = times.strftime('%Y-%m-%dT%H:%M:%SZ')
    regimes = _regimes(panel, times)
    link_ids = [str(link) for link in links]
    n = len(link_ids)
    mask = np.zeros((len(times), n), dtype=bool)
    for row, regime in enumerate(regimes):
        drawn = stable_mask(
            panel, str(regime),
            pd.Series([dates[row]] * n),
            pd.Series([stamps[row]] * n),
            pd.Series(link_ids))
        mask[row] = np.asarray(drawn, dtype=bool)
    return mask

def fit_trees(values, times, up, down, params, rounds=160, max_rows=60000):
    """Audited tree imputer: same sampler and round count, no neural head and no queue heads."""
    rng = np.random.default_rng(42)
    masked, hidden = artificial_mask(values, rng)
    x, ix = sampled_features(masked, times, up, down, params, hidden, rng, max_rows, True)
    return [booster(42, rounds).fit(x, values[..., c].ravel()[ix] / SCALE[c]) for c in range(2)]

def tree_reconstruct(models, values, times, up, down, params):
    current = values.copy()
    for _ in range(2):
        pred = np.empty_like(values)
        for start, end, x in prediction_blocks(current, times, up, down, params, True):
            pred[start:end] = np.stack([
                m.booster_.predict(x.reshape(-1, x.shape[-1]), num_threads=4).reshape(x.shape[:2]) * SCALE[c]
                for c, m in enumerate(models)], -1)
        current = np.where(np.isfinite(values), values, np.maximum(pred, 0))
    return current

def score_masked_block(panel, values, times, links, up, down, params, residual, trees, block):
    """Train-prefix models see the official mask, including one day of context on each side."""
    idx = np.flatnonzero(block)
    lo = max(0, int(idx[0]) - 144)
    hi = min(len(times), int(idx[-1]) + 1 + 144)
    span_values = values[lo:hi]
    span_times = times[lo:hi]
    target = official_cell_mask(panel, span_times, links)
    masked = span_values.copy()
    masked[target] = np.nan
    tree = project_fd(tree_reconstruct(trees, masked, span_times, up, down, params), params)
    resid = project_fd(residual.reconstruct(masked, span_times, up, down, params), params)
    inside = np.zeros(len(span_times), dtype=bool)
    inside[int(idx[0]) - lo:int(idx[-1]) + 1 - lo] = True
    regimes = _regimes(panel, span_times)
    rows = []
    for regime in ('R1', 'R2', 'R3'):
        score = target & inside[:, None] & (regimes == regime)[:, None] & np.isfinite(span_values).all(-1)
        for name, pred in (('tree_only', tree), ('residual', resid)):
            if not score.any():
                metrics = {'rmse_speed_kmh': np.nan, 'rmse_flow_vph_per_lane': np.nan, 'S_state': 0.0, 'n_targets': 0}
            else:
                metrics = state_metrics(span_values, pred, params.lanes, score)
            rows.append({'panel': panel, 'regime': regime, 'model': name, 'projection': 'fd', **metrics})
    sl = slice(int(idx[0]) - lo, int(idx[-1]) + 1 - lo)
    return rows, np.asarray(pd.DatetimeIndex(span_times[sl]).strftime('%Y-%m-%dT%H:%M:%SZ')), tree[sl], resid[sl]

def _regimes(panel, times):
    official = Path(__file__).resolve().parents[1] / 'starter' / 'src'
    if str(official) not in sys.path:
        sys.path.insert(0, str(official))
    from task1.baseline_task1_historical_mean import regime_of_dates
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize('UTC')
    return regime_of_dates(panel, pd.Series(times.strftime('%Y-%m-%d')))
