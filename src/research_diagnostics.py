"""Assumption-dependent observation noise and missing-run diagnostics."""
import numpy as np


def noise_nugget(values, stable, max_lag=3):
    """Extrapolate short-lag semivariances, not an identified latent noise floor.

    Linear-lag and quadratic-lag fits represent different local dynamics. Their
    disagreement is reported rather than hidden by choosing the favorable fit.
    Quality selection and temporal correlation can bias either estimate.
    """
    x = np.asarray(values, dtype=float)
    stable = np.broadcast_to(stable, x.shape) & np.isfinite(x)
    semivariance, counts = [], []
    for lag in range(1, max_lag+1):
        ok = stable[lag:] & stable[:-lag]
        difference = (x[lag:]-x[:-lag])[ok]
        counts.append(int(ok.sum()))
        semivariance.append(float(np.var(difference)/2) if len(difference) >= 20 else np.nan)
    lags = np.arange(1, max_lag+1, dtype=float)
    valid = np.isfinite(semivariance)
    if valid.sum() < 3:
        return {'sigma_linear': None, 'sigma_quadratic': None, 'semivariance': [None if not np.isfinite(v) else v for v in semivariance],
                'pair_counts': counts, 'assumption': 'Insufficient stable adjacent pairs'}
    intercepts = [np.linalg.lstsq(np.stack([np.ones(max_lag), lags**power], -1)[valid],
                  np.asarray(semivariance)[valid], rcond=None)[0][0] for power in (1, 2)]
    return {'sigma_linear': float(np.sqrt(max(intercepts[0], 0))),
            'sigma_quadratic': float(np.sqrt(max(intercepts[1], 0))),
            'unclipped_variance_intercepts': [float(v) for v in intercepts],
            'semivariance': semivariance, 'pair_counts': counts,
            'assumption': 'Zero-mean temporally white measurement noise plus locally smooth latent traffic; not independently identifiable.'}


def missing_run_lengths(mask):
    hidden = np.asarray(mask, bool)
    index = np.arange(len(hidden))[:, None]
    previous = np.maximum.accumulate(np.where(~hidden, index, -1), axis=0)
    following = np.minimum.accumulate(np.where(~hidden, index, len(hidden))[::-1], axis=0)[::-1]
    return np.where(hidden, following-previous-1, 0)
