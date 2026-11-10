"""Experimental Task 4 OD estimation. Not an official score and not a submission.

The fitted objective on observed links is

    0.5 * ||A f - c||^2 + (lambda / 2) * sum_p w_p (f_p - b_p)^2,  f >= 0.

Absolute weights are w_p = 1. Relative weights are w_p = reference^2 / max(b_p, floor)^2,
with the floor set from the prior so a zero or tiny prior cannot receive an
infinite penalty. Missing counts stay missing. Connectors and off-ramps without
a released count are not entered as zeros.

S_od has coefficient 0.20 * 0.45 = 0.09 in the composite. A +0.01 change in
S_od therefore moves the composite by 0.0009. This module does not observe
S_od, S_dev, or S_attr. Prior-deviation L1 and L2 are scale-free residuals
against the released prior, not S_dev.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.sparse import coo_matrix, csr_matrix, diags, issparse, vstack
from threadpoolctl import threadpool_limits

from .data import read_csv

ODME_WEIGHT_IN_TOTAL = 0.20
S_OD_WEIGHT_IN_ODME = 0.45
S_OD_COEFFICIENT_IN_TOTAL = ODME_WEIGHT_IN_TOTAL * S_OD_WEIGHT_IN_ODME
DELTA_TOTAL_PER_0_01_S_OD = 0.01 * S_OD_COEFFICIENT_IN_TOTAL
LAMBDA_GRID = (0.005, 0.05, 0.5)
BASELINE_REGULARIZATION = 0.05
BASELINE_WEIGHT = 'absolute'
WEIGHT_MODES = ('absolute', 'relative')
HOLDOUT_FRACTION = 0.15
HELD_LINK_GAIN_REQUIRED = 1e-3
SYNTHETIC_REGRESSION_TOLERANCE = 1e-3
REQUIRED_MECHANISMS = (
    'additive_absolute',
    'lognormal_multiplicative',
    'temporal_smoothed_biased_shares',
)
TRAIN_SPLIT = 'train'
_ISO_TIMESTAMP = re.compile(
    r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$'
)
IDENTIFICATION = {
    'shared_operator': (
        'Repeating the same incidence matrix with a separate flow vector at each '
        'period does not add count equations inside that period\'s null space.'
    ),
    'prior_share_model': (
        'The pooling model is f_t = theta_t * u with theta_t >= 0 and u >= 0. '
        'Every period then informs the same share. A component of u in the null '
        'space of A is still invisible to every period\'s counts; a positive '
        'prior penalty is what pins it. The rank-1 constraint excludes '
        'time-varying null components instead of estimating them.'
    ),
    'single_period': (
        'One period makes theta * u a reparameterization of one flow vector. '
        'It does not add a temporal constraint.'
    ),
    'task2': (
        'Period coupling is only among Task 4 departure intervals supplied by '
        'the caller. Later records are not used to adjust an earlier Task 2 forecast.'
    ),
}


class SolverError(RuntimeError):
    """The dual iteration did not meet its residual or KKT check."""


def relative_deviation_weights(prior, floor_fraction=0.01):
    """Map a prior to positive relative-deviation weights and the floor used.

    The reference level is the mean of priors that are not negligible next to
    the largest prior. Zero and tiny priors share that floor, so their weight
    is reference^2/floor^2 rather than unbounded. The reference normalization
    makes the lambda grid comparable to absolute weighting at a typical prior
    magnitude and makes weights invariant to a change of flow units.
    """
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 1 or prior.size == 0:
        raise ValueError('prior must be a non-empty vector')
    if not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError('prior must be finite and nonnegative')
    if floor_fraction <= 0:
        raise ValueError('relative floor fraction must be positive')
    peak = float(np.max(prior))
    support = prior[prior > max(peak * 1e-6, 0.0)]
    reference = float(np.mean(support)) if support.size else 1.0
    floor = max(floor_fraction * reference, 1e-8)
    scale = np.maximum(prior, floor)
    return np.square(reference / scale), float(floor)


def _as_csr(operator, n_counts):
    matrix = operator.tocsr() if issparse(operator) else csr_matrix(operator)
    matrix = matrix.astype(np.float64, copy=False).tocsr()
    if matrix.ndim != 2:
        raise ValueError('incidence must be two-dimensional')
    if matrix.shape[0] != n_counts:
        raise ValueError('counts do not match incidence rows')
    if matrix.shape[1] == 0 or matrix.shape[0] == 0:
        raise ValueError('incidence is empty')
    if not np.isfinite(matrix.data).all():
        raise ValueError('incidence contains non-finite entries')
    return matrix


def _prepare_problem(operator, counts, prior, regularization, weights, weight_mode, relative_floor_fraction):
    """Validate once at the boundary and hand typed arrays to the dual solver."""
    counts = np.array(counts, dtype=np.float64, copy=True)
    prior = np.array(prior, dtype=np.float64, copy=True)
    if counts.ndim != 1 or prior.ndim != 1:
        raise ValueError('counts and prior must be vectors')
    matrix = _as_csr(operator, counts.size)
    if prior.size != matrix.shape[1]:
        raise ValueError('prior does not match incidence columns')
    if not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError('prior must be finite and nonnegative')
    if np.isinf(counts).any():
        raise ValueError('counts must be finite or NaN for missing observations')
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError('Positive prior regularization required')
    observed = np.isfinite(counts)
    if not observed.any():
        raise ValueError('No observed link counts')
    if np.any(counts[observed] < 0):
        raise ValueError('observed counts must be nonnegative')
    floor = None
    if weights is None:
        if weight_mode == 'absolute':
            weights = np.ones(prior.size, dtype=np.float64)
        elif weight_mode == 'relative':
            weights, floor = relative_deviation_weights(prior, relative_floor_fraction)
        else:
            raise ValueError('weight_mode must be absolute or relative')
        source = weight_mode
    else:
        weights = np.array(weights, dtype=np.float64, copy=True)
        if weights.shape != prior.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('weights must be positive, finite, and aligned with the prior')
        source = 'explicit'
    measured = matrix[observed]
    return {
        'measured': measured,
        'counts': np.ascontiguousarray(counts[observed]),
        'prior': np.ascontiguousarray(prior),
        'weights': np.ascontiguousarray(weights),
        'observed': observed,
        'regularization': float(regularization),
        'weight_source': source,
        'relative_floor': floor,
        'n_paths': int(matrix.shape[1]),
    }


def _dual_newton(measured, target, base, regularization, weights, max_iter, grad_tol, fail_tol):
    n_links = measured.shape[0]
    unit = bool(np.all(weights == 1.0))

    def dual(dual_variable):
        projected_dual = np.asarray(measured.T @ dual_variable, dtype=np.float64).ravel()
        unrestricted = base - projected_dual / (regularization * weights)
        flows = np.maximum(unrestricted, 0.0)
        value = (
            0.5 * float(dual_variable @ dual_variable)
            + float(target @ dual_variable)
            + 0.5 * regularization * float(weights @ (flows * flows))
        )
        gradient = dual_variable + target - np.asarray(measured @ flows, dtype=np.float64).ravel()
        return value, gradient, unrestricted

    dual_variable = np.zeros(n_links, dtype=np.float64)
    converged = False
    iteration = -1
    for iteration in range(max_iter):
        value, gradient, unrestricted = dual(dual_variable)
        residual = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        if residual < grad_tol:
            converged = True
            break
        active_mask = unrestricted > 0
        if not np.any(active_mask):
            hessian = np.eye(n_links, dtype=np.float64)
        else:
            active = measured[:, active_mask]
            if unit:
                hessian = np.eye(n_links, dtype=np.float64) + (active @ active.T).toarray() / regularization
            else:
                scales = 1.0 / np.sqrt(regularization * weights[active_mask])
                weighted = active @ diags(scales)
                hessian = np.eye(n_links, dtype=np.float64) + (weighted @ weighted.T).toarray()
        try:
            factor = cho_factor(hessian, lower=True, check_finite=False)
            step = -cho_solve(factor, gradient, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise SolverError('OD dual Hessian factorization failed') from exc
        slope = float(gradient @ step)
        if not np.isfinite(slope) or slope >= 0:
            raise SolverError('OD Newton step is not a descent direction')
        fraction = 1.0
        accepted = False
        candidate = dual_variable
        for _ in range(40):
            candidate = dual_variable + fraction * step
            if dual(candidate)[0] <= value + 1e-4 * fraction * slope:
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            raise SolverError('OD line search failed')
        dual_variable = candidate
    _, gradient, unrestricted = dual(dual_variable)
    residual = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    if residual >= fail_tol:
        raise SolverError(f'OD dual residual {residual:.3e} did not converge')
    flows = np.maximum(unrestricted, 0.0)
    return flows, residual, int(iteration + 1), bool(converged or residual < fail_tol)


def _kkt_diagnostics(measured, counts, prior, flows, regularization, weights, scale):
    scaled_flows = flows / scale
    residual = np.asarray(measured @ scaled_flows, dtype=np.float64).ravel() - counts / scale
    stationarity = (
        np.asarray(measured.T @ residual, dtype=np.float64).ravel()
        + regularization * weights * (scaled_flows - prior / scale)
    )
    free = scaled_flows > 0
    stationarity_inf = float(np.max(np.abs(stationarity[free]))) if np.any(free) else 0.0
    multiplier = np.where(free, 0.0, stationarity)
    dual_infeasibility = float(np.max(np.maximum(-multiplier, 0.0))) if multiplier.size else 0.0
    complementary = float(np.max(np.abs(multiplier * scaled_flows))) if multiplier.size else 0.0
    primal_bound = float(np.max(np.maximum(-flows, 0.0))) if flows.size else 0.0
    original_residual = np.asarray(measured @ flows, dtype=np.float64).ravel() - counts
    original_stationarity = (
        np.asarray(measured.T @ original_residual, dtype=np.float64).ravel()
        + regularization * weights * (flows - prior)
    )
    return {
        'kkt_stationarity_inf': stationarity_inf,
        'kkt_dual_infeasibility_inf': dual_infeasibility,
        'kkt_complementary_inf': complementary,
        'kkt_primal_bound_inf': primal_bound,
        'kkt_stationarity_original_inf': float(np.max(np.abs(original_stationarity[free]))) if np.any(free) else 0.0,
        'kkt_ok': bool(
            stationarity_inf <= 1e-4
            and dual_infeasibility <= 1e-4
            and complementary <= 1e-6
            and primal_bound <= 1e-8
        ),
    }


def solve_weighted_od(
    operator,
    counts,
    prior,
    regularization=BASELINE_REGULARIZATION,
    weights=None,
    weight_mode=BASELINE_WEIGHT,
    relative_floor_fraction=0.01,
    max_iter=100,
    grad_tol=1e-8,
    fail_tol=1e-6,
):
    """Solve nonnegative weighted ODME in the measured-link dual.

    Returns flows only after the dual residual and the KKT check both pass.
    """
    problem = _prepare_problem(
        operator, counts, prior, regularization, weights, weight_mode, relative_floor_fraction
    )
    observed_counts = problem['counts']
    scale = max(float(np.mean(np.abs(observed_counts))), 1.0)
    with threadpool_limits(limits=4):
        scaled_flows, residual, iterations, converged = _dual_newton(
            problem['measured'],
            observed_counts / scale,
            problem['prior'] / scale,
            problem['regularization'],
            problem['weights'],
            max_iter,
            grad_tol,
            fail_tol,
        )
    flows = scaled_flows * scale
    diagnostics = _kkt_diagnostics(
        problem['measured'], observed_counts, problem['prior'], flows,
        problem['regularization'], problem['weights'], scale,
    )
    if not diagnostics['kkt_ok'] or not converged:
        raise SolverError('OD solver failed the KKT check')
    loaded = np.asarray(problem['measured'] @ flows, dtype=np.float64).ravel()
    diagnostics.update({
        'converged': True,
        'dual_gradient_inf': float(residual),
        'iterations': int(iterations),
        'n_measured_links': int(observed_counts.size),
        'n_paths': problem['n_paths'],
        'regularization': problem['regularization'],
        'weight_source': problem['weight_source'],
        'relative_floor': problem['relative_floor'],
        'count_normalized_l1': normalized_abs_error(loaded, observed_counts),
        'count_rmse': float(np.sqrt(np.mean((loaded - observed_counts) ** 2))),
        'prior_deviation_l1': normalized_abs_error(flows, problem['prior']),
        'prior_deviation_l2': normalized_l2(flows, problem['prior']),
        'solver_cost': primal_objective(
            problem['measured'], observed_counts, problem['prior'], flows,
            problem['regularization'], problem['weights'],
        ),
    })
    return flows, diagnostics


def primal_objective(measured, counts, prior, flows, regularization, weights):
    residual = np.asarray(measured @ flows, dtype=np.float64).ravel() - counts
    deviation = flows - prior
    return 0.5 * float(residual @ residual) + 0.5 * regularization * float(weights @ (deviation * deviation))


def scipy_nnls_reference(operator, counts, prior, regularization, weights):
    """Dense reference for small problems. Not used on corridor-sized path sets."""
    matrix = operator.toarray() if issparse(operator) else np.asarray(operator, dtype=np.float64)
    penalty = np.sqrt(regularization) * np.sqrt(np.asarray(weights, dtype=np.float64))
    design = np.vstack([matrix, np.diag(penalty)])
    target = np.concatenate([np.asarray(counts, dtype=np.float64), penalty * np.asarray(prior, dtype=np.float64)])
    flows, _ = nnls(design, target)
    return flows


def normalized_abs_error(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape or not np.isfinite(estimate).all() or not np.isfinite(reference).all():
        raise ValueError('normalized error received non-finite or misaligned values')
    denominator = max(float(np.sum(np.abs(reference))), 1e-9)
    return float(np.sum(np.abs(estimate - reference)) / denominator)


def normalized_l2(estimate, reference):
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = max(float(np.linalg.norm(reference)), 1e-9)
    return float(np.linalg.norm(estimate - reference) / denominator)


def held_link_l1(operator, flows, counts, rows):
    predicted = np.asarray(operator[rows] @ flows, dtype=np.float64).ravel()
    return normalized_abs_error(predicted, np.asarray(counts)[rows])


def destination_attraction(flows, destinations):
    flows = np.asarray(flows, dtype=np.float64)
    zones = np.asarray(destinations)
    if flows.shape[0] != zones.shape[0]:
        raise ValueError('destinations do not match flows')
    order = np.unique(zones)
    totals = np.array([float(flows[zones == zone].sum()) for zone in order], dtype=np.float64)
    total = float(totals.sum())
    if total <= 0:
        return order, np.full(order.shape, np.nan)
    return order, totals / total


def attraction_l1(flows, truth, destinations):
    _, left = destination_attraction(flows, destinations)
    _, right = destination_attraction(truth, destinations)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('attraction is undefined for a nonpositive flow total')
    return float(0.5 * np.abs(left - right).sum())


def holdout_size(n_observed, fraction=HOLDOUT_FRACTION):
    if n_observed < 5:
        raise ValueError(f'{n_observed} observed links cannot support a 10-20% holdout')
    lower = max(1, int(np.ceil(0.10 * n_observed)))
    upper = int(np.floor(0.20 * n_observed))
    if upper < lower:
        raise ValueError('the 10-20% holdout band is empty')
    target = int(np.round(fraction * n_observed))
    return int(min(max(target, lower), upper))


def repeated_holdouts(n_observed, folds, fraction, rng):
    """Disjoint reproducible holdouts, each containing 10-20% of the rows."""
    if folds < 1:
        raise ValueError('folds must be positive')
    width = holdout_size(n_observed, fraction)
    if folds * width > n_observed:
        raise ValueError(f'{folds} disjoint holdouts of {width} do not fit in {n_observed} links')
    order = rng.permutation(n_observed)
    groups = [np.sort(order[i * width:(i + 1) * width]) for i in range(folds)]
    return groups, width


def _config_sort_key(regularization, weight_mode, error, frequency=0):
    return (error, -frequency, 0 if weight_mode == 'absolute' else 1, abs(regularization - BASELINE_REGULARIZATION), regularization)


def select_config(operator, counts, prior, eligible_rows, rng, inner_folds, fraction=HOLDOUT_FRACTION):
    """Choose lambda and weight from inner holdouts of the eligible rows only."""
    eligible_rows = np.asarray(eligible_rows, dtype=np.int64)
    safe = np.full(len(counts), np.nan, dtype=np.float64)
    safe[eligible_rows] = np.asarray(counts, dtype=np.float64)[eligible_rows]
    finite = np.flatnonzero(np.isfinite(safe))
    if not np.array_equal(np.sort(finite), np.sort(eligible_rows)):
        raise ValueError('eligible rows must be exactly the finite training counts')
    groups, width = repeated_holdouts(len(eligible_rows), inner_folds, fraction, rng)
    scores = []
    for regularization in LAMBDA_GRID:
        for weight_mode in WEIGHT_MODES:
            errors = []
            failed = False
            for held_local in groups:
                held = eligible_rows[held_local]
                train = np.setdiff1d(eligible_rows, held, assume_unique=False)
                masked = np.full(len(counts), np.nan, dtype=np.float64)
                masked[train] = safe[train]
                try:
                    flows, _ = solve_weighted_od(
                        operator, masked, prior, regularization=regularization, weight_mode=weight_mode
                    )
                except SolverError:
                    failed = True
                    break
                errors.append(held_link_l1(operator, flows, safe, held))
            if not failed:
                scores.append({
                    'regularization': float(regularization),
                    'weight_mode': weight_mode,
                    'inner_held_link_l1': float(np.mean(errors)),
                    'inner_folds': int(inner_folds),
                    'inner_holdout_links': int(width),
                })
    if not scores:
        raise SolverError('every selection candidate failed')
    chosen = min(scores, key=lambda row: _config_sort_key(row['regularization'], row['weight_mode'], row['inner_held_link_l1']))
    info = {
        'n_finite_observed': int(finite.size),
        'max_abs_count': float(np.max(np.abs(safe[finite]))),
        'candidates': scores,
        'selected_regularization': chosen['regularization'],
        'selected_weight_mode': chosen['weight_mode'],
    }
    return (chosen['regularization'], chosen['weight_mode']), info


def nested_link_assessment(operator, counts, prior, folds, seed, panel='panel', fraction=HOLDOUT_FRACTION):
    """Outer holdouts score models selected on the complementary train links."""
    counts = np.asarray(counts, dtype=np.float64)
    observed = np.flatnonzero(np.isfinite(counts))
    outer_rng = np.random.default_rng([seed, 17])
    groups, width = repeated_holdouts(len(observed), folds, fraction, outer_rng)
    outer_rows = []
    inner_rows = []
    choices = []
    for fold, held_local in enumerate(groups):
        held = observed[held_local]
        train = np.setdiff1d(observed, held, assume_unique=False)
        choice, info = select_config(
            operator, counts, prior, train,
            np.random.default_rng([seed, 29, fold]), folds, fraction,
        )
        choices.append(choice)
        for candidate in info['candidates']:
            inner_rows.append({'panel': panel, 'fold': int(fold), 'role': 'inner_selection', 'used_for_gate': False, **candidate})
        fit_counts = np.full(len(counts), np.nan, dtype=np.float64)
        fit_counts[train] = counts[train]
        configs = {
            'baseline': (BASELINE_REGULARIZATION, BASELINE_WEIGHT),
            'selected': choice,
        }
        for name, (regularization, weight_mode) in configs.items():
            flows, diagnostics = solve_weighted_od(
                operator, fit_counts, prior, regularization=regularization, weight_mode=weight_mode
            )
            outer_rows.append({
                'panel': panel,
                'fold': int(fold),
                'role': 'outer_assessment',
                'model': name,
                'regularization': float(regularization),
                'weight_mode': weight_mode,
                'held_link_l1': held_link_l1(operator, flows, counts, held),
                'prior_deviation_l1': diagnostics['prior_deviation_l1'],
                'prior_deviation_l2': diagnostics['prior_deviation_l2'],
                'train_count_normalized_l1': held_link_l1(operator, flows, counts, train),
                'n_held': int(held.size),
                'n_train': int(train.size),
                'holdout_links': int(width),
                'used_for_gate': True,
            })
    modal = min(
        {(regularization, weight_mode): choices.count((regularization, weight_mode)) for regularization, weight_mode in set(choices)}.items(),
        key=lambda item: _config_sort_key(item[0][0], item[0][1], 0.0, item[1]),
    )[0]
    return {
        'outer': outer_rows,
        'inner': inner_rows,
        'modal_regularization': float(modal[0]),
        'modal_weight_mode': modal[1],
        'fold_choices': [{'regularization': float(item[0]), 'weight_mode': item[1]} for item in choices],
    }


def promotion_gate(held_link_baseline, held_link_candidate, synthetic):
    """Keep the baseline unless every required mechanism retains path recovery."""
    missing = []
    retained = {}
    wins = {}
    for mechanism in REQUIRED_MECHANISMS:
        report = synthetic.get(mechanism) if synthetic else None
        baseline = None if report is None else report.get('baseline_path_l1')
        candidate = None if report is None else report.get('candidate_path_l1')
        if baseline is None or candidate is None or not np.isfinite([baseline, candidate]).all():
            missing.append(mechanism)
            retained[mechanism] = False
            wins[mechanism] = False
            continue
        retained[mechanism] = bool(candidate <= baseline + SYNTHETIC_REGRESSION_TOLERANCE)
        wins[mechanism] = bool(candidate < baseline - SYNTHETIC_REGRESSION_TOLERANCE)
    benefit = bool(
        np.isfinite([held_link_baseline, held_link_candidate]).all()
        and held_link_candidate < held_link_baseline - HELD_LINK_GAIN_REQUIRED
    )
    passed = bool(benefit and not missing and all(retained.values()))
    if missing:
        reason = 'missing evidence for ' + ', '.join(missing)
    elif not benefit:
        reason = 'outer held-link error does not improve on the baseline by the preregistered margin'
    elif not all(retained.values()):
        won = [name for name, did in wins.items() if did]
        if len(won) == 1:
            reason = 'only one synthetic assumption improves path recovery'
        else:
            reason = 'path recovery is not retained on every synthetic assumption'
    else:
        reason = 'preregistered gate passed on this run; production artifacts were not changed'
    return {
        'gate_passed': passed,
        'decision': 'gate_passed_not_deployed' if passed else 'keep_baseline',
        'held_link_benefit': benefit,
        'held_link_baseline': None if held_link_baseline is None else float(held_link_baseline),
        'held_link_candidate': None if held_link_candidate is None else float(held_link_candidate),
        'mechanisms_retained': retained,
        'mechanism_wins': wins,
        'missing_mechanisms': missing,
        'reason': reason,
        'production_change': False,
        's_od_coefficient_in_total': S_OD_COEFFICIENT_IN_TOTAL,
        'delta_total_per_0_01_s_od': DELTA_TOTAL_PER_0_01_S_OD,
        'hidden_scores_observed': False,
    }


def independent_lognormal_truth(n_paths, rng, log_location=None, sigma=0.75):
    """Path truth drawn without reference to a released or corrupted prior."""
    if log_location is None:
        log_location = float(np.log(80.0))
    return rng.lognormal(log_location, sigma, size=n_paths)


def additive_absolute_prior(truth, rng, relative_sigma=0.25):
    sigma = relative_sigma * float(np.mean(truth))
    prior = np.maximum(truth + rng.normal(0.0, sigma, size=truth.shape), 0.0)
    return prior, {
        'mechanism': 'additive_absolute',
        'noise': 'homoscedastic normal added to truth, then truncated at zero',
        'sigma': float(sigma),
        'sigma_over_mean_truth': float(relative_sigma),
        'truth_law': 'independent lognormal path flows; the prior is not an input',
    }


def lognormal_multiplicative_prior(truth, rng, sigma=0.35):
    factor = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=truth.shape))
    return truth * factor, {
        'mechanism': 'lognormal_multiplicative',
        'formula': 'truth * exp(Normal(-0.5*sigma^2, sigma))',
        'sigma': float(sigma),
        'mean_unbiased_factor': True,
        'truth_law': 'independent lognormal path flows; the prior is not an input',
    }


def temporal_smoothed_biased_shares(n_paths, periods, destinations, rng, origin='2030-06-01T15:00:00Z'):
    """Synthetic multi-period truth and a causally smoothed, level-biased prior.

    The random walk is the latent share. The prior at period t uses the latent
    share through t only. No Task 2 history or later forecast window is read.
    """
    if periods < 2:
        raise ValueError('temporal synthetic evidence needs at least two periods')
    zones = np.asarray(destinations)
    if zones.shape != (n_paths,):
        raise ValueError('destinations must align with paths')
    timestamps = pd.date_range(origin, periods=periods, freq='h', tz='UTC')
    log_share = rng.normal(0.0, 0.5, size=n_paths)
    shares = []
    for _ in range(periods):
        log_share = log_share + rng.normal(0.0, 0.15, size=n_paths)
        share = np.exp(log_share)
        shares.append(share / share.sum())
    totals = 1000.0 * (1.0 + 0.15 * np.sin(np.arange(periods)))
    truth = [float(totals[t]) * shares[t] for t in range(periods)]
    unique_zones = np.unique(zones)
    biased_zones = set(unique_zones[:max(1, len(unique_zones) // 5)])
    bias = np.ones(n_paths, dtype=np.float64)
    bias[np.isin(zones, list(biased_zones))] = 1.5
    smooth = shares[0].copy()
    priors = []
    for t in range(periods):
        smooth = 0.7 * smooth + 0.3 * shares[t]
        biased = smooth * bias
        priors.append(1.25 * float(totals[t]) * biased / biased.sum())
    meta = {
        'mechanism': 'temporal_smoothed_biased_shares',
        'truth_law': 'log-share random walk with a smooth positive total; independent of any released prior',
        'prior_law': 'causal smooth 0.7*lag + 0.3*current, 1.5 share bias on one fifth of zones, 1.25 level bias',
        'uses_future_latent_share': False,
        'task2_forecast': False,
        'timestamps': [item.isoformat() for item in timestamps],
    }
    return truth, priors, [item.isoformat() for item in timestamps], meta


def counts_from_paths(operator, truth, observed_rows):
    loaded = np.asarray(operator @ truth, dtype=np.float64).ravel()
    counts = np.full(operator.shape[0], np.nan, dtype=np.float64)
    observed_rows = np.asarray(observed_rows, dtype=np.int64)
    counts[observed_rows] = loaded[observed_rows]
    return counts


def _mean_metric(values):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or array.size == 0:
        return None
    return float(np.mean(array))


def synthetic_recovery(operator, observed_rows, destinations, regularization, weight_mode, trials, periods, seed):
    """Path and attraction errors on synthetic truth. Count fit is stored separately."""
    observed_rows = np.asarray(observed_rows, dtype=np.int64)
    n_paths = operator.shape[1]
    records = []
    for trial in range(trials):
        for mechanism in ('additive_absolute', 'lognormal_multiplicative'):
            rng = np.random.default_rng([seed, trial, 1 if mechanism == 'additive_absolute' else 2])
            truth = independent_lognormal_truth(n_paths, rng)
            if mechanism == 'additive_absolute':
                prior, assumptions = additive_absolute_prior(truth, rng)
            else:
                prior, assumptions = lognormal_multiplicative_prior(truth, rng)
            counts = counts_from_paths(operator, truth, observed_rows)
            flows, diagnostics = solve_weighted_od(
                operator, counts, prior, regularization=regularization, weight_mode=weight_mode
            )
            records.append({
                'trial': int(trial),
                'period': 0,
                'mechanism': mechanism,
                'regularization': float(regularization),
                'weight_mode': weight_mode,
                'path_normalized_l1': normalized_abs_error(flows, truth),
                'attraction_l1': attraction_l1(flows, truth, destinations),
                'count_normalized_l1': diagnostics['count_normalized_l1'],
                'assumptions': assumptions,
            })
        if periods >= 2:
            rng = np.random.default_rng([seed, trial, 3])
            truths, priors, timestamps, assumptions = temporal_smoothed_biased_shares(
                n_paths, periods, np.asarray(destinations), rng
            )
            period_flows = []
            for period, (truth, prior) in enumerate(zip(truths, priors)):
                counts = counts_from_paths(operator, truth, observed_rows)
                flows, diagnostics = solve_weighted_od(
                    operator, counts, prior, regularization=regularization, weight_mode=weight_mode
                )
                period_flows.append(flows)
                records.append({
                    'trial': int(trial),
                    'period': int(period),
                    'mechanism': 'temporal_smoothed_biased_shares',
                    'regularization': float(regularization),
                    'weight_mode': weight_mode,
                    'path_normalized_l1': normalized_abs_error(flows, truth),
                    'attraction_l1': attraction_l1(flows, truth, destinations),
                    'count_normalized_l1': diagnostics['count_normalized_l1'],
                    'assumptions': assumptions,
                })
            counts_by_period = [counts_from_paths(operator, truth, observed_rows) for truth in truths]
            try:
                pooling = pool_prior_shares(
                    operator, counts_by_period, priors, timestamps,
                    regularization=regularization, weight_mode=weight_mode,
                )
            except SolverError as exc:
                records.append({
                    'trial': int(trial),
                    'period': None,
                    'mechanism': 'temporal_prior_share_pooling_diagnostic',
                    'regularization': float(regularization),
                    'weight_mode': weight_mode,
                    'path_normalized_l1': None,
                    'attraction_l1': None,
                    'count_normalized_l1': None,
                    'assumptions': {**assumptions, 'pooling_status': str(exc)},
                    'used_for_gate': False,
                })
            else:
                if pooling['status'] == 'solved':
                    for period, truth in enumerate(truths):
                        records.append({
                            'trial': int(trial),
                            'period': int(period),
                            'mechanism': 'temporal_prior_share_pooling_diagnostic',
                            'regularization': float(regularization),
                            'weight_mode': weight_mode,
                            'path_normalized_l1': normalized_abs_error(pooling['flows'][period], truth),
                            'attraction_l1': attraction_l1(pooling['flows'][period], truth, destinations),
                            'count_normalized_l1': normalized_abs_error(
                                np.asarray(operator[observed_rows] @ pooling['flows'][period]).ravel(),
                                np.asarray(operator[observed_rows] @ truth).ravel(),
                            ),
                            'assumptions': assumptions,
                            'used_for_gate': False,
                        })
    return records


def summarize_synthetic(records, candidate_weight, candidate_regularization):
    summary = {}
    frame = pd.DataFrame(records)
    for mechanism in REQUIRED_MECHANISMS:
        block = frame[frame.mechanism.eq(mechanism)]
        baseline = block[block.weight_mode.eq(BASELINE_WEIGHT) & np.isclose(block.regularization, BASELINE_REGULARIZATION)]
        candidate = block[block.weight_mode.eq(candidate_weight) & np.isclose(block.regularization, candidate_regularization)]
        summary[mechanism] = {
            'baseline_path_l1': _mean_metric(baseline.path_normalized_l1) if len(baseline) else None,
            'candidate_path_l1': _mean_metric(candidate.path_normalized_l1) if len(candidate) else None,
            'baseline_attraction_l1': _mean_metric(baseline.attraction_l1) if len(baseline) else None,
            'candidate_attraction_l1': _mean_metric(candidate.attraction_l1) if len(candidate) else None,
            'baseline_count_normalized_l1': _mean_metric(baseline.count_normalized_l1) if len(baseline) else None,
            'candidate_count_normalized_l1': _mean_metric(candidate.count_normalized_l1) if len(candidate) else None,
        }
    return summary


def timestamp_adjacency(timestamps, max_gap):
    """Connect periods by elapsed time. Interval names are rejected."""
    text = [str(item) for item in timestamps]
    if not text or not all(_ISO_TIMESTAMP.match(item) for item in text):
        raise ValueError('adjacency requires ISO-8601 timestamps, not period tokens or interval names')
    parsed = pd.to_datetime(text, utc=True, format='ISO8601')
    if parsed.isna().any():
        raise ValueError('unparsed timestamp')
    order = np.argsort(parsed.to_numpy())
    gap_limit = pd.Timedelta(max_gap)
    edges = []
    for left, right in zip(order[:-1], order[1:]):
        gap = parsed[right] - parsed[left]
        if gap > pd.Timedelta(0) and gap <= gap_limit:
            edges.append((int(left), int(right)))
    return [item.isoformat() for item in parsed], edges


def pool_prior_shares(
    operator,
    counts_by_period,
    priors_by_period,
    timestamps,
    regularization=BASELINE_REGULARIZATION,
    weight_mode=BASELINE_WEIGHT,
    max_alternations=25,
):
    """Nonnegative prior-share model f_t = theta_t * u, when time can constrain it."""
    n_periods = len(counts_by_period)
    if n_periods < 2 or len(priors_by_period) != n_periods:
        return {
            'status': 'not_applicable_single_static_interval',
            'flows': None,
            'fabricated_periods': 0,
            'identification': IDENTIFICATION,
        }
    parsed, _ = timestamp_adjacency(timestamps, '1s')
    if len(parsed) != n_periods:
        raise ValueError('timestamps do not match periods')
    first_counts = np.asarray(counts_by_period[0], dtype=np.float64)
    matrix = _as_csr(operator, first_counts.size)
    priors = [np.array(item, dtype=np.float64, copy=True) for item in priors_by_period]
    counts = [np.array(item, dtype=np.float64, copy=True) for item in counts_by_period]
    reference = np.mean(priors, axis=0)
    if weight_mode == 'absolute':
        weights = np.ones(matrix.shape[1], dtype=np.float64)
    elif weight_mode == 'relative':
        weights, _ = relative_deviation_weights(reference)
    else:
        raise ValueError('weight_mode must be absolute or relative')
    share = np.maximum(reference, 0.0)
    if float(share.sum()) <= 0:
        share = np.ones(matrix.shape[1], dtype=np.float64)
    theta = np.ones(n_periods, dtype=np.float64)
    last_objective = None
    for _ in range(max_alternations):
        alpha = float(np.sum(np.square(theta)))
        if alpha <= 0:
            raise SolverError('prior-share scales collapsed')
        beta = np.zeros(matrix.shape[1], dtype=np.float64)
        for scale, prior in zip(theta, priors):
            beta += scale * prior
        row_precision = np.zeros(matrix.shape[0])
        row_target = np.zeros(matrix.shape[0])
        for scale, count in zip(theta, counts):
            if scale == 0:
                continue
            valid = np.isfinite(count)
            if not np.any(valid):
                raise SolverError('a pooling period has no finite counts')
            row_precision[valid] += scale*scale
            row_target[valid] += scale*count[valid]
        observed = row_precision > 0
        if not observed.any():
            raise SolverError('prior-share scales collapsed')
        # Repeated periods share A. Collapse their sufficient statistics by
        # link, preserving the objective up to a constant without enlarging
        # the measured-link dual by the number of periods.
        root_precision = np.sqrt(row_precision[observed])
        stacked = diags(root_precision) @ matrix[observed]
        new_share, _ = solve_weighted_od(
            stacked, row_target[observed]/root_precision, beta / alpha,
            regularization=regularization, weights=weights * alpha,
        )
        new_theta = np.zeros(n_periods, dtype=np.float64)
        for period, (count, prior) in enumerate(zip(counts, priors)):
            valid = np.isfinite(count)
            pattern = np.asarray(matrix[valid] @ new_share, dtype=np.float64).ravel()
            denominator = float(pattern @ pattern) + regularization * float(np.sum(weights * new_share * new_share))
            if denominator <= 0:
                continue
            right_hand = float(count[valid] @ pattern) + regularization * float((weights * new_share) @ prior)
            new_theta[period] = max(0.0, right_hand / denominator)
        old_flows = theta[:, None] * share
        new_flows = new_theta[:, None] * new_share
        objective = 0.0
        for count, prior, flow in zip(counts, priors, new_flows):
            valid = np.isfinite(count)
            objective += primal_objective(matrix[valid], count[valid], prior, flow, regularization, weights)
        if last_objective is not None and objective > last_objective * (1 + 1e-5) + 1e-6:
            raise SolverError('prior-share alternation increased the objective')
        flow_change = float(np.max(np.abs(new_flows - old_flows)) / max(float(np.max(np.abs(new_flows))), 1e-8))
        theta = new_theta
        share = new_share
        last_objective = float(objective)
        if flow_change < 1e-6:
            break
    else:
        raise SolverError('prior-share alternation did not converge')
    return {
        'status': 'solved',
        'flows': [theta[period] * share for period in range(n_periods)],
        'theta': theta,
        'objective': float(last_objective),
        'timestamps': parsed,
        'adjacency_used': False,
        'fabricated_periods': 0,
        'identification': IDENTIFICATION,
    }


def experimental_correction_smooth(
    operator,
    counts_by_period,
    priors_by_period,
    timestamps,
    regularization=BASELINE_REGULARIZATION,
    weights=None,
    smoothness=0.05,
    max_gap='2h',
    column_cap=240,
):
    """Experimental quadratic penalty on timestamp-adjacent prior corrections.

    The penalty is (smoothness / 2) * ||(f_t - b_t) - (f_s - b_s)||^2 on edges
    from `timestamp_adjacency`. This is not the production solver.
    """
    if len(counts_by_period) < 2:
        return {'status': 'not_applicable_single_static_interval', 'flows': None, 'edges': [], 'identification': IDENTIFICATION}
    _, edges = timestamp_adjacency(timestamps, max_gap)
    n_times = len(counts_by_period)
    matrix = operator.tocsr() if issparse(operator) else csr_matrix(operator)
    n_paths = matrix.shape[1]
    if n_paths * n_times > column_cap:
        return {
            'status': 'refused_dense_factorization',
            'flows': None,
            'edges': edges,
            'reason': f'{n_paths * n_times} path-time columns exceed the dense cap of {column_cap}',
            'identification': IDENTIFICATION,
        }
    if weights is None:
        weights = np.ones(n_paths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rows = []
    targets = []
    for period, (count, prior) in enumerate(zip(counts_by_period, priors_by_period)):
        count = np.asarray(count, dtype=np.float64)
        prior = np.asarray(prior, dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(count))
        block = matrix[valid].toarray()
        for local_row, link in enumerate(valid):
            coefficient = np.zeros(n_times * n_paths, dtype=np.float64)
            coefficient[period * n_paths:(period + 1) * n_paths] = block[local_row]
            rows.append(coefficient)
            targets.append(count[link])
        penalty = np.sqrt(regularization) * np.sqrt(weights)
        for path in range(n_paths):
            coefficient = np.zeros(n_times * n_paths, dtype=np.float64)
            coefficient[period * n_paths + path] = penalty[path]
            rows.append(coefficient)
            targets.append(penalty[path] * prior[path])
    if smoothness < 0:
        raise ValueError('smoothness must be nonnegative')
    if smoothness > 0:
        scale = np.sqrt(smoothness)
        for left, right in edges:
            for path in range(n_paths):
                coefficient = np.zeros(n_times * n_paths, dtype=np.float64)
                coefficient[left * n_paths + path] = scale
                coefficient[right * n_paths + path] = -scale
                rows.append(coefficient)
                targets.append(scale * (
                    np.asarray(priors_by_period[left])[path] - np.asarray(priors_by_period[right])[path]
                ))
    if not rows:
        raise SolverError('experimental smoother received no equations')
    flows, _ = nnls(np.vstack(rows), np.asarray(targets, dtype=np.float64))
    split = [flows[period * n_paths:(period + 1) * n_paths] for period in range(n_times)]
    if any(np.any(item < -1e-8) or not np.isfinite(item).all() for item in split):
        raise SolverError('experimental smoother returned an invalid flow')
    return {
        'status': 'solved',
        'flows': [np.maximum(item, 0.0) for item in split],
        'edges': edges,
        'identification': IDENTIFICATION,
    }


def incidence_rank(matrix):
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError('rank input must be a non-empty matrix')
    singular = np.linalg.svd(array, compute_uv=False)
    tolerance = float(singular[0] * max(array.shape) * np.finfo(float).eps) if singular.size else 0.0
    rank = int(np.sum(singular > tolerance))
    return {
        'rank': rank,
        'nullity': int(array.shape[1] - rank),
        'n_rows': int(array.shape[0]),
        'n_columns': int(array.shape[1]),
        'tolerance': tolerance,
        'largest_singular_value': float(singular[0]),
        'smallest_singular_value': float(singular[-1]),
        'nonnegative_orthant_not_accounted': True,
        'dense_matrix_written': False,
    }


def _split_ids(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item for item in str(value).split(';') if item and item != 'nan']


def _period_kind(value):
    return 'timestamp' if _ISO_TIMESTAMP.match(str(value)) else 'token'


def _normalized_by_zone(zones, values):
    frame = pd.DataFrame({'zone': np.asarray(zones), 'value': np.asarray(values, dtype=np.float64)})
    if not np.isfinite(frame['value']).all():
        return None
    totals = frame.groupby('zone', sort=True)['value'].sum()
    total = float(totals.sum())
    if total <= 0:
        return None
    return totals / total


def attraction_observability(prior, offramp_counts):
    """Compare attractions only when zone, time, and unit metadata all align."""
    missing = []
    if prior is None or 'destination_zone' not in prior.columns or 'path_flow' not in prior.columns:
        missing.append('prior.destination_zone and prior.path_flow')
    if prior is not None and 'departure_time' not in prior.columns:
        missing.append('prior.departure_time')
    if offramp_counts is None:
        missing.extend([
            'off-ramp count table with destination_zone',
            'off-ramp count table with departure_time',
            'off-ramp count table with unit',
        ])
    else:
        for column in ('destination_zone', 'departure_time', 'unit', 'count'):
            if column not in offramp_counts.columns:
                missing.append(f'off-ramp counts.{column}')
    if prior is not None and 'unit' not in getattr(prior, 'columns', []):
        missing.append('prior.unit')
    if missing:
        return {'status': 'unidentifiable', 'comparison_performed': False, 'missing_fields': missing}
    prior_times = set(prior.departure_time.astype(str))
    ramp_times = set(offramp_counts.departure_time.astype(str))
    prior_units = set(prior.unit.astype(str))
    ramp_units = set(offramp_counts.unit.astype(str))
    if prior_times != ramp_times:
        missing.append('equal departure_time sets on the prior and the off-ramp counts')
    if prior_units != ramp_units or len(prior_units) != 1:
        missing.append('one shared unit on the prior and the off-ramp counts')
    if any(_period_kind(item) != 'timestamp' for item in prior_times | ramp_times):
        missing.append('timestamp departure_time rather than a period token')
    left = _normalized_by_zone(prior.destination_zone, pd.to_numeric(prior.path_flow, errors='coerce'))
    right = _normalized_by_zone(offramp_counts.destination_zone, pd.to_numeric(offramp_counts['count'], errors='coerce'))
    if left is None or right is None:
        missing.append('finite nonnegative attraction and off-ramp totals')
    elif set(left.index) != set(right.index):
        missing.append('identical destination zone sets; absent zones are not filled with zero')
    if missing:
        return {'status': 'unidentifiable', 'comparison_performed': False, 'missing_fields': missing}
    distance = float(0.5 * np.abs(left - right).sum())
    return {
        'status': 'compared',
        'comparison_performed': True,
        'prior_attraction_vs_offramp_l1': distance,
        'official_s_attr': False,
        'missing_fields': [],
    }


def inventory_tables(bundle):
    """Rank, OD multiplicity, time index, and attraction observability."""
    paths = bundle['paths']
    incidence = bundle['incidence']
    counts = bundle['counts']
    prior = bundle['prior']
    topology = bundle['topology']
    attachment = bundle['attachment']
    ramp_info = bundle.get('ramp_info') or {}
    link_ids, operator = build_incidence(incidence, paths.path_id.tolist())
    count_values = align_counts(counts, link_ids)
    groups = paths.groupby(['origin_zone', 'destination_zone'], sort=False).size()
    observed = np.isfinite(count_values)
    rank = incidence_rank(operator[observed].toarray()) if observed.any() else None
    prefixes = incidence.link_id.astype(str).map(_link_class)
    class_counts = {
        name: int(incidence.loc[prefixes.eq(name), 'link_id'].nunique())
        for name in ('mainline_like', 'ROFF', 'RON', 'CONN', 'other')
    }
    count_ids = set(counts.link_id.astype(str))
    incidence_ids = set(incidence.link_id.astype(str))
    roff_ids = set(incidence.loc[prefixes.eq('ROFF'), 'link_id'].astype(str))
    matched = incidence.loc[prefixes.eq('ROFF'), ['link_id', 'path_id']].merge(
        paths[['path_id', 'destination_zone']], on='path_id', how='left'
    )
    name_match = float(matched.link_id.eq('ROFF-' + matched.destination_zone).mean()) if len(matched) else None
    topology_off = []
    topology_on = []
    connectors = set()
    if topology is not None and len(topology):
        for value in topology.get('off_ramp_link_ids', pd.Series(dtype=str)).fillna(''):
            topology_off.extend(_split_ids(value))
        for value in topology.get('on_ramp_link_ids', pd.Series(dtype=str)).fillna(''):
            topology_on.extend(_split_ids(value))
        for column in ('incoming_link_ids', 'outgoing_link_ids'):
            if column not in topology.columns:
                continue
            for value in topology[column].fillna(''):
                connectors.update(item for item in _split_ids(value) if item.startswith('CONN'))
    topology_off = set(topology_off)
    attachment_ids = set()
    attachment_types = {}
    if attachment is not None and len(attachment) and 'ramp_link_id' in attachment.columns:
        attachment_ids = set(attachment.ramp_link_id.astype(str))
        if 'ramp_type' in attachment.columns:
            attachment_types = attachment.groupby(attachment.ramp_type.astype(str)).ramp_link_id.nunique().astype(int).to_dict()
    ramp_ids = set(ramp_info.get('sampled_ramp_ids') or [])
    periods = prior.departure_time.astype(str).drop_duplicates().tolist() if 'departure_time' in prior.columns else []
    count_periods = counts.departure_time.astype(str).drop_duplicates().tolist() if 'departure_time' in counts.columns else []
    observability = attraction_observability(prior, bundle.get('offramp_counts'))
    structural_missing = [
        'synthetic_link_counts.departure_time' if 'departure_time' not in counts.columns else None,
        'finite counts on ROFF-<zone> incidence links' if len(roff_ids & count_ids) == 0 else None,
        'shared ids between ROFF-<zone> links and topology off_ramp_link_ids' if len(roff_ids & topology_off) == 0 else None,
        'finite counts on topology off_ramp_link_ids' if len(topology_off & count_ids) == 0 else None,
        'ramp_states.destination_zone' if not ramp_info.get('has_destination_zone') else None,
        'ramp_attachment_map.destination_zone' if attachment is None or 'destination_zone' not in attachment.columns else None,
        'a released unit shared by synthetic_link_counts and ramp flow_vph' if 'unit' not in counts.columns else None,
    ]
    if observability['status'] != 'compared':
        observability = {
            'status': 'unidentifiable',
            'comparison_performed': False,
            'missing_fields': [item for item in structural_missing + observability['missing_fields'] if item],
        }
    return {
        'panel': bundle.get('panel'),
        'split': TRAIN_SPLIT,
        'n_paths': int(len(paths)),
        'n_od_pairs': int(groups.size),
        'paths_per_od_min': int(groups.min()) if len(groups) else 0,
        'paths_per_od_mean': float(groups.mean()) if len(groups) else 0.0,
        'paths_per_od_max': int(groups.max()) if len(groups) else 0,
        'incidence_links': int(len(incidence_ids)),
        'observed_incidence': rank,
        'link_classes': class_counts,
        'counts': {
            'columns': list(counts.columns),
            'n_rows': int(len(counts)),
            'n_links': int(counts.link_id.nunique()),
            'has_departure_time': bool('departure_time' in counts.columns),
            'periods': count_periods,
            'finite_on_incidence': int(observed.sum()),
            'incidence_links_without_counts': int((~observed).sum()),
            'count_links_outside_incidence': int(len(count_ids - incidence_ids)),
            'missing_counts_filled_as_zero': False,
        },
        'prior': {
            'columns': list(prior.columns),
            'periods': periods,
            'period_kinds': [_period_kind(item) for item in periods],
            'n_periods': int(len(periods)),
        },
        'temporal': {
            'multiple_time_indexed_counts': bool(len(count_periods) >= 2 and all(_period_kind(item) == 'timestamp' for item in count_periods)),
            'multiple_timestamped_priors': bool(len(periods) >= 2 and all(_period_kind(item) == 'timestamp' for item in periods)),
            'pooling': 'not_applicable_single_static_interval' if len(periods) < 2 or 'departure_time' not in counts.columns or any(_period_kind(item) != 'timestamp' for item in periods) else 'inputs_can_constrain_pooling',
            'december_used_as_validation': False,
        },
        'coverage': {
            'roff_links': int(len(roff_ids)),
            'roff_links_with_counts': int(len(roff_ids & count_ids)),
            'roff_name_matches_destination_when_present': name_match,
            'name_agreement_is_not_a_count_comparison': True,
            'topology_off_ramp_ids': int(len(topology_off)),
            'topology_on_ramp_ids': int(len(topology_on)),
            'topology_connector_ids': int(len(connectors)),
            'connectors_with_counts': int(len(connectors & count_ids)),
            'connector_counts': None,
            'unknown_connectors_imputed_as_zero': False,
            'attachment_ramp_ids': int(len(attachment_ids)),
            'attachment_type_counts': attachment_types,
            'attachment_ids_overlapping_topology_off_ramps': int(len(attachment_ids & topology_off)),
            'attachment_ids_overlapping_counts': int(len(attachment_ids & count_ids)),
            'sampled_ramp_state_ids_overlapping_topology_off_ramps': int(len(ramp_ids & topology_off)),
            'sampled_ramp_state_ids_overlapping_roff': int(len(ramp_ids & roff_ids)),
            'sampled_ramp_state_ids_overlapping_counts': int(len(ramp_ids & count_ids)),
            'ramp_state': {key: value for key, value in ramp_info.items() if key != 'sampled_ramp_ids'},
        },
        'attraction': observability,
    }


def _link_class(link_id):
    text = str(link_id)
    if text.startswith('ROFF'):
        return 'ROFF'
    if text.startswith('RON'):
        return 'RON'
    if text.startswith('CONN'):
        return 'CONN'
    if text.startswith('L'):
        return 'mainline_like'
    return 'other'


def build_incidence(incidence, path_ids):
    pairs = incidence[['link_id', 'path_id']].drop_duplicates()
    unknown = set(pairs.path_id.astype(str)) - set(map(str, path_ids))
    if unknown:
        raise ValueError('incidence contains a path_id absent from the path set')
    link_ids = pairs.link_id.astype(str).drop_duplicates().tolist()
    path_ids = [str(item) for item in path_ids]
    link_index = {link: i for i, link in enumerate(link_ids)}
    path_index = {path: i for i, path in enumerate(path_ids)}
    rows = pairs.link_id.astype(str).map(link_index).to_numpy(dtype=np.int64)
    columns = pairs.path_id.astype(str).map(path_index).to_numpy(dtype=np.int64)
    matrix = coo_matrix(
        (np.ones(len(pairs), dtype=np.float64), (rows, columns)),
        shape=(len(link_ids), len(path_ids)),
    ).tocsr()
    return link_ids, matrix


def align_counts(counts, link_ids):
    if counts.link_id.astype(str).duplicated().any():
        raise ValueError('duplicate count link_id')
    aligned = counts.assign(link_id=counts.link_id.astype(str)).set_index('link_id')['count'].reindex(link_ids)
    return pd.to_numeric(aligned, errors='coerce').to_numpy(dtype=np.float64)


def align_prior(prior, path_ids):
    if prior.path_id.astype(str).duplicated().any():
        raise ValueError('duplicate prior path_id')
    aligned = prior.assign(path_id=prior.path_id.astype(str)).set_index('path_id')['path_flow'].reindex([str(item) for item in path_ids])
    values = pd.to_numeric(aligned, errors='coerce').to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError('released prior does not cover every path')
    return values


def _ramp_sample(root, panel, split):
    ramp_dir = Path(root) / 'corridors' / panel / split / 'ramp_states'
    files = sorted(ramp_dir.rglob('*.parquet'))
    info = {
        'ramp_state_files': len(files),
        'columns': [],
        'files_sampled': [],
        'sampled_id_sets_equal': None,
        'sampled_ramp_ids': [],
        'sampled_ramp_types': [],
        'has_destination_zone': False,
        'has_flow_vph': False,
        'december_used_as_validation': False,
    }
    if not files:
        return info
    import pyarrow.parquet as pq
    schema = list(pq.ParquetFile(files[0]).schema_arrow.names)
    info['columns'] = schema
    info['has_destination_zone'] = 'destination_zone' in schema
    info['has_flow_vph'] = 'flow_vph' in schema
    columns = [name for name in ('ramp_link_id', 'ramp_type') if name in schema]
    sample_indexes = sorted({0, len(files) // 2, len(files) - 1})
    id_sets = []
    types = set()
    for index in sample_indexes:
        frame = pq.read_table(files[index], columns=columns).to_pandas()
        info['files_sampled'].append(files[index].name)
        id_sets.append(set(frame.ramp_link_id.astype(str)))
        if 'ramp_type' in frame.columns:
            types.update(frame.ramp_type.astype(str))
    info['sampled_id_sets_equal'] = all(item == id_sets[0] for item in id_sets)
    info['sampled_ramp_ids'] = sorted(set.union(*id_sets)) if id_sets else []
    info['sampled_ramp_types'] = sorted(types)
    return info


def load_train_panel(root, panel, split=TRAIN_SPLIT):
    """Load Task 4 train assets. Validation and private splits are refused."""
    if split != TRAIN_SPLIT:
        raise ValueError('OD v6 selection reads the train split only')
    root = Path(root)
    network = root / 'corridors' / panel / 'network'
    task = root / 'task4' / panel / split
    paths = read_csv(network / 'path_set.csv')
    incidence = read_csv(network / 'path_link_incidence.csv')
    counts = read_csv(task / 'synthetic_link_counts.csv')
    prior = read_csv(task / 'synthetic_weak_prior.csv')
    topology_path = network / 'lwr_mainline_topology.csv'
    attachment_path = network / 'ramp_attachment_map.csv'
    topology = read_csv(topology_path) if topology_path.exists() else pd.DataFrame()
    attachment = read_csv(attachment_path) if attachment_path.exists() else pd.DataFrame()
    link_ids, operator = build_incidence(incidence, paths.path_id.tolist())
    count_values = align_counts(counts, link_ids)
    prior_values = align_prior(prior, paths.path_id.tolist())
    destinations = paths.set_index(paths.path_id.astype(str)).reindex([str(item) for item in paths.path_id]).destination_zone.to_numpy()
    bundle = {
        'panel': panel,
        'paths': paths,
        'incidence': incidence,
        'counts': counts,
        'prior': prior,
        'topology': topology,
        'attachment': attachment,
        'ramp_info': _ramp_sample(root, panel, split),
        'offramp_counts': None,
    }
    return {
        'panel': panel,
        'operator': operator,
        'link_ids': link_ids,
        'path_ids': [str(item) for item in paths.path_id.tolist()],
        'counts': count_values,
        'prior': prior_values,
        'destinations': destinations,
        'inventory': inventory_tables(bundle),
    }
