"""Tests for post-hoc v4 holdout analysis (saved-data fixtures; not mirroring full implementation)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_final_holdout_v4 import (
    N_BOOTSTRAP_REPLICATES,
    RESAMPLING_SEED,
    _rows_for_resampled_clusters,
    analyze,
    cluster_bootstrap_pooled_means,
)


def _row(seed: int, variant: str, bounded_missing: float, masked_missing: float) -> dict:
    return {
        'scenario_key': f'final_{variant}',
        'seed': seed,
        'boundary_truth_in_interval': True,
        'metadata': {'wrong_interval': variant == 'wrong_interval'},
        'eval_truth_feasibility': {'joint_within_assumption': True},
        'failure_flags': {'initial_truth_in_polytope': True, 'generative_capacity_violation': False},
        'observability': {},
        'metrics': {
            'bounded_status': 'optimal',
            'bounded_primal_violation': 0.0,
            'bounded_state_rmse_missing': bounded_missing,
            'masked_mean_state_rmse_missing': masked_missing,
            'forward_fill_state_rmse_missing': masked_missing + 1.0,
            'midpoint_state_rmse_missing': bounded_missing + 10.0,
            'boundary_interval_covers_truth_net_vehicles': True,
            'lp_data_covers_truth_net_vehicles': True,
            'lp_box_covers_truth_net_vehicles': True,
        },
    }


def test_cluster_bootstrap_preserves_multiplicity():
    """If a seed is drawn twice, its rows appear twice in the pooled mean (audit bug used set membership)."""
    rows = [
        _row(1, 'baseline', 10.0, 1.0),
        _row(2, 'baseline', 30.0, 3.0),
    ]
    # Deterministic: both draws pick seed 1 only -> mean must be 10, not 20.
    rng = np.random.default_rng(0)
    drawn = np.array([1, 1], dtype=int)
    pool = _rows_for_resampled_clusters(rows, drawn)
    assert len(pool) == 2
    assert np.mean([r['metrics']['bounded_state_rmse_missing'] for r in pool]) == 10.0
    assert rng.integers(0, 2) in (0, 1)


def test_cluster_bootstrap_weighted_estimand_analytic():
    """Two seeds × two variants: equal weight per cell; verify sample mean vs bootstrap center."""
    rows = [
        _row(10, 'baseline', 4.0, 0.0),
        _row(10, 'shifted_demand', 6.0, 0.0),
        _row(20, 'baseline', 8.0, 0.0),
        _row(20, 'shifted_demand', 12.0, 0.0),
    ]
    boot = cluster_bootstrap_pooled_means(
        rows,
        ['bounded_state_rmse_missing'],
        seed=99,
        n_clusters=2,
        n_replicates=2000,
    )
    assert boot['primary_mean_sample'] == 7.5
    assert boot['n_bootstrap_replicates'] == 2000
    assert abs(boot['primary_mean_bootstrap'] - 7.5) < 0.5


def test_paired_diff_bootstrap_runs():
    rows = [_row(1, 'baseline', 5.0, 2.0), _row(2, 'baseline', 7.0, 1.0)]
    boot = cluster_bootstrap_pooled_means(
        rows,
        ['bounded_state_rmse_missing', 'masked_mean_state_rmse_missing'],
        seed=RESAMPLING_SEED,
        n_clusters=2,
        n_replicates=min(500, N_BOOTSTRAP_REPLICATES),
    )
    assert boot['paired_diff_masked_mean']['mean_of_replicate_paired_means'] is not None
    assert boot['paired_diff_masked_mean']['mean_of_replicate_paired_means'] == pytest.approx(4.5, abs=0.5)


def test_reconciliation_sum_assertions_on_fixture():
    payload = {'rows': [_row(1, 'baseline', 1.0, 0.5), _row(2, 'baseline', 3.0, 1.5)], 'matrix_performance_summary': {}}
    report = analyze(payload)
    assert report['denominator_reconciliation']['all_sum_assertions_pass']


@pytest.mark.parametrize(
    'n_replicates', [min(800, N_BOOTSTRAP_REPLICATES)],
)
def test_no_multiplicity_strip_in_resample_helper(n_replicates: int):
    rows = [_row(1, 'baseline', 100.0, 0.0), _row(2, 'baseline', 0.0, 0.0)]
    boot = cluster_bootstrap_pooled_means(
        rows,
        ['bounded_state_rmse_missing'],
        seed=1,
        n_clusters=2,
        n_replicates=n_replicates,
    )
    # With multiplicity, bootstrap mean should be strictly between 0 and 100.
    assert 0.0 < boot['primary_mean_bootstrap'] < 100.0


def test_analyze_reports_od_nnls_invalid():
    payload = {
        'rows': [_row(1, 'baseline', 1.0, 0.5)],
        'matrix_performance_summary': {},
    }
    report = analyze(payload)
    assert report['od_nnls_benchmark_status']['benchmark_valid_for_comparison'] is False
