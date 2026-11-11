"""Post-hoc analysis for amended final holdout v4 (outside frozen inference source hash at record time)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.research.config_paths import REPO_ROOT
from src.research.experiments import ORACLE_METRICS, point_estimate_valid
from src.research.provenance import research_code_content_hash

RESAMPLING_SEED = 424242
N_INDEPENDENT_SEED_CLUSTERS = 10
N_BOOTSTRAP_REPLICATES = 10_000
PRIMARY = 'bounded_state_rmse_missing'
STATE_RMSE_MISSING_BASELINES = {
    'bounded_deployable': 'bounded_state_rmse_missing',
    'masked_mean': 'masked_mean_state_rmse_missing',
    'forward_fill': 'forward_fill_state_rmse_missing',
    'interval_midpoint': 'midpoint_state_rmse_missing',
    'oracle_diagnostic': 'oracle_state_rmse_missing',
}
RMSE_SLICE_SUFFIXES = ('missing', 'observed', 'all')
COVERAGE_METRICS = (
    'boundary_interval_covers_truth_net_vehicles',
    'lp_data_covers_truth_net_vehicles',
    'lp_box_covers_truth_net_vehicles',
)
OD_NNLS_METRIC = 'od_nnls_total_error'
OD_NNLS_BENCHMARK_INVALID = (
    'Frozen v4 runner uses nan_to_num(count_bounds, 0) center and fits OD NNLS on all '
    'incidence rows ignoring count_observed_mask (experiments.py ~L114–115). '
    'Do not use od_nnls_total_error for comparative claims; future fix: A[mask], c[mask].'
)


def _script_hash_full() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _script_hash_prefix() -> str:
    return _script_hash_full()[:16]


def _variant_from_key(key: str) -> str:
    if '_' not in key:
        return key
    return key.split('_', 1)[1]


def _finite_metric(metrics: dict[str, Any], key: str) -> float | None:
    v = metrics.get(key)
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def _rows_for_resampled_clusters(
    valid_rows: list[dict],
    drawn_seeds: np.ndarray,
) -> list[dict]:
    """Include every shared-valid row for each seed draw; preserve bootstrap multiplicity."""
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in valid_rows:
        by_seed[int(row['seed'])].append(row)
    pool: list[dict] = []
    for seed in drawn_seeds:
        pool.extend(by_seed[int(seed)])
    return pool


def cluster_bootstrap_pooled_means(
    valid_rows: list[dict],
    metric_keys: list[str],
    *,
    seed: int = RESAMPLING_SEED,
    n_clusters: int = N_INDEPENDENT_SEED_CLUSTERS,
    n_replicates: int = N_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    cluster_ids = sorted({int(r['seed']) for r in valid_rows})
    if not cluster_ids or len(cluster_ids) != n_clusters:
        n_clusters = len(cluster_ids)
    rng = np.random.default_rng(seed)
    replicate_means: dict[str, list[float]] = {k: [] for k in metric_keys}
    replicate_paired_diffs: dict[str, list[float]] = {}
    primary_key = metric_keys[0]
    for baseline_key in metric_keys[1:]:
        label = baseline_key
        for bl_name, mk in STATE_RMSE_MISSING_BASELINES.items():
            if mk == baseline_key:
                label = f'paired_diff_{bl_name}'
                break
        replicate_paired_diffs[label] = []

    for _ in range(n_replicates):
        drawn = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        pool = _rows_for_resampled_clusters(valid_rows, drawn)
        for mk in metric_keys:
            vals = [_finite_metric(r['metrics'], mk) for r in pool]
            vals_f = [v for v in vals if v is not None]
            if vals_f:
                replicate_means[mk].append(float(np.mean(vals_f)))
        b_vals = [_finite_metric(r['metrics'], primary_key) for r in pool]
        for baseline_key in metric_keys[1:]:
            label = baseline_key
            for bl_name, mk in STATE_RMSE_MISSING_BASELINES.items():
                if mk == baseline_key:
                    label = f'paired_diff_{bl_name}'
                    break
            diffs = []
            for row in pool:
                b = _finite_metric(row['metrics'], primary_key)
                o = _finite_metric(row['metrics'], baseline_key)
                if b is not None and o is not None:
                    diffs.append(b - o)
            if diffs:
                replicate_paired_diffs[label].append(float(np.mean(diffs)))

    out: dict[str, Any] = {
        'resampling_seed': seed,
        'n_independent_seed_clusters': n_clusters,
        'n_bootstrap_replicates': n_replicates,
        'estimand': (
            'Pooled arithmetic mean of metric over all stress×seed cells in each bootstrap '
            'resample; clusters are independent seeds; all valid cells per drawn seed are kept '
            'and multiplicity when a seed is redrawn is preserved.'
        ),
        'primary_metric': primary_key,
        'limitations': (
            'Ten seed-level clusters only; inference is conditional on bounded optimal cells; '
            'not cell-level Bayesian inference. Prior analysis used 10 iterations and dropped '
            'multiplicity via set membership (post-hoc correction, not frozen at preregistration).'
        ),
    }
    sample_primary = [
        _finite_metric(r['metrics'], primary_key)
        for r in valid_rows
        if _finite_metric(r['metrics'], primary_key) is not None
    ]
    out['primary_mean_sample'] = float(np.mean(sample_primary)) if sample_primary else None
    primary_boot = replicate_means.get(primary_key, [])
    out['primary_mean_bootstrap'] = float(np.mean(primary_boot)) if primary_boot else None
    if primary_boot:
        out['primary_ci95_low'] = float(np.percentile(primary_boot, 2.5))
        out['primary_ci95_high'] = float(np.percentile(primary_boot, 97.5))
    for mk, series in replicate_means.items():
        if mk == primary_key:
            continue
        out[f'bootstrap_{mk}'] = {
            'mean_of_replicate_means': float(np.mean(series)) if series else None,
            'ci95_low': float(np.percentile(series, 2.5)) if series else None,
            'ci95_high': float(np.percentile(series, 97.5)) if series else None,
        }
    for label, series in replicate_paired_diffs.items():
        out[label] = {
            'mean_of_replicate_paired_means': float(np.mean(series)) if series else None,
            'ci95_low': float(np.percentile(series, 2.5)) if series else None,
            'ci95_high': float(np.percentile(series, 97.5)) if series else None,
        }
    return out


def _coverage_eligible(metrics: dict, field: str) -> bool:
    return isinstance(metrics.get(field), bool)


def reconcile(rows: list[dict]) -> dict[str, Any]:
    valid_rows = [r for r in rows if point_estimate_valid(r['metrics'])]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[_variant_from_key(row['scenario_key'])].append(row)

    variant_records = []
    sum_planned = 0
    sum_valid = 0
    sum_infeasible = 0
    for variant, subset in sorted(by_variant.items()):
        n_planned = len(subset)
        valid_b = [r for r in subset if point_estimate_valid(r['metrics'])]
        infeasible = [r for r in subset if r['metrics'].get('bounded_status') == 'infeasible_polytope']
        rec = {
            'stress_variant': variant,
            'n_planned': n_planned,
            'n_bounded_point_valid': len(valid_b),
            'n_infeasible_polytope': len(infeasible),
            'n_boundary_truth_in_interval': sum(1 for r in subset if r.get('boundary_truth_in_interval')),
            'n_joint_within_assumption': sum(
                1 for r in subset if (r.get('eval_truth_feasibility') or {}).get('joint_within_assumption')
            ),
            'n_initial_truth_in_polytope': sum(
                1 for r in subset if (r.get('failure_flags') or {}).get('initial_truth_in_polytope')
            ),
            'display_note_wrong_interval': (
                'Stress identifier wrong_interval: generative truth remains inside the stated '
                'boundary box (narrower truth-containing interval); not a misspecified empty set.'
                if variant == 'wrong_interval'
                else None
            ),
        }
        for field in COVERAGE_METRICS:
            elig = [r for r in subset if _coverage_eligible(r['metrics'], field)]
            covered = sum(1 for r in elig if r['metrics'].get(field))
            rec[f'{field}_n_eligible'] = len(elig)
            rec[f'{field}_n_covered'] = covered
        variant_records.append(rec)
        sum_planned += n_planned
        sum_valid += len(valid_b)
        sum_infeasible += len(infeasible)

    infeasible_all = [r for r in rows if r['metrics'].get('bounded_status') == 'infeasible_polytope']
    infeasible_by_variant = defaultdict(int)
    for r in infeasible_all:
        infeasible_by_variant[_variant_from_key(r['scenario_key'])] += 1

    totals = {
        'n_rows_matrix': len(rows),
        'n_bounded_point_valid': len(valid_rows),
        'n_infeasible_polytope': len(infeasible_all),
        'n_generative_capacity_violation': sum(
            1 for r in rows if (r.get('failure_flags') or {}).get('generative_capacity_violation')
        ),
        'n_boundary_truth_in_interval_rows': sum(1 for r in rows if r.get('boundary_truth_in_interval')),
        'n_variants_with_generative_truth_in_boundary_box': sum(
            1 for v, sub in by_variant.items() if all(r.get('boundary_truth_in_interval') for r in sub)
        ),
        'expected_infeasible_breakdown': {
            'truth_outside_interval': infeasible_by_variant.get('truth_outside_interval', 0),
            'intentional_boundary_inconsistency': infeasible_by_variant.get(
                'intentional_boundary_inconsistency', 0
            ),
            'gaussian_misspecification': infeasible_by_variant.get('gaussian_misspecification', 0),
            'sum_assert_infeasible': (
                infeasible_by_variant.get('truth_outside_interval', 0)
                + infeasible_by_variant.get('intentional_boundary_inconsistency', 0)
                + infeasible_by_variant.get('gaussian_misspecification', 0)
            ),
        },
        'feasibility_note': (
            'LP feasibility / optimal bounded point estimates do not imply the generative truth '
            'witness lies in the stated identification polytope; feasible alternative trajectories '
            'may exclude the true state trajectory.'
        ),
    }
    for field in COVERAGE_METRICS:
        elig_all = [r for r in rows if _coverage_eligible(r['metrics'], field)]
        cov_all = sum(1 for r in elig_all if r['metrics'].get(field))
        elig_valid = [r for r in valid_rows if _coverage_eligible(r['metrics'], field)]
        cov_valid = sum(1 for r in elig_valid if r['metrics'].get(field))
        elig_joint = [
            r
            for r in rows
            if (r.get('eval_truth_feasibility') or {}).get('joint_within_assumption')
            and _coverage_eligible(r['metrics'], field)
        ]
        cov_joint = sum(1 for r in elig_joint if r['metrics'].get(field))
        totals[field] = {
            'unconditional_matrix': {'n_eligible': len(elig_all), 'n_covered': cov_all},
            'conditional_on_bounded_point_valid': {'n_eligible': len(elig_valid), 'n_covered': cov_valid},
            'conditional_on_joint_within_assumption': {'n_eligible': len(elig_joint), 'n_covered': cov_joint},
        }

    sum_assertions = {
        'sum_variant_n_planned_eq_matrix_rows': sum_planned == len(rows),
        'sum_variant_valid_eq_matrix_valid': sum_valid == len(valid_rows),
        'sum_variant_infeasible_eq_matrix_infeasible': sum_infeasible == len(infeasible_all),
        'infeasible_breakdown_eq_total_infeasible': totals['expected_infeasible_breakdown'][
            'sum_assert_infeasible'
        ]
        == len(infeasible_all),
        'boundary_truth_in_interval_leq_rows': totals['n_boundary_truth_in_interval_rows'] <= len(rows),
    }

    return {
        'per_variant': variant_records,
        'totals': totals,
        'sum_assertions': sum_assertions,
        'all_sum_assertions_pass': all(sum_assertions.values()),
    }


def _mean_on_rows(rows: list[dict], metric_key: str) -> dict[str, Any]:
    vals = [_finite_metric(r['metrics'], metric_key) for r in rows]
    finite = [v for v in vals if v is not None]
    return {
        'metric': metric_key,
        'n_finite': len(finite),
        'mean': float(np.mean(finite)) if finite else None,
    }


def analyze(payload: dict) -> dict:
    rows = payload['rows']
    valid_rows = [r for r in rows if point_estimate_valid(r['metrics'])]
    reconciliation = reconcile(rows)

    variant_tables = []
    for rec in reconciliation['per_variant']:
        subset = [r for r in rows if _variant_from_key(r['scenario_key']) == rec['stress_variant']]
        valid_bounded = [r for r in subset if point_estimate_valid(r['metrics'])]
        prim = [
            _finite_metric(r['metrics'], PRIMARY)
            for r in valid_bounded
            if _finite_metric(r['metrics'], PRIMARY) is not None
        ]
        entry = {
            **rec,
            'primary_metric': PRIMARY,
            'primary_metric_denom_valid_bounded': len(valid_bounded),
            'primary_mean_conditional': float(np.mean(prim)) if prim else None,
            'primary_std_conditional': float(np.std(prim)) if prim else None,
        }
        variant_tables.append(entry)

    paired_shared = {}
    paired_full_matrix = {}
    for label, metric_key in STATE_RMSE_MISSING_BASELINES.items():
        paired_shared[label] = {
            **_mean_on_rows(valid_rows, metric_key),
            'denominator_label': 'shared_valid_bounded_optimal',
            'n_rows': len(valid_rows),
            'note': 'Conditional on bounded_status optimal and primal violation <= 1e-5.',
        }
        if label == 'bounded_deployable':
            full_rows = valid_rows
        else:
            full_rows = rows
        paired_full_matrix[label] = {
            **_mean_on_rows(full_rows, metric_key),
            'denominator_label': 'all_matrix_rows_finite_metric' if label != 'bounded_deployable' else 'shared_valid_only',
        }

    paired_diffs_shared = {}
    for label, metric_key in STATE_RMSE_MISSING_BASELINES.items():
        if label == 'bounded_deployable':
            continue
        diffs = []
        for row in valid_rows:
            b = _finite_metric(row['metrics'], PRIMARY)
            o = _finite_metric(row['metrics'], metric_key)
            if b is not None and o is not None:
                diffs.append(b - o)
        paired_diffs_shared[label] = {
            'metric_baseline': metric_key,
            'n_shared_valid': len(diffs),
            'mean_bounded_minus_baseline': float(np.mean(diffs)) if diffs else None,
        }

    slice_comparison = {}
    for suffix in RMSE_SLICE_SUFFIXES:
        slice_comparison[suffix] = {}
        for label, prefix in [
            ('bounded', 'bounded_state_rmse'),
            ('masked_mean', 'masked_mean_state_rmse'),
            ('forward_fill', 'forward_fill_state_rmse'),
        ]:
            key = f'{prefix}_{suffix}'
            slice_comparison[suffix][label] = _mean_on_rows(valid_rows, key)

    boot_keys = [PRIMARY] + [mk for name, mk in STATE_RMSE_MISSING_BASELINES.items() if name != 'bounded_deployable']
    uncertainty = cluster_bootstrap_pooled_means(valid_rows, boot_keys)

    od_nnls_note = {
        'metric': OD_NNLS_METRIC,
        'benchmark_valid_for_comparison': False,
        'erratum': OD_NNLS_BENCHMARK_INVALID,
        'raw_matrix_mean_preserved_for_audit': _mean_on_rows(rows, OD_NNLS_METRIC),
    }

    partial_cov = []
    for row in rows:
        diag = row.get('observability') or {}
        partial_cov.append({
            'seed': row['seed'],
            'variant': _variant_from_key(row['scenario_key']),
            'path_rank_observed': diag.get('path_rank_observed'),
            'truth_assumption_flags': row.get('eval_truth_feasibility'),
        })

    oracle_note = {m: 'diagnostic_only_not_deployable' for m in ORACLE_METRICS}

    accuracy_finding = {
        'summary': (
            'On shared-valid cells (n=81), masked-mean missing RMSE is lower than deployable '
            'bounded missing RMSE; bounded does not beat simple imputation baselines on this metric. '
            'No state-of-the-art or superiority claim is supported.'
        ),
        'bounded_minus_masked_mean_missing': paired_diffs_shared.get('masked_mean', {}).get(
            'mean_bounded_minus_baseline'
        ),
        'consistency_only_evidence': (
            'Interval tightening / tube coverage statistics are descriptive of stated assumptions; '
            'see reconciliation.totals coverage blocks with explicit denominators.'
        ),
    }

    return {
        'receipt_type': 'crest_research_final_analysis_v4',
        'analysis_revision': 'posthoc_bootstrap_and_reconciliation_v1',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'analysis_script': str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        'analysis_script_sha256_prefix': _script_hash_prefix(),
        'inference_source_hash_at_analysis': research_code_content_hash(REPO_ROOT),
        'immutable_raw_inputs_note': (
            'Metrics recomputed from frozen pilot_results.json rows only; no scenario rerun.'
        ),
        'variant_denominators': variant_tables,
        'denominator_reconciliation': reconciliation,
        'paired_conditional_comparisons_shared_valid': paired_shared,
        'paired_full_matrix_baselines': paired_full_matrix,
        'paired_diff_bounded_minus_baseline_shared_valid': paired_diffs_shared,
        'rmse_slice_comparison_shared_valid': slice_comparison,
        'uncertainty_bootstrap': uncertainty,
        'od_nnls_benchmark_status': od_nnls_note,
        'partial_identification_coverage_note': (
            'Coverage columns use distinct eligibility: unconditional matrix rows with boolean field, '
            'conditional on bounded point valid, or conditional on joint_within_assumption. '
            'LP infeasible rows omit lp_data coverage booleans.'
        ),
        'partial_identification_rows': len(partial_cov),
        'partial_identification_row_index': partial_cov,
        'oracle_metrics': oracle_note,
        'accuracy_finding': accuracy_finding,
        'matrix_summary': payload.get('matrix_performance_summary'),
    }


def write_analysis_csv(report: dict, path: Path) -> None:
    rows_out: list[dict[str, Any]] = []
    for v in report['variant_denominators']:
        rows_out.append({'section': 'variant', **v})
    for label, block in report['paired_conditional_comparisons_shared_valid'].items():
        rows_out.append({'section': 'paired_shared_valid', 'baseline': label, **block})
    with path.open('w', newline='', encoding='utf-8') as fh:
        if not rows_out:
            return
        fields = sorted({k for r in rows_out for k in r})
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows_out)


def write_analysis_receipt(
    report: dict,
    *,
    receipt_path: Path,
    prior_script_sha256: str | None,
    raw_sha256: dict[str, str],
) -> None:
    receipt = {
        'receipt_type': 'crest_research_final_analysis_correction',
        'correction_utc': report['generated_utc'],
        'prior_analysis_script_sha256': prior_script_sha256,
        'corrected_analysis_script_sha256': _script_hash_full(),
        'inference_source_hash': report['inference_source_hash_at_analysis'],
        'immutable_input_sha256': raw_sha256,
        'immutable_verified_unchanged': True,
        'posthoc_note': (
            'Bootstrap and reconciliation corrected after final v4 execution; prior bootstrap loop '
            'was not part of frozen inference hash c15d37e54fe38d63.'
        ),
        'outputs': {
            'final_analysis_json': 'results/research_final_v4/final_analysis.json',
            'final_analysis_csv': 'results/research_final_v4/final_analysis.csv',
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')


def immutable_raw_hashes(repo_root: Path) -> dict[str, str]:
    paths = {
        'pilot_results.json': repo_root / 'results' / 'research_final_v4' / 'pilot_results.json',
        'pilot_metrics.csv': repo_root / 'results' / 'research_final_v4' / 'pilot_metrics.csv',
        'final_receipt.json': repo_root / 'results' / 'research_final_v4' / 'final_receipt.json',
        'pre_evaluation_freeze_v4.json': repo_root / 'results' / 'research' / 'pre_evaluation_freeze_v4.json',
        'registry': repo_root / '.crest_research_protocol_registry.json',
    }
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items() if p.exists()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze final holdout v4 artifacts')
    parser.add_argument(
        '--input',
        type=Path,
        default=REPO_ROOT / 'results' / 'research_final_v4' / 'pilot_results.json',
    )
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'research_final_v4')
    parser.add_argument(
        '--prior-script-sha256',
        type=str,
        default=None,
        help='Archived pre-correction script digest for analysis_receipt.json',
    )
    args = parser.parse_args(argv)
    pre_hashes = immutable_raw_hashes(REPO_ROOT)
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    report = analyze(payload)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'final_analysis.json'
    csv_path = out_dir / 'final_analysis.csv'
    receipt_path = out_dir / 'analysis_receipt.json'

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_analysis_csv(report, csv_path)
    post_hashes = immutable_raw_hashes(REPO_ROOT)
    if pre_hashes != post_hashes:
        raise SystemExit('Immutable raw artifacts changed during analysis; aborting.')
    write_analysis_receipt(
        report,
        receipt_path=receipt_path,
        prior_script_sha256=args.prior_script_sha256,
        raw_sha256=post_hashes,
    )

    try:
        import matplotlib.pyplot as plt

        variants = report['variant_denominators']
        labels = [v['stress_variant'] for v in variants]
        means = [v.get('primary_mean_conditional') or float('nan') for v in variants]
        ns = [v['n_bounded_point_valid'] for v in variants]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(labels))
        ax.bar(x, means, color='#4a6fa5')
        for i, n in enumerate(ns):
            y = means[i] if np.isfinite(means[i]) else 0
            ax.text(i, y, f'n={n}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'{PRIMARY} (conditional mean)')
        ax.set_title('Final holdout v4: primary metric by stress variant (valid bounded n)')
        fig.tight_layout()
        fig_path = out_dir / 'final_primary_by_variant.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        report['figure'] = str(fig_path.relative_to(REPO_ROOT))
    except ImportError:
        report['figure'] = None

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(
        json.dumps(
            {
                'wrote': str(json_path),
                'n_variants': len(report['variant_denominators']),
                'sum_assertions_pass': report['denominator_reconciliation']['all_sum_assertions_pass'],
            },
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
