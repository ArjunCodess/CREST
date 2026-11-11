"""Post-hoc analysis for amended final holdout v4 (outside frozen inference source hash at record time)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.research.config_paths import REPO_ROOT
from src.research.experiments import ORACLE_METRICS, POINT_ESTIMATE_METRIC_PREFIX, point_estimate_valid
from src.research.provenance import research_code_content_hash

RESAMPLING_SEED = 424242
N_CLUSTERS = 10
PRIMARY = 'bounded_state_rmse_missing'
STATE_RMSE_MISSING_BASELINES = {
    'bounded_deployable': 'bounded_state_rmse_missing',
    'masked_mean': 'masked_mean_state_rmse_missing',
    'forward_fill': 'forward_fill_state_rmse_missing',
    'interval_midpoint': 'midpoint_state_rmse_missing',
    'oracle_diagnostic': 'oracle_state_rmse_missing',
}
OD_AGGREGATE = 'od_nnls_total_error'


def _script_hash() -> str:
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _variant_from_key(key: str) -> str:
    if '_' not in key:
        return key
    return key.split('_', 1)[1]


def analyze(payload: dict) -> dict:
    rows = payload['rows']
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[_variant_from_key(row['scenario_key'])].append(row)

    variant_tables = []
    for variant, subset in sorted(by_variant.items()):
        n_planned = len(subset)
        valid_bounded = [r for r in subset if point_estimate_valid(r['metrics'])]
        infeasible = [
            r for r in subset if r['metrics'].get('bounded_status') == 'infeasible_polytope'
        ]
        failed = [r for r in subset if r['metrics'].get('bounded_status') == 'numerical_failure']
        gen_bad = [r for r in subset if r['failure_flags'].get('generative_capacity_violation')]
        entry = {
            'stress_variant': variant,
            'n_planned': n_planned,
            'n_completed': n_planned,
            'n_bounded_point_valid': len(valid_bounded),
            'n_infeasible_polytope': len(infeasible),
            'n_numerical_failure': len(failed),
            'n_generative_capacity_violation': len(gen_bad),
            'primary_metric': PRIMARY,
            'primary_metric_denom_valid_bounded': len(valid_bounded),
        }
        if valid_bounded:
            prim = [
                r['metrics'][PRIMARY]
                for r in valid_bounded
                if isinstance(r['metrics'].get(PRIMARY), (int, float)) and np.isfinite(r['metrics'][PRIMARY])
            ]
            entry['primary_mean_conditional'] = float(np.mean(prim)) if prim else None
            entry['primary_std_conditional'] = float(np.std(prim)) if prim else None
        variant_tables.append(entry)

    paired = {}
    valid_rows = [r for r in rows if point_estimate_valid(r['metrics'])]
    seeds_valid = {r['seed'] for r in valid_rows}
    for label, metric_key in STATE_RMSE_MISSING_BASELINES.items():
        vals = []
        for r in valid_rows:
            m = r['metrics'].get(metric_key)
            if isinstance(m, (int, float)) and np.isfinite(m):
                vals.append(float(m))
        paired[label] = {
            'metric': metric_key,
            'n_shared_valid': len(vals),
            'mean_conditional': float(np.mean(vals)) if vals else None,
            'note': 'Conditional on bounded optimal + primal <= 1e-5; infeasible rows excluded.',
        }

    rng = np.random.default_rng(RESAMPLING_SEED)
    cluster_ids = sorted(seeds_valid)
    if cluster_ids:
        clusters = {s: i % N_CLUSTERS for i, s in enumerate(cluster_ids)}
        boot_means = []
        for _ in range(N_CLUSTERS):
            draw = rng.choice(N_CLUSTERS, size=N_CLUSTERS, replace=True)
            picked = [s for s in cluster_ids if clusters[s] in draw]
            if not picked:
                continue
            vals = [
                r['metrics'][PRIMARY]
                for r in valid_rows
                if r['seed'] in picked
                and isinstance(r['metrics'].get(PRIMARY), (int, float))
                and np.isfinite(r['metrics'][PRIMARY])
            ]
            if vals:
                boot_means.append(float(np.mean(vals)))
        uncertainty = {
            'resampling_seed': RESAMPLING_SEED,
            'n_clusters': N_CLUSTERS,
            'n_bootstrap_draws': len(boot_means),
            'primary_mean_bootstrap': float(np.mean(boot_means)) if boot_means else None,
            'primary_ci95_low': float(np.percentile(boot_means, 2.5)) if boot_means else None,
            'primary_ci95_high': float(np.percentile(boot_means, 97.5)) if boot_means else None,
            'precision_limitation': 'Cluster bootstrap at independent seed level; not cell-level inference.',
        }
    else:
        uncertainty = {
            'resampling_seed': RESAMPLING_SEED,
            'n_clusters': N_CLUSTERS,
            'n_bootstrap_draws': 0,
            'precision_limitation': 'No valid bounded rows; bootstrap skipped.',
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

    return {
        'receipt_type': 'crest_research_final_analysis_v4',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'analysis_script': str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        'analysis_script_sha256_prefix': _script_hash(),
        'inference_source_hash_at_analysis': research_code_content_hash(REPO_ROOT),
        'variant_denominators': variant_tables,
        'paired_conditional_comparisons': paired,
        'uncertainty_bootstrap': uncertainty,
        'partial_identification_coverage_note': (
            'Coverage widths conditional on stated truth-assumption flags; excluded when LP infeasible.'
        ),
        'partial_identification_rows': len(partial_cov),
        'oracle_metrics': oracle_note,
        'matrix_summary': payload.get('matrix_performance_summary'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze final holdout v4 artifacts')
    parser.add_argument('--input', type=Path, default=REPO_ROOT / 'results' / 'research_final_v4' / 'pilot_results.json')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'research_final_v4')
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    report = analyze(payload)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'final_analysis.json'
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

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
            ax.text(i, means[i] if np.isfinite(means[i]) else 0, f'n={n}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'{PRIMARY} (conditional mean)')
        ax.set_title('Final holdout v4: primary metric by stress variant')
        fig.tight_layout()
        fig_path = out_dir / 'final_primary_by_variant.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        report['figure'] = str(fig_path.relative_to(REPO_ROOT))
    except ImportError:
        report['figure'] = None

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'wrote': str(json_path), 'n_variants': len(report['variant_denominators'])}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
