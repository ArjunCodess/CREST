"""Scenario matrix, paired seed summaries, and artifact export."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import baselines as bl
from .boundary_conservation import boundary_aggregate_lp_bundle, conservation_lp_box_only, conservation_lp_extrema
from .conservation_constraints import boundary_aggregate_weights
from .metrics import (
    bound_coverage,
    dynamics_balance_rmse,
    flow_rmse,
    od_aggregate_error,
    physical_recovery_errors,
    state_rmse,
    timed,
)
from .partial_identification import observability_diagnostics
from .provenance import research_code_content_hash
from .stress_variants import STRESS_VARIANTS
from .synthetic_sim import generate_scenario
from .truth_feasibility import classify_bounded_lp_status, classify_od_lp_status, evaluate_truth_feasibility

PRIMAL_OK_TOL = 1e-5
POINT_ESTIMATE_METRIC_PREFIX = 'bounded_'
POINT_ESTIMATE_STATUS_KEYS = frozenset({'bounded_status', 'bounded_primal_violation'})
BOOLEAN_COVERAGE_METRICS = frozenset({
    'boundary_interval_covers_truth_net_vehicles',
    'lp_data_covers_truth_net_vehicles',
    'lp_box_covers_truth_net_vehicles',
    'lp_data_covers_truth_total_vehicles',
    'lp_box_covers_truth_total_vehicles',
    'od_lp_covers_true_total',
})
ORACLE_METRICS = frozenset({'oracle_state_rmse_all', 'oracle_state_rmse_observed', 'oracle_state_rmse_missing'})

SUPERSEDED_ARTIFACT_NOTE = (
    'Prior artifacts using Gaussian storage noise with hard 2.5σ tubes (joint misspecification) '
    'and OD LPs without declared path caps are superseded. Baseline contract: uniform_bounded noise '
    'with shared obs_tolerance; Gaussian and boundary/count stresses are named variants only.'
)


def source_provenance(repo_root: Path, code_hash: str | None = None) -> dict[str, Any]:
    if code_hash is None:
        code_hash = research_code_content_hash(repo_root)
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        rev, dirty = None, None
    return {
        'git_revision': rev,
        'git_dirty': bool(dirty),
        'research_code_hash': code_hash,
        'supersedes_note': SUPERSEDED_ARTIFACT_NOTE,
        'embargo_note': __import__('src.research.protocol', fromlist=['embargo_note']).embargo_note(),
    }


def point_estimate_valid(metrics: dict[str, Any]) -> bool:
    if metrics.get('bounded_status') != 'optimal':
        return False
    viol = metrics.get('bounded_primal_violation')
    if viol is None or not np.isfinite(viol):
        return False
    return float(viol) <= PRIMAL_OK_TOL


def _null_invalid_point_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    if point_estimate_valid(metrics):
        return out
    for key in list(out):
        if key.startswith(POINT_ESTIMATE_METRIC_PREFIX) and key not in POINT_ESTIMATE_STATUS_KEYS:
            out[key] = None
    return out


def run_one_scenario(scenario_key: str, seed: int, params: dict) -> dict[str, Any]:
    kwargs = {**params, 'seed': seed}
    scenario = generate_scenario(**kwargs)
    mask = scenario.obs_mask
    obs_tol = scenario.metadata.get('obs_tolerance', 25.0)
    z_upper = scenario.metadata.get('max_edge_flow_vph', scenario.metadata['demand_vph'] * 1.25)
    z_seed = np.full_like(scenario.z_true, 0.5 * scenario.metadata['demand_vph'])

    imputed, t_impute = timed(bl.masked_mean_imputer, scenario.obs_values, mask)
    ff, t_ff = timed(bl.unconstrained_forward_fill, scenario.obs_values, mask)
    bounded, t_bnd = timed(bl.bounded_boundary_estimate, scenario)
    midpoint = bl.assumed_exact_boundary_midpoint(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        z_seed,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
    )
    oracle = bl.oracle_exact_boundary(
        scenario.B, scenario.R, scenario.dt_hours, scenario.n0, scenario.z_true, scenario.u_true
    )

    interval = bl.interval_baseline_aggregate(
        scenario.u_interval_lower, scenario.u_interval_upper, scenario.dt_hours
    )
    prior_scale = np.full(scenario.incidence.shape[1], 100.0)
    count_center = 0.5 * (np.nan_to_num(scenario.count_lower, 0) + np.nan_to_num(scenario.count_upper, 0))
    od_flows = bl.od_nnls_baseline(scenario.incidence, count_center, prior_scale)
    od_lp = bl.od_lp_bounds_baseline(
        scenario.incidence,
        scenario.count_lower,
        scenario.count_upper,
        np.ones(scenario.incidence.shape[1]),
        prior_scale,
        observed_mask=scenario.count_observed_mask,
        path_flow_cap=scenario.path_flow_cap,
    )

    truth_eval = evaluate_truth_feasibility(scenario)
    lp_diag = classify_bounded_lp_status(
        scenario,
        bounded.status,
        bounded.primal_feasibility_max_violation,
    )
    od_status = od_lp['aggregate_bounds'].get('status', 'unknown')
    od_reason = classify_od_lp_status(scenario, od_status, message=od_lp['aggregate_bounds'].get('message'))

    diag = observability_diagnostics(
        scenario.incidence,
        scenario.count_observed_mask,
    )

    Tm1 = scenario.u_interval_lower.shape[0]
    n_bound = scenario.R.shape[1]
    true_bnd = bl.boundary_truth_vehicle_totals(scenario.u_true, scenario.dt_hours)
    lp_kw = dict(
        obs_mask=scenario.obs_mask,
        obs_values=np.nan_to_num(scenario.obs_values, nan=0.0),
        obs_tolerance=obs_tol,
        z_upper=z_upper,
        capacity=scenario.capacity,
    )
    bnd_lp = boundary_aggregate_lp_bundle(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        include_observations=True,
        **lp_kw,
    )
    bnd_lp_box = boundary_aggregate_lp_bundle(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        z_upper=z_upper,
        capacity=scenario.capacity,
        include_observations=False,
    )
    net_w = boundary_aggregate_weights(Tm1, n_bound, mode='net')
    bnd_bounds = conservation_lp_extrema(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        net_w,
        include_observations=True,
        **lp_kw,
    )
    bnd_box = conservation_lp_box_only(
        scenario.B,
        scenario.R,
        scenario.dt_hours,
        scenario.n0,
        scenario.u_interval_lower,
        scenario.u_interval_upper,
        net_w,
        z_upper=z_upper,
        capacity=scenario.capacity,
    )

    bounded_phys = physical_recovery_errors(
        bounded.storage,
        scenario.storage_true,
        bounded.internal_flow,
        scenario.z_true,
        bounded.boundary_flow,
        scenario.u_true,
    )
    midpoint_phys = physical_recovery_errors(
        midpoint.storage,
        scenario.storage_true,
        midpoint.internal_flow,
        scenario.z_true,
        midpoint.boundary_flow,
        scenario.u_true,
    )

    cap_ok = bool(np.all(bounded.storage <= scenario.capacity[:, None].T + 1e-3))

    def score_all(pred_storage: np.ndarray) -> dict[str, float]:
        return {
            'state_rmse_observed': state_rmse(pred_storage, scenario.storage_true, mask, region='observed'),
            'state_rmse_missing': state_rmse(pred_storage, scenario.storage_true, mask, region='missing'),
            'state_rmse_all': state_rmse(pred_storage, scenario.storage_true, mask, region='all'),
        }

    impute_scores = score_all(imputed)
    ff_scores = score_all(ff)
    bounded_scores = score_all(bounded.storage)
    midpoint_scores = score_all(midpoint.storage)
    oracle_scores = score_all(oracle.storage)

    raw_metrics = {
            **{f'masked_mean_{k}': v for k, v in impute_scores.items()},
            **{f'forward_fill_{k}': v for k, v in ff_scores.items()},
            **{f'bounded_{k}': v for k, v in bounded_scores.items()},
            **{f'midpoint_{k}': v for k, v in midpoint_scores.items()},
            **{f'oracle_{k}': v for k, v in oracle_scores.items()},
            'bounded_boundary_flow_vs_truth_rmse': bounded_phys['boundary_flow_vs_truth_rmse'],
            'bounded_storage_vs_truth_rmse': bounded_phys['storage_vs_truth_rmse'],
            'bounded_dynamics_algebraic_rmse': bounded.dynamics_algebraic_residual_rmse,
            'bounded_dynamics_balance_rmse': dynamics_balance_rmse(
                bounded.storage, scenario.B, scenario.R, bounded.internal_flow, bounded.boundary_flow, scenario.dt_hours
            ),
            'midpoint_storage_vs_truth_rmse': midpoint_phys['storage_vs_truth_rmse'],
            'oracle_storage_vs_truth_rmse': oracle_scores['state_rmse_all'],
            'capacity_satisfied_bounded': cap_ok,
            'true_net_inflow_vehicles': true_bnd['net_inflow_vehicles'],
            'true_inflow_vehicles': true_bnd['inflow_vehicles'],
            'true_outflow_vehicles': true_bnd['outflow_vehicles'],
            'corridor_total_storage_change_vehicles': float(
                scenario.storage_true[-1].sum() - scenario.storage_true[0].sum()
            ),
            'boundary_interval_covers_truth_net_vehicles': bound_coverage(
                true_bnd['net_inflow_vehicles'],
                interval['net'].get('lower'),
                interval['net'].get('upper'),
            ),
            'lp_data_covers_truth_net_vehicles': bound_coverage(
                true_bnd['net_inflow_vehicles'], bnd_bounds.get('lower'), bnd_bounds.get('upper')
            ),
            'lp_box_covers_truth_net_vehicles': bound_coverage(
                true_bnd['net_inflow_vehicles'], bnd_box.get('lower'), bnd_box.get('upper')
            ),
            'lp_data_net_width_vehicles': bnd_bounds.get('width'),
            'lp_box_net_width_vehicles': bnd_box.get('width'),
            'lp_data_inflow_width_vehicles': (bnd_lp.get('inflow') or {}).get('width'),
            'lp_data_outflow_width_vehicles': (bnd_lp.get('outflow') or {}).get('width'),
            'lp_data_width_vehicles': bnd_bounds.get('width'),
            'lp_box_width_vehicles': bnd_box.get('width'),
            'od_nnls_total_error': od_aggregate_error(float(np.sum(od_flows)), float(np.sum(scenario.path_flow_true))),
            'od_lp_aggregate_width': od_lp['aggregate_bounds'].get('width'),
            'od_lp_covers_true_total': bound_coverage(
                float(np.sum(scenario.path_flow_true)),
                od_lp['aggregate_bounds'].get('lower'),
                od_lp['aggregate_bounds'].get('upper'),
            ),
            'bounded_status': bounded.status,
            'bounded_primal_violation': bounded.primal_feasibility_max_violation,
            'runtime_imputer_s': t_impute,
            'runtime_bounded_s': t_bnd,
    }
    metrics = _null_invalid_point_metrics(raw_metrics)
    record = {
        'scenario_key': scenario_key,
        'seed': seed,
        'metadata': scenario.metadata,
        'boundary_truth_in_interval': scenario.boundary_truth_in_interval(),
        'truth_respects_capacity': scenario.truth_respects_capacity(),
        'observability': diag.to_dict(),
        'metrics': metrics,
        'eval_truth_feasibility': truth_eval['flags'],
        'truth_witness': {
            'path_flow_cap': truth_eval.get('truth_witness_path_flow_cap'),
            'total_path_flow': truth_eval.get('truth_witness_total_flow'),
        },
        'failure_flags': {
            'bounded_infeasible_or_failed': not point_estimate_valid(metrics),
            'bounded_optimizer_status': metrics.get('bounded_status'),
            'bounded_lp_status_reason': lp_diag.get('lp_status_reason'),
            'initial_truth_in_polytope': lp_diag.get('initial_truth_in_polytope'),
            'truth_outside_stated_interval': not scenario.boundary_truth_in_interval(),
            'od_lp_status': od_status,
            'od_lp_status_reason': od_reason,
            'od_lp_failed': od_status not in ('optimal', 'no_count_constraints'),
            'generative_capacity_violation': not scenario.truth_respects_capacity(),
            'joint_within_assumption': truth_eval['flags'].get('joint_within_assumption'),
        },
    }
    return record


def summarize_matrix(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    valid = [r for r in rows if point_estimate_valid(r['metrics'])]
    failed = [r for r in rows if r['metrics'].get('bounded_status') == 'numerical_failure']
    infeasible = [r for r in rows if r['metrics'].get('bounded_status') == 'infeasible_polytope']
    return {
        'n_rows': n,
        'n_point_estimate_valid': len(valid),
        'n_point_estimate_failed': len(failed),
        'n_point_estimate_infeasible_polytope': len(infeasible),
        'n_point_estimate_other_nonoptimal': n - len(valid) - len(failed) - len(infeasible),
        'point_estimate_failure_rate': (n - len(valid)) / n if n else None,
        'paired_comparison_note': (
            'Conditional means use identical seeds where bounded_status is optimal and '
            'primal violation <= 1e-5; failures are excluded, not imputed.'
        ),
    }


def summarize_by_scenario(rows: list[dict]) -> list[dict]:
    keys = sorted({r['scenario_key'] for r in rows})
    out = []
    for key in keys:
        subset = [r for r in rows if r['scenario_key'] == key]
        metric_names = subset[0]['metrics'].keys()
        n_total = len(subset)
        valid_rows = [r for r in subset if point_estimate_valid(r['metrics'])]
        n_valid = len(valid_rows)
        n_failed = sum(1 for r in subset if r['metrics'].get('bounded_status') == 'numerical_failure')
        summary = {
            'scenario_key': key,
            'n_rows': n_total,
            'n_point_estimate_valid': n_valid,
            'n_point_estimate_failed': n_failed,
            'n_point_estimate_ineligible': n_total - n_valid,
            'point_estimate_failure_rate': (n_total - n_valid) / n_total if n_total else None,
        }
        for name in metric_names:
            if name in BOOLEAN_COVERAGE_METRICS or name.endswith('_covers_truth_net_vehicles'):
                eligible = [r['metrics'][name] for r in subset if r['metrics'][name] is not None]
                covered = [v for v in eligible if v is True]
                summary[f'{name}_n_eligible'] = len(eligible)
                summary[f'{name}_n_covered'] = len(covered)
                summary[f'{name}_coverage_rate'] = (len(covered) / len(eligible)) if eligible else None
                continue
            if name.startswith(POINT_ESTIMATE_METRIC_PREFIX):
                if name in POINT_ESTIMATE_STATUS_KEYS:
                    continue
                vals = [
                    r['metrics'][name]
                    for r in valid_rows
                    if isinstance(r['metrics'][name], (int, float)) and np.isfinite(r['metrics'][name])
                ]
            else:
                vals = [
                    r['metrics'][name]
                    for r in subset
                    if isinstance(r['metrics'][name], (int, float)) and np.isfinite(r['metrics'][name])
                ]
            if vals:
                if name.startswith(POINT_ESTIMATE_METRIC_PREFIX):
                    summary[f'{name}_mean_conditional'] = float(np.mean(vals))
                    summary[f'{name}_std_conditional'] = float(np.std(vals))
                    summary[f'{name}_n_finite_conditional'] = len(vals)
                else:
                    summary[f'{name}_mean'] = float(np.mean(vals))
                    summary[f'{name}_std'] = float(np.std(vals))
                    if name in ORACLE_METRICS:
                        summary[f'{name}_n_finite'] = len(vals)
        if valid_rows:
            paired_oracle = [
                float(r['metrics']['oracle_state_rmse_all'])
                for r in valid_rows
                if isinstance(r['metrics'].get('oracle_state_rmse_all'), (int, float))
                and np.isfinite(r['metrics']['oracle_state_rmse_all'])
            ]
            paired_bounded = [
                float(r['metrics']['bounded_state_rmse_all'])
                for r in valid_rows
                if isinstance(r['metrics'].get('bounded_state_rmse_all'), (int, float))
                and np.isfinite(r['metrics']['bounded_state_rmse_all'])
            ]
            summary['paired_oracle_bounded_n'] = min(len(paired_oracle), len(paired_bounded))
            if paired_oracle:
                summary['oracle_state_rmse_all_mean_on_valid_bounded_subset'] = float(np.mean(paired_oracle))
            if paired_bounded and 'bounded_state_rmse_all_mean_conditional' not in summary:
                summary['bounded_state_rmse_all_mean_conditional'] = float(np.mean(paired_bounded))
        out.append(summary)
    return out


def run_experiment_matrix(config: dict, repo_root: Path, *, code_hash: str | None = None) -> dict[str, Any]:
    seeds = list(config['seeds'])
    scenario_set = config.get('scenario_set', 'development')
    base_params = dict(config.get('scenario_params', {}))
    variants = config.get('stress_variants') or list(STRESS_VARIANTS.keys())
    rows = []
    for variant in variants:
        if variant not in STRESS_VARIANTS:
            raise ValueError(f'unknown stress variant {variant}')
        params = {**base_params, **STRESS_VARIANTS[variant]}
        for seed in seeds:
            rows.append(run_one_scenario(f'{scenario_set}_{variant}', seed, params))
    clean_config = {k: v for k, v in config.items() if not str(k).startswith('_')}
    return {
        'provenance': source_provenance(repo_root, code_hash),
        'config': clean_config,
        'rows': rows,
        'matrix_performance_summary': summarize_matrix(rows),
        'scenario_summaries': summarize_by_scenario(rows),
    }


def write_artifacts(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'pilot_results.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    csv_path = output_dir / 'pilot_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scenario_key', 'seed', 'bounded_state_rmse_all', 'oracle_state_rmse_all',
            'lp_data_width_vehicles', 'lp_box_width_vehicles', 'od_nnls_total_error',
            'truth_outside_stated_interval', 'capacity_satisfied_bounded',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload['rows']:
            writer.writerow({
                'scenario_key': row['scenario_key'],
                'seed': row['seed'],
                'bounded_state_rmse_all': row['metrics']['bounded_state_rmse_all'],
                'oracle_state_rmse_all': row['metrics']['oracle_state_rmse_all'],
                'lp_data_width_vehicles': row['metrics']['lp_data_width_vehicles'],
                'lp_box_width_vehicles': row['metrics']['lp_box_width_vehicles'],
                'od_nnls_total_error': row['metrics']['od_nnls_total_error'],
                'truth_outside_stated_interval': row['failure_flags']['truth_outside_stated_interval'],
                'capacity_satisfied_bounded': row['metrics']['capacity_satisfied_bounded'],
            })


def plot_diagnostics(payload: dict, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    summaries = payload['scenario_summaries']
    labels = [s['scenario_key'].split('_', 1)[-1] for s in summaries]
    bounded = [s.get('bounded_state_rmse_all_mean_conditional', np.nan) for s in summaries]
    oracle = [s.get('oracle_state_rmse_all_mean_on_valid_bounded_subset', np.nan) for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, bounded, w, label='bounded boundary (deployable)')
    ax.bar(x + w / 2, oracle, w, label='oracle exact boundary (diagnostic)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('state RMSE all cells (vehicles)')
    ax.set_title('Synthetic pilot: conditional RMSE (valid bounded subset only)')
    ax.legend()
    fig.tight_layout()
    path = output_dir / 'pilot_state_rmse.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
