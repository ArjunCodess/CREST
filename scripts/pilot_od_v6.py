"""Train-split OD pilot. Writes metrics only; it does not change a production solver."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.od_v6 import (
    BASELINE_REGULARIZATION,
    BASELINE_WEIGHT,
    DELTA_TOTAL_PER_0_01_S_OD,
    HELD_LINK_GAIN_REQUIRED,
    IDENTIFICATION,
    LAMBDA_GRID,
    REQUIRED_MECHANISMS,
    S_OD_COEFFICIENT_IN_TOTAL,
    SYNTHETIC_REGRESSION_TOLERANCE,
    TRAIN_SPLIT,
    SolverError,
    load_train_panel,
    nested_link_assessment,
    promotion_gate,
    summarize_synthetic,
    synthetic_recovery,
)


def jsonify(value):
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonify(value.tolist())
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path.write_text(json.dumps(jsonify(payload), indent=2), encoding='utf-8')


def _macro_summary(summaries, panels):
    combined = {}
    for mechanism in REQUIRED_MECHANISMS:
        combined[mechanism] = {}
        for key in (
            'baseline_path_l1', 'candidate_path_l1', 'baseline_attraction_l1',
            'candidate_attraction_l1', 'baseline_count_normalized_l1', 'candidate_count_normalized_l1',
        ):
            values = [item[mechanism][key] for item in summaries]
            if any(value is None for value in values):
                combined[mechanism][key] = None
            else:
                frame = pd.DataFrame({'panel': panels, 'value': values})
                frame['family'] = frame.panel.str.rsplit('_', n=1).str[0]
                combined[mechanism][key] = float(frame.groupby('family').value.mean().mean())
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', action='append', required=True)
    parser.add_argument('--periods', type=int, default=4)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--synthetic-trials', type=int, default=4)
    parser.add_argument('--seed', type=int, default=606)
    args = parser.parse_args()
    if args.folds < 1 or args.synthetic_trials < 1 or args.periods < 1:
        raise ValueError('--folds, --synthetic-trials and --periods must be positive')
    output = args.output.resolve()
    if 'research_v6' not in output.parts:
        raise ValueError('Write OD v6 outputs under artifacts/research_v6')
    if (output / 'protocol.json').exists():
        raise ValueError('Use a fresh output directory to preserve the registered experiment')
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'split': TRAIN_SPLIT,
        'validation_and_private_read': False,
        'december_used_as_validation': False,
        'lambda_grid': list(LAMBDA_GRID),
        'weight_modes': ['absolute', 'relative'],
        'relative_weight_scale': 'reference_prior^2/max(prior, .01*reference_prior)^2; invariant to flow units',
        'baseline': {'regularization': BASELINE_REGULARIZATION, 'weight_mode': BASELINE_WEIGHT},
        'holdout_fraction_band': [0.10, 0.20],
        'holdout_fraction_target': 0.15,
        'outer_folds': args.folds,
        'selection': 'Inner holdouts on the complement of each outer holdout. Outer links are NaN during selection and refit.',
        'held_link_gain_required': HELD_LINK_GAIN_REQUIRED,
        'synthetic_regression_tolerance': SYNTHETIC_REGRESSION_TOLERANCE,
        'required_mechanisms': list(REQUIRED_MECHANISMS),
        'synthetic_periods': args.periods,
        'synthetic_trials': args.synthetic_trials,
        'seed': args.seed,
        'promotion_gate': (
            'Pass only with an outer held-link improvement and path-recovery retention '
            'on every synthetic mechanism. One winning mechanism, or a missing mechanism, keeps the baseline.'
        ),
        's_od_coefficient_in_total': S_OD_COEFFICIENT_IN_TOTAL,
        'delta_total_per_0_01_s_od': DELTA_TOTAL_PER_0_01_S_OD,
        'identification': IDENTIFICATION,
        'production_change': False,
        'panels': list(args.panel),
    }
    write_json(output / 'protocol.json', protocol)
    started = time.perf_counter()
    inventories = []
    outer_rows = []
    inner_rows = []
    synthetic_rows = []
    panel_gates = []
    summaries = []
    for panel in args.panel:
        panel_started = time.perf_counter()
        print(f'[od_v6] inventory {panel}', flush=True)
        loaded = load_train_panel(args.release_root, panel, TRAIN_SPLIT)
        inventories.append(loaded['inventory'])
        print(
            f"[od_v6] rank {loaded['inventory']['observed_incidence']['rank']} "
            f"nullity {loaded['inventory']['observed_incidence']['nullity']} "
            f"pooling {loaded['inventory']['temporal']['pooling']}",
            flush=True,
        )
        print(f'[od_v6] nested holdout {panel}', flush=True)
        assessment = nested_link_assessment(
            loaded['operator'], loaded['counts'], loaded['prior'],
            folds=args.folds, seed=args.seed, panel=panel,
        )
        outer_rows.extend(assessment['outer'])
        inner_rows.extend(assessment['inner'])
        observed = np.flatnonzero(np.isfinite(loaded['counts']))
        configs = [(BASELINE_REGULARIZATION, BASELINE_WEIGHT)]
        modal = (assessment['modal_regularization'], assessment['modal_weight_mode'])
        if modal != configs[0]:
            configs.append(modal)
        print(f'[od_v6] synthetic recovery {panel} configs {configs}', flush=True)
        panel_records = []
        for regularization, weight_mode in configs:
            try:
                recovered = synthetic_recovery(
                    loaded['operator'], observed, loaded['destinations'],
                    regularization, weight_mode, args.synthetic_trials, args.periods, args.seed,
                )
            except SolverError as exc:
                recovered = [{
                    'trial': None, 'period': None, 'mechanism': mechanism,
                    'regularization': float(regularization), 'weight_mode': weight_mode,
                    'path_normalized_l1': None, 'attraction_l1': None, 'count_normalized_l1': None,
                    'error': str(exc),
                } for mechanism in REQUIRED_MECHANISMS]
            for row in recovered:
                row['panel'] = panel
            panel_records.extend(recovered)
        synthetic_rows.extend(panel_records)
        summary = summarize_synthetic(panel_records, modal[1], modal[0])
        summaries.append(summary)
        outer = pd.DataFrame(assessment['outer'])
        baseline_error = float(outer.loc[outer.model.eq('baseline'), 'held_link_l1'].mean())
        candidate_error = float(outer.loc[outer.model.eq('selected'), 'held_link_l1'].mean())
        panel_gates.append({
            'panel': panel,
            'modal_regularization': assessment['modal_regularization'],
            'modal_weight_mode': assessment['modal_weight_mode'],
            'seconds': time.perf_counter() - panel_started,
            **promotion_gate(baseline_error, candidate_error, summary),
        })
        print(f"[od_v6] {panel} {panel_gates[-1]['decision']} in {panel_gates[-1]['seconds']:.1f}s", flush=True)
    macro = _macro_summary(summaries, args.panel)
    outer_frame = pd.DataFrame(outer_rows)
    outer_frame['family'] = outer_frame.panel.str.rsplit('_', n=1).str[0]
    panel_errors = outer_frame.groupby(['family', 'panel', 'model']).held_link_l1.mean()
    macro_errors = panel_errors.groupby(['family', 'model']).mean().groupby('model').mean()
    macro_baseline = float(macro_errors['baseline'])
    macro_candidate = float(macro_errors['selected'])
    overall = promotion_gate(macro_baseline, macro_candidate, macro)
    overall['panel_gates'] = panel_gates
    overall['all_panels_passed'] = all(item['gate_passed'] for item in panel_gates)
    if not overall['all_panels_passed']:
        overall['gate_passed'] = False
        overall['decision'] = 'keep_baseline'
        if overall['reason'].startswith('preregistered'):
            overall['reason'] = 'at least one panel missed the preregistered gate'
    overall['production_change'] = False
    overall['official_labels_used'] = False
    overall['limitations'] = [
        'Held-link error and synthetic path error are surrogate evidence. They are not S_od, S_dev, or S_attr.',
        'Train counts have already been used by earlier OD work. This is not an untouched confirmation set.',
        'A single static period token cannot identify a temporal model. Synthetic periods are generated, not read from the release.',
        'The nonnegative prior identifies count-null components. Counts alone do not.',
        'No production checkpoint, delivery archive, or Kaggle submission was changed.',
    ]
    write_json(output / 'inventory.json', inventories)
    write_json(output / 'decisions.json', overall)
    outer_frame.to_csv(output / 'outer_folds.csv', index=False)
    pd.DataFrame(inner_rows).to_csv(output / 'selection_folds.csv', index=False)
    synthetic_frame = pd.DataFrame(synthetic_rows)
    if 'assumptions' in synthetic_frame.columns:
        synthetic_frame['assumptions'] = synthetic_frame['assumptions'].map(
            lambda item: json.dumps(jsonify(item)) if isinstance(item, dict) else item
        )
    synthetic_frame.to_csv(output / 'synthetic_trials.csv', index=False)
    elapsed = time.perf_counter() - started
    write_json(output / 'runtime.json', {
        'seconds': elapsed,
        'panels': list(args.panel),
        'folds': args.folds,
        'periods': args.periods,
        'synthetic_trials': args.synthetic_trials,
        'seed': args.seed,
    })
    print(f'[od_v6] {overall["decision"]} in {elapsed:.1f}s -> {output}', flush=True)


if __name__ == '__main__':
    main()
