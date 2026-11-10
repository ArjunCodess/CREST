"""Apply the registered queue gates to completed development-window records."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.queue_scenarios_v6 import aggregate_macro, paired_day_block_bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment', type=Path)
    args = ap.parse_args()
    root = args.experiment
    protocol = json.loads((root/'protocol.json').read_text())
    if not (root/'runtime.json').exists():
        raise ValueError('Experiment has not completed')
    frame = pd.read_csv(root/'windows.csv')
    rows = frame[frame.block.eq('selection') & frame.target.eq('raw')].to_dict('records')
    arms = ['smoothed_hard', 'smoothed_soft']
    selected = max(arms, key=lambda arm: aggregate_macro(rows, arm)['overall'])
    checks = []
    for block in ('november', 'january'):
        result = {'block': block}
        for target in ('raw', 'smoothed'):
            subset = frame[frame.block.eq(block) & frame.target.eq(target)].to_dict('records')
            result[target] = paired_day_block_bootstrap(subset, selected, 'raw')
            new = aggregate_macro(subset, selected)['family']
            old = aggregate_macro(subset, 'raw')['family']
            result[target]['positive_families'] = sum(new[f] > old[f] for f in new)
            result[target]['family_count'] = len(new)
        result['metric_gate'] = (result['raw']['delta_point'] >= -.005
            and result['smoothed']['delta_point'] >= .01
            and result['smoothed']['positive_families'] >= 4
            and result['smoothed']['family_count'] >= 5)
        checks.append(result)
    diagnostics = json.loads((root/'diagnostics.json').read_text())
    flips = [r['flips'][0] for r in diagnostics if r['block'] == 'fit']
    share = sum(r['flips'] for r in flips)/sum(r['scored_cells'] for r in flips)
    passed = share >= .03 and all(r['metric_gate'] for r in checks)
    decision = {'selected_on_october_raw_proxy': selected, 'fit_flip_share': share,
        'flip_gate': share >= .03, 'development_checks': checks,
        'development_gate_passed': passed, 'production_promoted': False,
        'decision': 'keep_verified_v5',
        'limitations': ['Raw-selected matched windows assess sensitivity to labels, not event resampling.',
            'Smoothed labels are assumption-dependent proxies, not official latent truth.',
            'November and January were previously inspected; this is not a sealed test.',
            'One corridor per family is a pilot, not complete directional coverage.',
            f"Scenario decoding uses {protocol['scenarios']} Monte Carlo samples; a smaller effect could depend on simulation error."]}
    (root/'decisions.json').write_text(json.dumps(decision, indent=2))
    lines = ['# Queue v6 development assessment', '',
        f'October raw proxy selects `{selected}`. The development gate ' + ('passes.' if passed else 'fails.'),
        'The verified v5 submission remains selected. No official score is established by this experiment.', '',
        '| Block | Raw proxy delta | Smoothed proxy delta | Positive families |',
        '| --- | ---: | ---: | ---: |']
    for row in checks:
        lines.append(f"| {row['block']} | {row['raw']['delta_point']:+.5f} | {row['smoothed']['delta_point']:+.5f} | {row['smoothed']['positive_families']}/{row['smoothed']['family_count']} |")
    lines += ['', f'Training label flip share is {share:.2%}.', '',
        'All target arms share the raw-selected decoder, architecture, event sample, and restarted causal filter features.',
        'Decoder-only comparisons use identical raw probabilities. The seed-footprint comparison changes the joint distribution and reports its marginal mismatch.', '',
        'Bootstrap draws resample paired days within each panel and condition, keeping macro weights fixed. They do not retain cross-condition day dependence.', '',
        *decision['limitations']]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
