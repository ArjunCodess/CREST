"""Reassess saved v6 rows on official queue families without rewriting evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.queue_scenarios_v6 import aggregate_macro, paired_day_block_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.experiment / 'windows.csv')
    excluded = frame.panel.str.startswith('D12_I405_')
    frame = frame[~excluded]
    rows = frame[frame.block.eq('selection') & frame.target.eq('raw')].to_dict('records')
    selected = max(('smoothed_hard', 'smoothed_soft'), key=lambda arm: aggregate_macro(rows, arm)['overall'])
    checks = []
    for block in ('november', 'january'):
        for target in ('raw', 'smoothed'):
            rows = frame[frame.block.eq(block) & frame.target.eq(target)].to_dict('records')
            checks.append({'block': block, 'target': target, **paired_day_block_bootstrap(rows, selected, 'raw')})
    result = {'selected': selected, 'excluded_rows': int(excluded.sum()), 'checks': checks,
              'production_promoted': False, 'interpretation': 'Post-hoc corrected development comparison; full directional coverage and original label-flip gate still required.'}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
