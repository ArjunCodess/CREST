"""Write an evidence-only pilot report without revising the selected IEEE paper."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    lines = ['# V7 pilot evidence', '',
        'The verified v5 entry remains selected. These are training-only development experiments, not official task scores or a private-score estimate.', '']
    for task in ('state', 'queue'):
        folder = args.campaign/task
        lines.extend([f'## {task.capitalize()}', ''])
        if not (folder/'runtime.json').exists():
            lines.extend(['No complete real-data run is recorded.', ''])
            continue
        decisions = json.loads((folder/'decisions.json').read_text())
        runtime = json.loads((folder/'runtime.json').read_text())
        lines.append(f"Completed in {runtime['seconds']:.1f} seconds. The control uses the v5 architecture refitted on the same prefix as the challenger.")
        lines.extend(['', '| Panel | Pilot gate | Remaining evaluation |', '| --- | --- | --- |'])
        for row in decisions:
            passed = 'pass' if row['pilot_gate_passed'] else 'fail'
            lines.append(f"| {row['panel']} | {passed} | {row['remaining_gates']} |")
        lines.extend(['', 'A pilot pass does not authorize production promotion. Previous development exposure and proxy-label uncertainty remain.', ''])
    lines.extend(['## Reproduction and selection', '',
        'No new Kaggle result is claimed. A candidate needs complete affected-direction coverage, the retention and uncertainty checks, a comparison with deployed policies, and exact submission reproduction before a final-selection decision.', ''])
    (args.campaign/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
