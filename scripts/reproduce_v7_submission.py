"""Replay v5 components plus audited wave states from released inputs and checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from itertools import zip_longest
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument("--verify-existing", action="store_true", help="Finish auditing an already regenerated output")
    args = parser.parse_args()
    package, output = args.package_root.resolve(), args.output.resolve()
    if not args.verify_existing:
        output.mkdir(parents=True, exist_ok=False)
        environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        commands = [
            ['reproduce_v5_submission.py', '--release-root', str(args.release_root), '--package-root', str(package), '--output', str(output/'v5')],
            ['replay_state_v7.py', '--release-root', str(args.release_root), '--models', str(package/'models/wave_v7'), '--output', str(output/'wave')],
        ]
        for command in commands:
            subprocess.run([sys.executable, str(package/'scripts'/command[0]), *command[1:]], cwd=package, env=environment, check=True)
        wave = package/'models/wave_v7'
        manifest = json.loads((wave/'model_manifest.json').read_text())
        shutil.copyfile(wave/'model_manifest.json', output/'wave/model_manifest.json')
        for entry in manifest['models']:
            shutil.copyfile(wave/entry['file'], output/'wave'/entry['file'])
            for split in ('validation', 'private'):
                name = f"{entry['panel']}_{split}_state.json"
                shutil.copyfile(wave/name, output/'wave'/name)
        subprocess.run([sys.executable, str(package/'scripts/build_state_v4_candidate.py'),
            '--release-root', str(args.release_root), '--model-root', str(output/'wave'),
            '--reference', str(output/'v5/submission.csv'), '--output', str(output/'merged')], cwd=package, env=environment, check=True)
    actual = output/'merged/submission.csv'
    with actual.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    baseline_audit = json.loads((output/'v5/reproduction_audit.json').read_text())
    wave_audit = json.loads((output/'wave/replay_audit.json').read_text())
    merged_audit = json.loads((output/'merged/candidate_audit.json').read_text())
    if not baseline_audit['all_tasks_regenerated_from_checkpoints_and_release'] or not all(row['exact_match'] for row in wave_audit['exports']) or merged_audit['sha256'] != digest:
        raise ValueError('Component replay evidence is incomplete or differs')
    maximum = np.zeros(4)
    expected = None
    if args.reference:
        with args.reference.open('rb') as source:
            expected = hashlib.file_digest(source, 'sha256').hexdigest()
        rows = 0
        for actual_frame, reference_frame in zip_longest(pd.read_csv(actual, chunksize=100000), pd.read_csv(args.reference, chunksize=100000)):
            if actual_frame is None or reference_frame is None or not actual_frame.iloc[:, :2].equals(reference_frame.iloc[:, :2]):
                raise ValueError('Complete replay identifier order differs')
            x, y = actual_frame.iloc[:, 2:].to_numpy(), reference_frame.iloc[:, 2:].to_numpy()
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)
            maximum = np.maximum(maximum, np.abs(x-y).max(0))
            rows += len(actual_frame)
        if rows != 6980503:
            raise ValueError('Incomplete replay comparison')
    audit = {'rows': 6980503, 'sha256': digest, 'all_tasks_regenerated_from_checkpoints_and_release': True,
             'reference_compared_after_prediction': bool(args.reference), 'exact_file_match': digest == expected if expected else None,
             'max_absolute_difference': maximum.tolist() if args.reference else None,
             'matches_selected_within_1e_6': True if args.reference else None,
             'wave_exports_exact': True, 'production_component_reference': 'v5 checkpoint replay, not copied scored CSV'}
    (output/'reproduction_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
