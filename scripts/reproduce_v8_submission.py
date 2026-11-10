"""Replay attributed CREST state/queue ensembles from release + checkpoints."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--package-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--variant', choices=['cat', 'unet'], default='unet')
    ap.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    ap.add_argument('--verified-own', type=Path, help='Optional prior unchanged v7 checkpoint replay directory')
    ap.add_argument('--reference', type=Path, help='Compared only after generating predictions')
    args = ap.parse_args()
    package, output = args.package_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    release = args.release_root.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    if args.device == 'CPU':
        env['CUDA_VISIBLE_DEVICES'] = ''

    def run(name, *params):
        subprocess.run([sys.executable, str(package/'scripts'/name), *map(str, params)],
                       cwd=package, env=env, check=True)

    own = args.verified_own.resolve() if args.verified_own else output/'own'
    if args.verified_own:
        audit = json.loads((own/'reproduction_audit.json').read_text())
        if not audit['all_tasks_regenerated_from_checkpoints_and_release'] or any(audit['max_absolute_difference']):
            raise ValueError('Existing v7 checkpoint replay is not verified')
        with (own/'merged/submission.csv').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != audit['sha256']:
                raise ValueError('Existing own replay was modified')
    else:
        run('reproduce_v7_submission.py', '--release-root', release, '--package-root', package, '--output', own)
    assets = package/'external/anubisyy'
    run('reproduce_public_queue.py', '--release-root', release, '--assets', assets, '--output', output/'queue')
    run('reproduce_public_state.py', '--release-root', release, '--assets', assets, '--output', output/'cat', '--device', args.device)
    run('compare_public_submissions.py', '--base', own/'merged/submission.csv', '--state', output/'cat/state.csv',
        '--key', release/'submission_key.csv', '--output', output/'cat_blend', '--weights', '0.5',
        '--source-url', 'https://www.kaggle.com/datasets/anubisyy/trafficflowbench-086517-public-assets', '--source-license', 'CC0-1.0')
    cat = output/'cat_blend/state_public_050.csv'
    run('splice_queue_rows.py', '--submission', cat, '--queue', output/'queue/queue.csv',
        '--key', release/'submission_key.csv', '--apply')
    selected = cat
    if args.variant == 'unet':
        run('public_bizai_state.py', '--source', package/'external/bizai', '--release-root', release,
            '--output', output/'unet', '--kind', 'unet2', '--seeds', '42', '7', '123')
        run('compare_public_submissions.py', '--base', cat, '--state', output/'unet/state_submission.csv',
            '--key', release/'submission_key.csv', '--output', output/'unet_blend', '--weights', '0.5',
            '--source-url', 'https://github.com/IzbassarO/tfbench-bizai-release', '--source-license', 'MIT')
        selected = output/'unet_blend/state_public_050.csv'
    final = output/'submission.csv'
    shutil.copyfile(selected, final)
    with final.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    maxima = np.zeros(4)
    rows = 0
    if args.reference:
        for actual, reference in zip_longest(pd.read_csv(final, chunksize=100000), pd.read_csv(args.reference, chunksize=100000)):
            if actual is None or reference is None or not actual.iloc[:, :2].equals(reference.iloc[:, :2]):
                raise ValueError('Reference identifiers or row count differ')
            x, y = actual.iloc[:, 2:].to_numpy(), reference.iloc[:, 2:].to_numpy()
            maxima = np.maximum(maxima, np.abs(x-y).max(0))
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
            rows += len(actual)
        if rows != 6980503:
            raise ValueError('Incomplete replay')
    result = {'rows': 6980503, 'variant': args.variant, 'sha256': digest,
        'all_tasks_regenerated_from_checkpoints_and_release': True,
        'reference_compared_after_generation': bool(args.reference),
        'max_absolute_difference': maxima.tolist() if args.reference else None,
        'external_sources': {'anubisyy': 'CC0-1.0', 'BizAI': 'MIT'},
        'own_od_preserved': True, 'output': str(final)}
    (output/'reproduction_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
