"""Regenerate all v4 tasks from packaged checkpoints and the licensed release."""
import argparse
import atexit
import gc
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import panels
from src.od import build_od
from src.submission import merge_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--package-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()
    root, package, output = args.release_root.resolve(), args.package_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.reproduction.lock'
    with lock.open('x') as owner:
        owner.write(str(os.getpid()))
    atexit.register(lock.unlink, missing_ok=True)
    if any((output / name).exists() for name in ('state.csv', 'baseline_queue.csv', 'odme.csv', 'submission.csv')):
        raise ValueError('Reproduction requires a fresh output directory; retain failed runs separately')
    from src.pipeline import predict_state, predict_queue
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=package, env=env, check=True)
    manifest = json.loads((package / 'models/enhanced/model_manifest.json').read_text())
    selected = {r['panel']: r for r in manifest['models']}
    if set(selected) != set(panels(root)):
        raise ValueError('This release requires a promoted enhanced state model on every panel')
    queue_selection = pd.read_csv(package / 'models/queue/queue_proxy_cv.csv')
    classifiers = set(queue_selection.loc[queue_selection.accepted, 'panel'])
    paths = {task: output / f'{task}.csv' for task in ('state', 'baseline_queue', 'odme')}
    written = {task: False for task in paths}
    for panel in panels(root):
        row = selected[panel]
        checkpoint = package / 'models/enhanced' / row['file']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Enhanced state checkpoint hash differs')
        with checkpoint.open('rb') as source:
            state_payload = pickle.load(source)
        with (package / 'models/state' / f'{panel}.pkl').open('rb') as source:
            queue_payload = pickle.load(source)
        if panel in classifiers:
            with (package / 'models/queue' / f'{panel}_queue.pkl').open('rb') as source:
                queue_payload['queue_classifier'] = pickle.load(source)
        for split in ('validation', 'private'):
            print('Regenerate v4', panel, split, flush=True)
            with threadpool_limits(limits=row['inference_blas_threads'], user_api='blas'):
                state, prediction, data = predict_state(root, panel, split, state_payload)
            queue = predict_queue(root, panel, split, queue_payload)
            od, _ = build_od(root, panel, split)
            for task, frame in (('state', state), ('baseline_queue', queue), ('odme', od)):
                if len(frame):
                    frame.to_csv(paths[task], index=False, mode='a' if written[task] else 'w', header=not written[task])
                    written[task] = True
            del state, prediction, data, queue, od
            gc.collect()
        del state_payload, queue_payload
        gc.collect()
    for script, model_dir, baseline, dest in (
        ('export_event_queue.py', 'events', paths['baseline_queue'], output / 'events'),
        ('export_onset_v4.py', 'onset', output / 'events/queue.csv', output / 'onset')):
        subprocess.run([sys.executable, str(package / 'scripts' / script), '--release-root', str(root),
            '--model-root', str(package / 'models' / model_dir), '--baseline-queue', str(baseline),
            '--output', str(dest)], cwd=package, env=env, check=True)
    actual = output / 'submission.csv'
    rows = merge_files(root, {'state': paths['state'], 'queue': output / 'onset/queue.csv', 'odme': paths['odme']}, actual)
    if rows != 6980503:
        raise ValueError('Unexpected reproduction coverage')
    maximum = np.zeros(4)
    if args.reference:
        for x, y in zip_longest(pd.read_csv(actual, chunksize=100000), pd.read_csv(args.reference, chunksize=100000)):
            if x is None or y is None or not x.iloc[:, :2].equals(y.iloc[:, :2]):
                raise ValueError('Reproduction identifier or task order differs')
            xx, yy = x.iloc[:, 2:].to_numpy(), y.iloc[:, 2:].to_numpy()
            maximum = np.maximum(maximum, np.abs(xx - yy).max(axis=0))
            np.testing.assert_allclose(xx, yy, rtol=1e-6, atol=1e-6)
    with actual.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    audit = {'rows': rows, 'sha256': digest, 'all_tasks_regenerated_from_checkpoints_and_release': True,
        'reference_compared': bool(args.reference), 'matches_selected_submission_within_1e_6': True if args.reference else None,
        'max_absolute_difference': maximum.tolist() if args.reference else None, 'reference_used_only_after_prediction': True}
    (output / 'reproduction_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
