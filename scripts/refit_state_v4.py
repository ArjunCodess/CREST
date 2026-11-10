"""Refit accepted state models on training observations and export both held-out months."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
from threadpoolctl import threadpool_info, threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_state, graph
from src.pipeline import predict_state
from src.state_v4 import EnhancedReconstructor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--pilot', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--export-only', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'model_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'models': []}
    for decision_path in sorted(args.pilot.glob('*_decision.json')):
        decision = json.loads(decision_path.read_text())
        if not decision['accepted']:
            continue
        panel = decision['panel']
        # Every promoted trial selected the enhanced residual alone. Reject a
        # different blend instead of silently changing its validated policy.
        if decision['new_weight'] != 1 or decision['factor_weight'] != 0:
            raise ValueError(f'{panel}: this exporter requires the selected pure residual model')
        checkpoint = args.output / f'{panel}.pkl'
        entry = next((r for r in manifest['models'] if r['panel'] == panel), None)
        if entry is None:
            if args.export_only:
                raise ValueError(f'Missing production checkpoint: {panel}')
            print('Full state refit', panel, flush=True)
            data = load_state(args.release_root, panel, 'train', True)
            up, down, _, params = graph(args.release_root, panel, data.links)
            model = EnhancedReconstructor().fit(data.values, data.times, up, down, params)
            payload = {'model': model, 'links': data.links, 'up': up, 'down': down, 'params': params,
                       'fit_start': str(data.times[0]), 'fit_end': str(data.times[-1])}
            temporary = checkpoint.with_suffix('.partial.pkl')
            with temporary.open('wb') as target:
                pickle.dump(payload, target)
            temporary.replace(checkpoint)
            del data, model
            gc.collect()
            entry = {'panel': panel, 'file': checkpoint.name,
                     'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'fit_start': payload['fit_start'], 'fit_end': payload['fit_end'],
                     'inference_blas_threads': max(r['num_threads'] for r in threadpool_info() if r['user_api'] == 'blas'),
                     'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()}
            manifest['models'].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        else:
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Checkpoint hash differs: {panel}')
            if hashlib.sha256(decision_path.read_bytes()).hexdigest() != entry['decision_sha256']:
                raise ValueError(f'Selection decision changed: {panel}')
            with checkpoint.open('rb') as source:
                payload = pickle.load(source)
        for split in ('validation', 'private'):
            dest = args.output / f'{panel}_{split}_state.csv'
            audit_path = dest.with_suffix('.json')
            if dest.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit['model_sha256'] == entry['sha256'] and hashlib.sha256(dest.read_bytes()).hexdigest() == audit['sha256']:
                    print('Keep state export', panel, split, flush=True)
                    continue
            print('Export state', panel, split, flush=True)
            with threadpool_limits(limits=entry['inference_blas_threads'], user_api='blas'):
                frame, prediction, data = predict_state(args.release_root, panel, split, payload)
            partial = dest.with_suffix('.partial.csv')
            frame.to_csv(partial, index=False)
            partial.replace(dest)
            audit_path.write_text(json.dumps({'panel': panel, 'split': split, 'rows': len(frame),
                'model_sha256': entry['sha256'], 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2))
            del frame, prediction, data
            gc.collect()
        del payload
        gc.collect()


if __name__ == '__main__':
    main()
