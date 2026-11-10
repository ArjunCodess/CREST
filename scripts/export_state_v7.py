"""Refit an audited pure-wave policy and export an isolated official trial."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from threadpoolctl import threadpool_limits
from src.data import load_state, graph
from src.wave_v7 import WaveReconstructor
from src.pipeline import predict_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if not audit['state_audit_passed'] or audit['selected_weight'] != 1.:
        parser.error('A passing audited pure-wave policy is required; no unvalidated blend is substituted.')
    prefix = args.pilot/f"{audit['panel']}_models.pkl"
    if hashlib.sha256(prefix.read_bytes()).hexdigest() != audit['checkpoint_sha256']:
        parser.error('The audited pilot checkpoint differs.')
    protocol = json.loads((args.pilot/'protocol.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    panel = audit['panel']
    print('Refit all released training months:', panel, flush=True)
    data = load_state(args.release_root, panel, 'train', True)
    up, down, _, params = graph(args.release_root, panel, data.links)
    with threadpool_limits(limits=4):
        model = WaveReconstructor(protocol['rounds'], protocol['max_rows']).fit(data.values, data.times, up, down, params)
    payload = {'model': model, 'links': data.links, 'up': up, 'down': down, 'params': params,
               'fit_start': str(data.times[0]), 'fit_end': str(data.times[-1])}
    checkpoint = args.output/f'{panel}.pkl'
    with checkpoint.open('wb') as target:
        pickle.dump(payload, target)
    entry = {'panel': panel, 'file': checkpoint.name, 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'fit_start': payload['fit_start'], 'fit_end': payload['fit_end'], 'inference_blas_threads': 4,
             'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(), 'production_promoted': False}
    (args.output/'model_manifest.json').write_text(json.dumps({'models': [entry]}, indent=2))
    del data
    for split in ('validation', 'private'):
        print('Export:', panel, split, flush=True)
        with threadpool_limits(limits=4):
            frame, prediction, data = predict_state(args.release_root, panel, split, payload)
        destination = args.output/f'{panel}_{split}_state.csv'
        frame.to_csv(destination, index=False)
        destination.with_suffix('.json').write_text(json.dumps({'panel': panel, 'split': split,
            'rows': len(frame), 'model_sha256': entry['sha256'],
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))
        del frame, prediction, data


if __name__ == '__main__':
    main()
