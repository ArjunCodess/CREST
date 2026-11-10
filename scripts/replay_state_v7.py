"""Regenerate state exports from checkpoints before comparing their file hashes."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from threadpoolctl import threadpool_limits
from src.pipeline import predict_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.models/'model_manifest.json').read_text())
    records = []
    for entry in manifest['models']:
        checkpoint = args.models/entry['file']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Checkpoint hash differs')
        with checkpoint.open('rb') as source:
            payload = pickle.load(source)
        for split in ('validation', 'private'):
            print(entry['panel'], split, flush=True)
            with threadpool_limits(limits=4):
                frame, prediction, data = predict_state(args.release_root, entry['panel'], split, payload)
            destination = args.output/f"{entry['panel']}_{split}_state.csv"
            frame.to_csv(destination, index=False)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            # The expected prediction export is consulted only after regeneration.
            expected = json.loads((args.models/destination.name).with_suffix('.json').read_text())
            if digest != expected['sha256'] or len(frame) != expected['rows']:
                raise ValueError('Regenerated prediction export differs')
            records.append({'panel': entry['panel'], 'split': split, 'rows': len(frame), 'sha256': digest, 'exact_match': True})
            del frame, prediction, data
    (args.output/'replay_audit.json').write_text(json.dumps({'exports': records, 'reference_used_only_after_prediction': True}, indent=2))


if __name__ == '__main__':
    main()
