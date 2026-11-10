"""Portable replay of the attributed CC0 CatBoost state ensemble."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['CPU', 'GPU'], default='CPU')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    assets = args.assets.resolve()
    os.environ.update(TFB_DATA_ROOT=str(args.release_root.resolve()), TFB_OUTPUT_ROOT=str(args.output.resolve()),
                      TFB_THREADS='4', TFB_PREDICT_DEVICE=args.device,
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, str(assets))
    if (assets/'_pydeps').exists():
        sys.path.insert(0, str(assets/'_pydeps'))
    from state_inference import StatePredictor, predict_states
    predict_states(StatePredictor(assets))
    dest = args.output/'state.csv'
    with dest.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    with dest.open('rb') as f:
        rows = sum(1 for _ in f)-1
    if rows != 6735795:
        raise ValueError(f'Incomplete state output: {rows}')
    (args.output/'replay_audit.json').write_text(json.dumps({'rows': rows, 'sha256': digest,
        'device': args.device, 'license': 'CC0-1.0', 'shipped_predictions_used': False,
        'source': 'https://www.kaggle.com/datasets/anubisyy/trafficflowbench-086517-public-assets'}, indent=2))
    print(f'Replayed {rows} state cells; sha256 {digest}', flush=True)


if __name__ == '__main__':
    main()
