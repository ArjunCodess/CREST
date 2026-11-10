"""Align independent public inference, audit differences, and build state blends.

Difference statistics measure disagreement, not accuracy. Queue and OD remain
unchanged unless an external full submission is explicitly selected as the base.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--key', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--weights', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--source-url', required=True)
    ap.add_argument('--source-license', required=True)
    args = ap.parse_args()
    if any(w <= 0 or w > 1 for w in args.weights):
        ap.error('Public model shares must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=True)
    numeric = ['speed_kmh', 'flow_vph']
    keys = ['panel', 'timestamp', 'link_id']
    external = pl.read_csv(args.state, columns=keys+numeric,
                           schema_overrides={k: pl.String for k in keys})
    if external.select(keys).is_duplicated().any():
        raise ValueError('External state has duplicate keys')
    mapping = pl.read_csv(args.key, columns=['submission_id', 'task']+keys,
                          schema_overrides={k: pl.String for k in keys})
    state_keys = mapping.filter(pl.col('task') == 'state').drop('task')
    aligned = state_keys.join(external, on=keys, how='left', validate='1:1')
    if len(aligned) != len(external) or any(aligned[c].null_count() for c in numeric):
        raise ValueError('External states do not exactly cover the required state rows')
    ids = aligned['submission_id'].to_numpy().astype(np.int64)
    if (ids < 1).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid state identifiers')
    values = aligned.select(numeric).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 0] <= 0).any():
        raise ValueError('Invalid external state values')
    total = len(mapping)
    replacements = np.full((total, 2), np.nan, dtype=np.float64)
    replacements[ids-1] = values
    del external, aligned, mapping, state_keys
    report = {'source_url': args.source_url, 'license': args.source_license,
              'base': str(args.base), 'state': str(args.state),
              'comparison_is_disagreement_not_accuracy': True,
              'queue_and_od_preserved': True, 'rows': total, 'state_rows': len(ids),
              'candidates': []}
    candidates = [(w, args.output/f'state_public_{int(round(w*100)):03d}.csv') for w in args.weights]
    if any(p.exists() for _, p in candidates):
        raise FileExistsError('Choose a fresh output folder to preserve prior candidates')
    sse = np.zeros(2); differences = np.zeros(2, dtype=np.int64); maxima = np.zeros(2)
    offset = 0
    for chunk in pd.read_csv(args.base, chunksize=200000):
        expected = np.arange(offset+1, offset+len(chunk)+1)
        if not np.array_equal(chunk.submission_id.to_numpy(), expected):
            raise ValueError('Base identifiers must be complete, unique, and in sample order')
        state = chunk.task.eq('state').to_numpy()
        new = replacements[offset:offset+len(chunk)]
        if not np.array_equal(state, np.isfinite(new).all(1)):
            raise ValueError('Task identifiers disagree with the official key')
        old = chunk[numeric].to_numpy()
        delta = new[state]-old[state]
        sse += np.square(delta).sum(0)
        differences += (delta != 0).sum(0)
        if len(delta):
            maxima = np.maximum(maxima, np.abs(delta).max(0))
        for weight, dest in candidates:
            mixed = old.copy()
            mixed[state] = (1-weight)*old[state]+weight*new[state]
            chunk[numeric] = mixed
            chunk.to_csv(dest, index=False, mode='w' if offset == 0 else 'a', header=offset == 0)
        offset += len(chunk)
    if offset != total:
        raise ValueError('Base row count does not match sample')
    report['disagreement'] = dict(rmse=np.sqrt(sse/len(ids)).tolist(),
                                  changed=differences.tolist(), maximum=maxima.tolist())
    for weight, dest in candidates:
        with dest.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        report['candidates'].append({'public_state_share': weight, 'path': str(dest),
                                     'sha256': digest, 'public_score': None})
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
