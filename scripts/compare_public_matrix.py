"""Create controlled hybrids from independently reproduced full submissions.

The external-only file remains a research reference. Every hybrid retains our
OD solution, and changes exactly the declared state and/or queue components.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ours', type=Path, required=True)
    ap.add_argument('--external', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source-url', required=True)
    ap.add_argument('--license', required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    policies = [('state50_own_queue', 0.5, False),
                ('own_state_external_queue', 0.0, True),
                ('state50_external_queue', 0.5, True),
                ('state75_external_queue', 0.75, True)]
    columns = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']
    dtype = {'submission_id': np.int64, 'task': str, 'speed_kmh': float,
             'flow_vph': float, 'queue_pred': float, 'path_flow': float}
    readers = [pd.read_csv(p, chunksize=200000, dtype=dtype) for p in (args.ours, args.external)]
    counts = {t: 0 for t in ('state', 'queue', 'odme')}
    changed = {c: 0 for c in columns[2:]}
    squared = np.zeros(2)
    queue_intersection = queue_union = 0
    offset = 0
    while True:
        left = next(readers[0], None)
        right = next(readers[1], None)
        if left is None and right is None:
            break
        if left is None or right is None or len(left) != len(right):
            raise ValueError('Submission lengths differ')
        expected = np.arange(offset+1, offset+len(left)+1)
        if not np.array_equal(left.submission_id.to_numpy(), expected) or not np.array_equal(right.submission_id.to_numpy(), expected):
            raise ValueError('Identifiers must match official sample order')
        if not left.task.equals(right.task) or not left.task.isin(counts).all():
            raise ValueError('Task mapping differs')
        for frame in (left, right):
            if not np.isfinite(frame[columns[2:]].to_numpy()).all():
                raise ValueError('Nonfinite output')
            if not frame.queue_pred.isin([0, 1]).all() or (frame[columns[2:]] < 0).any().any():
                raise ValueError('Invalid binary/nonnegative outputs')
        state = left.task.eq('state').to_numpy()
        queue = left.task.eq('queue').to_numpy()
        if (left.loc[state, 'speed_kmh'] <= 0).any() or (right.loc[state, 'speed_kmh'] <= 0).any():
            raise ValueError('State speeds must be positive')
        for task in counts:
            counts[task] += int(left.task.eq(task).sum())
        for col in changed:
            changed[col] += int((left[col].to_numpy() != right[col].to_numpy()).sum())
        squared += np.square(left.loc[state, ['speed_kmh', 'flow_vph']].to_numpy()-right.loc[state, ['speed_kmh', 'flow_vph']].to_numpy()).sum(0)
        qa = left.loc[queue, 'queue_pred'].to_numpy(bool)
        qb = right.loc[queue, 'queue_pred'].to_numpy(bool)
        queue_intersection += int((qa & qb).sum())
        queue_union += int((qa | qb).sum())
        for name, share, use_queue in policies:
            out = left.copy()
            out.loc[state, ['speed_kmh', 'flow_vph']] = ((1-share)*left.loc[state, ['speed_kmh', 'flow_vph']].to_numpy()
                + share*right.loc[state, ['speed_kmh', 'flow_vph']].to_numpy())
            if use_queue:
                out.loc[queue, 'queue_pred'] = right.loc[queue, 'queue_pred'].to_numpy()
            out['queue_pred'] = out.queue_pred.astype(np.int8)
            out.to_csv(args.output/f'{name}.csv', index=False, mode='w' if offset == 0 else 'a', header=offset == 0)
        offset += len(left)
    if counts != {'state': 6735795, 'queue': 174000, 'odme': 70708}:
        raise ValueError(f'Official row coverage differs: {counts}')
    report = {'ours': str(args.ours), 'external': str(args.external), 'source': args.source_url,
              'license': args.license, 'rows': offset, 'counts': counts,
              'external_only_submitted': False, 'changed_between_references': changed,
              'state_disagreement_rmse': np.sqrt(squared/counts['state']).tolist(),
              'queue_prediction_agreement_iou': queue_intersection/max(queue_union, 1),
              'disagreement_is_not_accuracy': True, 'candidates': []}
    for name, share, use_queue in policies:
        path = args.output/f'{name}.csv'
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        report['candidates'].append({'name': name, 'public_state_share': share,
            'queue': 'external' if use_queue else 'ours', 'od': 'ours', 'path': str(path),
            'sha256': digest, 'public_score': None, 'submitted': False})
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
