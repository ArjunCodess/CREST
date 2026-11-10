"""Check the completed upload independently and write current local run notes."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import pickle
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    output = args.output.resolve()
    counts = {'state': 0, 'queue': 0, 'odme': 0}
    required = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}
    keys = pd.read_csv(args.release_root / 'submission_key.csv', chunksize=50000, usecols=['submission_id', 'task'], dtype=str)
    for frame in pd.read_csv(output / 'submission.csv', chunksize=50000, dtype={'submission_id': str, 'task': str}):
        expected = next(keys)
        if not frame[['submission_id', 'task']].reset_index(drop=True).equals(expected[['submission_id', 'task']].reset_index(drop=True)):
            raise ValueError('Final upload IDs or task order differ from the official key')
        if not np.isfinite(frame.iloc[:, 2:].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite final upload value')
        for task, fields in required.items():
            values = frame.loc[frame.task.eq(task), fields].to_numpy(dtype=float)
            if (values < 0).any() or (task == 'queue' and not np.isin(values, [0, 1]).all()):
                raise ValueError(f'Invalid final {task} value')
            counts[task] += len(values)
    if next(keys, None) is not None or not all(counts.values()):
        raise ValueError('Final upload has incomplete task coverage')
    choices = json.loads((output / 'prediction_models.json').read_text())
    for record in choices:
        with (output / f'{record["panel"]}.pkl').open('rb') as checkpoint:
            payload = pickle.load(checkpoint)
        if pd.Timestamp(payload['fit_end']) != pd.Timestamp('2031-02-28T23:55:00Z'):
            raise ValueError(f'Incomplete final training coverage: {record["panel"]}')
    link_scores = {}
    for split in ['validation', 'private']:
        score = pd.read_csv(output / f'official_link_score_{split}.csv')
        if score[['missing_paths', 'duplicate_rows', 'invalid_rows']].to_numpy().any():
            raise ValueError('Official OD scorer found invalid path coverage')
        link_scores[split] = float(score.loc[score.level.eq('overall'), 'S_link'].iloc[0])
    with (output / 'submission.csv').open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    with (root / 'paper/main.pdf').open('rb') as source:
        paper_digest = hashlib.file_digest(source, 'sha256').hexdigest()
    summary = json.loads((output / 'report/evidence.json').read_text())
    audit = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'rows': counts, 'submission_sha256': digest,
             'paper_sha256': paper_digest, 'author': 'Arjun Vijay Prakash', 'affiliation': 'Independent Researcher',
             'official_S_link': link_scores, 'model_choices': choices, 'holdout': summary,
             'python': platform.python_version(), 'dependencies': {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'lightgbm', 'torch', 'pyarrow']},
             'official_overall_score': None, 'uploaded_to_kaggle': False}
    (output / 'final_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
