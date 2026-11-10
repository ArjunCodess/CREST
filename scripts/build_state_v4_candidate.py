"""Stream accepted v4 state exports into an audited queue/OD reference submission."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.submission import KEYS, keyed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.model_root / 'model_manifest.json').read_text())
    accepted = {r['panel'] for r in manifest['models']}
    if not accepted:
        raise ValueError('No promoted state models')
    for row in manifest['models']:
        if hashlib.sha256((args.model_root / row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('State checkpoint hash differs')
        for split in ('validation', 'private'):
            dest = args.model_root / f"{row['panel']}_{split}_state.csv"
            audit = json.loads(dest.with_suffix('.json').read_text())
            if audit['model_sha256'] != row['sha256'] or hashlib.sha256(dest.read_bytes()).hexdigest() != audit['sha256']:
                raise ValueError('State export is stale or modified')
    final = args.output / 'submission.csv'
    partial = final.with_suffix('.partial.csv')
    counts, changed, updated, first = {}, 0, 0, True
    cache = {}
    streams = [pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100000)
               for path in (args.reference, args.release_root / 'submission_key.csv', args.release_root / 'sample_submission.csv')]
    for base, key, expected in zip_longest(*streams):
        if base is None or key is None or expected is None:
            raise ValueError('Reference, key and sample row counts differ')
        if not base.iloc[:, :2].equals(key[['submission_id', 'task']]) or not base.iloc[:, :2].equals(expected.iloc[:, :2]):
            raise ValueError('Official identifier or task order differs')
        out = base.copy()
        state = key.task.eq('state') & key.panel.isin(accepted)
        needed = set(key.loc[state, 'panel'])
        cache = {p: t for p, t in cache.items() if p in needed}
        for panel in needed:
            if panel not in cache:
                frames = [pd.read_csv(args.model_root / f'{panel}_{split}_state.csv',
                    dtype={c: str for c in KEYS['state'] if c != 'timestamp'}) for split in ('validation', 'private')]
                cache[panel] = keyed(pd.concat(frames, ignore_index=True), 'state')
            mask = state & key.panel.eq(panel)
            wanted = key.loc[mask, KEYS['state']].copy()
            wanted['timestamp'] = pd.to_datetime(wanted.timestamp, utc=True)
            values = cache[panel].reindex(pd.MultiIndex.from_frame(wanted)).to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Missing or invalid state target')
            changed += int(np.any(base.loc[mask, ['speed_kmh', 'flow_vph']].to_numpy(dtype=float) != values, axis=1).sum())
            updated += len(values)
            out.loc[mask, ['speed_kmh', 'flow_vph']] = values.astype(str)
        if not out.loc[~state].equals(base.loc[~state]) or not out.drop(columns=['speed_kmh', 'flow_vph']).equals(base.drop(columns=['speed_kmh', 'flow_vph'])):
            raise ValueError('Unselected fields changed')
        numeric = out.iloc[:, 2:].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.isin(numeric[:, 2], [0, 1]).all():
            raise ValueError('Invalid submission value')
        for task, n in out.task.value_counts().items():
            counts[task] = counts.get(task, 0) + int(n)
        out.to_csv(partial, mode='w' if first else 'a', header=first, index=False)
        first = False
    if counts != {'state': 6735795, 'queue': 174000, 'odme': 70708}:
        raise ValueError(f'Unexpected task coverage: {counts}')
    if accepted == set(pd.read_csv(args.release_root / 'submission_key.csv', usecols=['panel']).panel.dropna().unique()) and updated != counts['state']:
        raise ValueError('State coverage differs')
    partial.replace(final)
    with final.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    audit = {'rows': sum(counts.values()), 'tasks': counts, 'updated_state_rows': updated,
             'changed_state_rows': changed, 'panels': sorted(accepted), 'queue_and_od_exactly_preserved': True,
             'official_key_and_sample_order': True, 'finite_nonnegative': True, 'queue_binary': True,
             'reference': str(args.reference), 'sha256': digest, 'bytes': final.stat().st_size}
    (args.output / 'candidate_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
