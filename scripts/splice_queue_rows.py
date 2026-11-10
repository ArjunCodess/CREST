"""Replace queue_pred in an existing submission from a Task 2 queue CSV.

Streams the audited submission line-by-line so state and ODME rows stay byte-identical.
Requires submission_key.csv in parallel row order to locate queue natural keys.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']
QUEUE_KEY_COLS = ['window_id', 'timestamp', 'link_id']
KEY_USECOLS = ['submission_id', 'task', 'window_id', 'timestamp', 'link_id']
MIN_FREE_BYTES = 500 * 1024 * 1024
CHUNK = 250_000


def norm_timestamp(value: str) -> str:
    return pd.Timestamp(value, tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def format_queue_pred(value: float) -> str:
    v = int(value)
    if v not in (0, 1):
        raise ValueError(f'queue_pred must be 0 or 1, got {value!r}')
    return f'{v}.0'


def drive_root(path: Path) -> str:
    resolved = path.resolve()
    anchor = resolved.anchor
    if not anchor:
        raise ValueError(f'Cannot resolve drive for {path}')
    return anchor


def check_free_space(path: Path) -> int:
    free = shutil.disk_usage(drive_root(path)).free
    if free < MIN_FREE_BYTES:
        mb = free / (1024 * 1024)
        raise SystemExit(
            f'Refusing: {drive_root(path)} has {mb:.1f} MB free; need at least 500 MB for a partial upload.'
        )
    return free


def load_queue_index(queue_path: Path) -> dict[tuple[str, str, str], int]:
    required = set(QUEUE_KEY_COLS + ['queue_pred'])
    index: dict[tuple[str, str, str], int] = {}
    for chunk in pd.read_csv(queue_path, dtype=str, chunksize=100_000):
        if not required.issubset(chunk.columns):
            missing = required - set(chunk.columns)
            raise ValueError(f'{queue_path} missing columns: {sorted(missing)}')
        extra = set(chunk.columns) - required
        if extra:
            raise ValueError(f'{queue_path} has unexpected columns: {sorted(extra)}')
        values = pd.to_numeric(chunk['queue_pred'], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('queue_pred contains non-finite or negative values')
        if not np.isin(values, [0, 1]).all():
            raise ValueError('queue_pred must be binary (0 or 1)')
        for wid, ts, link, pred in zip(
            chunk['window_id'].astype(str),
            chunk['timestamp'].map(norm_timestamp),
            chunk['link_id'].astype(str),
            values.astype(int),
        ):
            key = (wid, ts, link)
            if key in index:
                raise ValueError(f'Duplicate queue key {key!r} in {queue_path}')
            index[key] = int(pred)
    if not index:
        raise ValueError(f'No queue rows loaded from {queue_path}')
    return index


def open_key_reader(key_path: Path):
    handle = key_path.open('r', newline='', encoding='utf-8')
    reader = csv.DictReader(handle)
    expected = set(KEY_USECOLS)
    if not expected.issubset(set(reader.fieldnames or [])):
        missing = expected - set(reader.fieldnames or [])
        raise ValueError(f'{key_path} missing columns: {sorted(missing)}')

    def rows():
        try:
            for row in reader:
                yield row
        finally:
            handle.close()

    return rows()


def validate_output(submission: Path, key_path: Path, expected_rows: int) -> None:
    key_reader = open_key_reader(key_path)
    n = 0
    prev_id: int | None = None
    with submission.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != COLUMNS:
            raise ValueError(f'Unexpected columns in {submission}')
        for row, key_row in zip(reader, key_reader):
            if len(row) != len(COLUMNS):
                raise ValueError(f'Row {n + 1} has {len(row)} columns, expected {len(COLUMNS)}')
            sid = int(row[0])
            if prev_id is not None and sid != prev_id + 1:
                raise ValueError(f'submission_id order break at {sid} (previous {prev_id})')
            prev_id = sid
            if row[1] != key_row['task'] or row[0] != key_row['submission_id']:
                raise ValueError('submission and key task/id mismatch')
            nums = np.asarray([float(row[i]) for i in range(2, len(COLUMNS))], dtype=float)
            if not np.isfinite(nums).all() or (nums < 0).any():
                raise ValueError(f'Non-finite or negative value on row {sid}')
            if row[1] == 'queue':
                qp = nums[2]
                if qp not in (0.0, 1.0):
                    raise ValueError(f'queue_pred not binary on row {sid}: {qp}')
            n += 1
    if n != expected_rows:
        raise ValueError(f'Row count {n} != expected {expected_rows}')
    if next(key_reader, None) is not None:
        raise ValueError('submission_key has extra rows after submission')


def splice(
    submission: Path,
    key_path: Path,
    queue_index: dict[tuple[str, str, str], int],
    partial: Path,
    *,
    apply: bool,
) -> dict[str, int]:
    stats = {
        'rows': 0,
        'queue_rows': 0,
        'matched': 0,
        'missing_key': 0,
        'changed': 0,
        'unchanged': 0,
    }
    key_iter = open_key_reader(key_path)
    out_handle = partial.open('w', newline='', encoding='utf-8') if apply else None
    try:
        with submission.open('r', encoding='utf-8') as src:
            header = src.readline()
            if apply:
                assert out_handle is not None
                out_handle.write(header)
            if header.strip() != ','.join(COLUMNS):
                raise ValueError(f'Unexpected header in {submission}')
            for line in src:
                stats['rows'] += 1
                body = line.rstrip('\r\n')
                parts = body.split(',')
                if len(parts) != len(COLUMNS):
                    raise ValueError(f'Row {stats["rows"]} has {len(parts)} fields')
                try:
                    key_row = next(key_iter)
                except StopIteration as exc:
                    raise ValueError('submission.csv has more rows than submission_key.csv') from exc
                if parts[0] != key_row['submission_id'] or parts[1] != key_row['task']:
                    raise ValueError(f'Row {stats["rows"]}: submission/key id or task mismatch')
                if parts[1] != 'queue':
                    if apply:
                        out_handle.write(line if line.endswith('\n') else line + '\n')
                    continue
                stats['queue_rows'] += 1
                wid = str(key_row['window_id'])
                ts = norm_timestamp(str(key_row['timestamp']))
                link = str(key_row['link_id'])
                lookup = (wid, ts, link)
                if lookup not in queue_index:
                    stats['missing_key'] += 1
                    raise ValueError(f'Missing queue prediction for key {lookup!r}')
                stats['matched'] += 1
                new_pred = queue_index[lookup]
                old_pred = int(float(parts[4]))
                if new_pred == old_pred:
                    stats['unchanged'] += 1
                    if apply:
                        out_handle.write(line if line.endswith('\n') else line + '\n')
                else:
                    stats['changed'] += 1
                    if apply:
                        parts[4] = format_queue_pred(new_pred)
                        out_handle.write(','.join(parts) + '\n')
        try:
            extra = next(key_iter)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError('submission_key.csv has rows not present in submission.csv')
    finally:
        if out_handle is not None:
            out_handle.close()
    return stats


def count_submission_rows(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        total = sum(1 for _ in handle)
    return total - 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission', type=Path, required=True, help='Existing full submission.csv')
    ap.add_argument('--queue', type=Path, required=True, help='New queue.csv (window_id,timestamp,link_id,queue_pred)')
    ap.add_argument(
        '--key',
        type=Path,
        default=root / 'data' / 'kaggle_public' / 'submission_key.csv',
        help='submission_key.csv in official template order',
    )
    ap.add_argument('--apply', action='store_true', help='Write partial file and os.replace into --submission')
    args = ap.parse_args()

    submission = args.submission.resolve()
    queue_path = args.queue.resolve()
    key_path = args.key.resolve()
    for label, path in (('submission', submission), ('queue', queue_path), ('key', key_path)):
        if not path.is_file():
            raise SystemExit(f'Missing {label}: {path}')

    queue_index = load_queue_index(queue_path)
    expected_rows = count_submission_rows(submission)
    partial = submission.with_suffix('.partial.csv')

    if args.apply:
        check_free_space(submission)
        if partial.exists():
            raise SystemExit(f'Refusing: partial file already exists: {partial}')
        stats = splice(submission, key_path, queue_index, partial, apply=True)
        validate_output(partial, key_path, expected_rows)
        os.replace(partial, submission)
        mode = 'applied'
    else:
        stats = splice(submission, key_path, queue_index, partial, apply=False)
        mode = 'dry-run'

    print(
        f'{mode}: submission={submission}\n'
        f'  queue_csv_rows={len(queue_index):,}\n'
        f'  submission_rows={stats["rows"]:,} queue_rows={stats["queue_rows"]:,}\n'
        f'  matched={stats["matched"]:,} missing={stats["missing_key"]:,}\n'
        f'  queue_pred_changed={stats["changed"]:,} unchanged={stats["unchanged"]:,}'
    )
    if stats['queue_rows'] != len(queue_index):
        print(
            f'  note: queue.csv has {len(queue_index):,} keys; submission has {stats["queue_rows"]:,} queue rows',
            file=sys.stderr,
        )


if __name__ == '__main__':
    main()
