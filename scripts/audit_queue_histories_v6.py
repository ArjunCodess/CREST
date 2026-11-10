"""Describe threshold crossings in released queue inputs, without hidden labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import network


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory')
    rows = []
    for panel_dir in sorted((args.release_root/'task2').iterdir()):
        panel = panel_dir.name
        free = network(args.release_root, panel).set_index('link_id').free_speed_kmh
        for split in ('validation', 'private'):
            folder = panel_dir/split
            windows = pd.read_csv(folder/'window_index.csv', dtype={'window_id': str})
            history = pd.read_parquet(folder/'window_history.parquet', columns=[
                'window_id', 'timestamp', 'link_id', 'speed_kmh', 'pct_observed'])
            history.timestamp = pd.to_datetime(history.timestamp, utc=True)
            history.link_id = history.link_id.astype(str)
            for window in windows.itertuples(index=False):
                origin = pd.Timestamp(window.forecast_origin)
                data = history[history.window_id.eq(window.window_id)]
                if not data.timestamp.between(origin-pd.Timedelta(minutes=60), origin).all():
                    raise ValueError('Released history crosses its permitted window')
                eligible = data.pct_observed.ge(75) & data.speed_kmh.notna()
                cutoff = data.link_id.map(free)*.6
                if cutoff.isna().any():
                    raise ValueError('History link lacks free-flow parameters')
                queued = eligible & data.speed_kmh.le(cutoff)
                before = data.timestamp.lt(origin)
                rows.append({'panel': panel, 'split': split, 'window_id': window.window_id,
                    'condition': window.condition, 'history_seen_cells': int((eligible & before).sum()),
                    'history_queued_cells': int((queued & before).sum()),
                    'history_has_crossing': bool((queued & before).any()),
                    'origin_rows_in_history': int((~before).sum()),
                    'origin_queued_cells_in_history': int((queued & ~before).sum())})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'windows.csv', index=False)
    summary = frame.groupby(['split', 'condition']).agg(
        windows=('window_id', 'size'), windows_with_crossing=('history_has_crossing', 'sum'),
        crossing_window_share=('history_has_crossing', 'mean'),
        observed_cells=('history_seen_cells', 'sum'), queued_cells=('history_queued_cells', 'sum'))
    summary.to_csv(args.output/'summary.csv')
    (args.output/'interpretation.json').write_text(json.dumps({
        'inputs': 'Released window_history only; no forecasts, labels or monthly reconstructions.',
        'time': 'History statistic excludes T; any T rows in window_history are counted separately.',
        'interpretation': 'An onset-history threshold crossing is not an identified false positive. The official condition refers to latent history, while these are observations.',
        'selection_use': 'Descriptive audit only; no thresholds or checkpoints selected from scored inputs.'}, indent=2))
    print(summary.to_string())


if __name__ == '__main__':
    main()
