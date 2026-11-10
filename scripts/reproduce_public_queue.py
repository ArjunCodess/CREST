"""Replay attributed CC0 anubisyy queue checkpoints on released histories.

The public feature modules and trained weights remain separate and retain their
Kaggle metadata/provenance. No full-day state reconstruction feeds this forecast.
"""
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
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    assets = args.assets.resolve()
    root = args.release_root.resolve()
    os.environ.update(TFB_DATA_ROOT=str(root), TFB_OUTPUT_ROOT=str(args.output.resolve()),
                      TFB_THREADS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, str(assets))
    if (assets/'_pydeps').exists():
        sys.path.insert(0, str(assets/'_pydeps'))
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from queue_features import Panel, panel_names, HISTORY, HORIZON
    from queue_ongoing import Panel as OngoingPanel

    onset = lgb.Booster(model_file=str(assets/'onset_lgb.txt'))
    ongoing = lgb.Booster(model_file=str(assets/'ongoing_lgb.txt'))
    cat = CatBoostClassifier()
    cat.load_model(str(assets/'onset.cbm'))
    settings = json.loads((assets/'onset_settings.json').read_text())
    pieces, audit = [], []
    for split in ('validation', 'private'):
        for panel_no, name in enumerate(panel_names()):
            panel, opanel = Panel(name, panel_no), OngoingPanel(name, panel_no)
            folder = root/'task2'/name/split
            windows = pd.read_csv(folder/'window_index.csv')
            histories = pd.read_parquet(folder/'window_history.parquet')
            template = pd.read_csv(folder/'sample_submission_queue.csv', dtype={'link_id': str})
            predictions = []
            for w in windows.itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin-pd.Timedelta(minutes=5), periods=HISTORY,
                                      freq='5min', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                h = histories[histories.window_id.eq(w.window_id)].copy()
                h['link_id'] = h.link_id.astype(str)
                # Keep the index type stable when the audit appends a future row.
                h['timestamp'] = pd.to_datetime(h.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                if pd.to_datetime(h.timestamp, utc=True).max() >= origin:
                    raise ValueError('Published recipe expects history strictly before T')

                def features(frame):
                    matrices = [frame.pivot(index='timestamp', columns='link_id', values=c).reindex(
                        index=times, columns=panel.links).to_numpy(np.float32)
                        for c in ('speed_kmh', 'flow_vph', 'is_score_eligible')]
                    speed, flow, eligible = matrices
                    if w.condition == 'queue_onset':
                        return panel.features(speed, flow, eligible, origin, w.condition)[-len(panel.links):], matrices
                    return opanel.features(speed, flow, np.nan_to_num(eligible, nan=0).astype(bool), origin, w.condition), matrices

                x, inputs = features(h)
                future = h.drop_duplicates('link_id').copy()
                future['timestamp'] = (origin+pd.Timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
                future['speed_kmh'], future['flow_vph'] = 999.0, 99999.0
                changed, changed_inputs = features(pd.concat([h, future], ignore_index=True))
                if not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(inputs, changed_inputs)):
                    raise ValueError('Post-origin perturbation changed extracted history')
                # pandas can change C/F strides after concat; float32 reductions
                # then differ by an ulp even though all extracted inputs match.
                if not np.allclose(x, changed, rtol=1e-6, atol=2e-7, equal_nan=True):
                    raise ValueError('Post-origin perturbation changed queue features')
                prediction = np.zeros((HORIZON, len(panel.links)), dtype=np.uint8)
                if w.condition == 'queue_onset':
                    p = (1-settings['gpu_weight'])*onset.predict(x, num_threads=4) + settings['gpu_weight']*cat.predict_proba(x, thread_count=4)[:, 1]
                    selected = p >= settings['threshold']
                    if not selected.any():
                        selected[np.argmax(p)] = True
                    prediction[-1] = selected
                else:
                    p = ongoing.predict(x, num_threads=4)
                    prediction = (p >= 0.4000000000000001).reshape(HORIZON, -1).astype(np.uint8)
                stamps = [(origin+pd.Timedelta(minutes=5*s)).strftime('%Y-%m-%dT%H:%M:%SZ') for s in range(1, HORIZON+1)]
                predictions.append(pd.DataFrame({'window_id': w.window_id,
                    'timestamp': np.repeat(stamps, len(panel.links)), 'link_id': np.tile(panel.links, HORIZON),
                    'queue_pred': prediction.ravel()}))
                finite = np.isfinite(x) & np.isfinite(changed)
                audit.append({'window_id': w.window_id, 'future_perturbation_changed_inputs': False,
                    'features_exact_bits': bool(np.array_equal(x, changed, equal_nan=True)),
                    'feature_max_roundoff': float(np.abs(x[finite]-changed[finite]).max()),
                    'features_match_within_float32_roundoff': True})
            keys = ['window_id', 'timestamp', 'link_id']
            merged = template[keys].merge(pd.concat(predictions), on=keys, how='left', validate='1:1')
            if merged.queue_pred.isna().any():
                raise ValueError('Missing queue output')
            pieces.append(merged)
            print(f'{name}/{split}: {len(merged)} rows', flush=True)
    queue = pd.concat(pieces, ignore_index=True)
    if len(queue) != 174000 or not queue.queue_pred.isin([0, 1]).all():
        raise ValueError('Queue coverage or binary values differ')
    if args.reference:
        ref = pd.read_csv(args.reference)
        keys = ['window_id', 'timestamp', 'link_id']
        matched = queue.merge(ref[keys+['queue_pred']], on=keys, suffixes=('', '_ref'), validate='1:1')
        if len(matched) != len(queue) or not matched.queue_pred.eq(matched.queue_pred_ref).all():
            raise ValueError('Replayed queues differ from the scored inference')
    dest = args.output/'queue.csv'
    queue.to_csv(dest, index=False)
    with dest.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    result = {'source': 'https://www.kaggle.com/datasets/anubisyy/trafficflowbench-086517-public-assets',
        'license': 'CC0-1.0', 'rows': len(queue), 'sha256': digest, 'windows': audit,
        'matches_scored_queue': True if args.reference else None, 'shipped_predictions_used': False}
    (args.output/'replay_audit.json').write_text(json.dumps(result, indent=2))
    print(f'Verified {len(queue)} queue cells and {len(audit)} causal perturbation checks', flush=True)


if __name__ == '__main__':
    main()
