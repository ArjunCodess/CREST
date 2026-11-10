"""Matched, bounded v5-control versus v7 pilots. Never replaces production files."""
import argparse
import gc
import json
import pickle
from pathlib import Path
import sys
import time
import hashlib
import platform
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_state, graph
from src.state_v4 import EnhancedReconstructor
from src.wave_v7 import WaveReconstructor, MotionQueueModel
from src.queue_v5 import LocalizedQueueModel
from src.queue_events import load_auxiliary, collect_events, visible_event, decode_queue
from src.evaluation import state_metrics, proxy_queue_iou
from src.holdout import official_cell_mask, _regimes
from src.physics import project_fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', action='append', required=True)
    parser.add_argument('--task', choices=['state', 'queue'], required=True)
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--max-rows', type=int, default=160000)
    parser.add_argument('--events', type=int, default=256)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--fit-start', default='2030-06-01T00:00:00Z')
    parser.add_argument('--fit-end', default='2030-09-30T12:00:00Z')
    args = parser.parse_args()
    if pd.Timestamp(args.fit_start) >= pd.Timestamp(args.fit_end) or pd.Timestamp(args.fit_end) > pd.Timestamp('2030-10-14T12:00:00Z'):
        parser.error('Fitting must end at least 12 hours before October selection and start before its end.')
    if not (args.release_root / 'corridors').is_dir():
        parser.error('Official release is missing; restore it before training.')
    if (args.release_root / 'FIXTURE_ONLY.json').exists():
        parser.error('Fixture data cannot establish a real-data promotion.')
    if args.task == 'queue' and any(p.startswith('D12_I405_') for p in args.panel):
        parser.error('D12 I405 is excluded from official Task 2.')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(development_only=True, production_promoted=False,
                    control='v5 architecture refit on identical prefix; not deployed all-data checkpoint',
                    queue_scope='ongoing only; onset preserved',
                    masks='published hash over eligible training cells; scored-split protection remains template-specific')
    source_root = Path(__file__).resolve().parents[1]
    protocol['source_sha256'] = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('src', 'scripts') for path in (source_root/folder).glob('*.py')}
    protocol['python'] = platform.python_version()
    protocol['threads'] = 4
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    started = time.monotonic()
    records, decisions = [], []
    for panel in args.panel:
        print(panel, 'loading fitting prefix', flush=True)
        d = load_state(args.release_root, panel, 'train', True, start=args.fit_start, end=args.fit_end)
        up, down, _, params = graph(args.release_root, panel, d.links)
        with threadpool_limits(limits=4):
            if args.task == 'state':
                models = [EnhancedReconstructor(args.rounds, args.max_rows), WaveReconstructor(args.rounds, args.max_rows)]
                for model in models:
                    print(panel, 'fitting', type(model).__name__, flush=True)
                    model.fit(d.values, d.times, up, down, params)
            else:
                aux = load_auxiliary(args.release_root, panel, d.times, d.links)
                events = collect_events(d.values, aux, d.times, np.ones(len(d.times), bool),
                    params.free_speed_kmh.to_numpy()*.6, np.random.default_rng(707), max_per_condition=args.events)
                models = [LocalizedQueueModel(rounds=args.rounds), MotionQueueModel(rounds=args.rounds)]
                for model in models:
                    print(panel, 'fitting', type(model).__name__, 'on', len(events), 'proxy events', flush=True)
                    model.fit(events, up, down, params)
                del aux, events
        with (args.output / f'{panel}_models.pkl').open('wb') as f:
            pickle.dump({'control': models[0], 'candidate': models[1], 'links': d.links, 'protocol': protocol}, f)
        del d
        gc.collect()
        selection_weight = 0.
        selection_policy = .3
        panel_scores = {}
        for block, date in [('selection', '2030-10-15'), ('november', '2030-11-15'), ('january', '2031-01-15')]:
            print(panel, 'evaluating', block, flush=True)
            begin = pd.Timestamp(date+'T00:00:00Z')
            data = load_state(args.release_root, panel, 'train', True,
                start=begin, end=begin+pd.Timedelta(days=args.days))
            if args.task == 'state':
                mask = official_cell_mask(panel, data.times, data.links) & np.isfinite(data.values).all(-1)
                observed = data.values.copy(); observed[mask] = np.nan
                with threadpool_limits(limits=4):
                    predictions = [m.reconstruct(observed, data.times, up, down, params) for m in models]
                regimes = _regimes(panel, data.times)
                scores = {}
                for weight in ([0., .25, .5, 1.] if block == 'selection' else [0., selection_weight]):
                    prediction = project_fd((1-weight)*predictions[0]+weight*predictions[1], params)
                    parts = []
                    for regime in ('R1', 'R2', 'R3'):
                        target = mask & (regimes == regime)[:, None]
                        if target.any():
                            metric = state_metrics(data.values, prediction, params.lanes, target)
                            parts.append(metric['S_state'])
                            records.append({'panel': panel, 'block': block, 'weight': weight, 'regime': regime, **metric})
                    scores[weight] = float(np.mean(parts)) if len(parts) == 3 else None
                if any(v is None for v in scores.values()):
                    raise ValueError('All three regimes must be represented; increase --days')
                if block == 'selection':
                    selection_weight = max(scores, key=scores.get)
                panel_scores[block] = scores[selection_weight]-scores[0.]
            else:
                auxiliary = load_auxiliary(args.release_root, panel, data.times, data.links)
                events = collect_events(data.values, auxiliary, data.times, np.ones(len(data.times), bool),
                    params.free_speed_kmh.to_numpy()*.6, np.random.default_rng(708), max_per_condition=args.events)
                events = [e for e in events if e.condition == 'queue_ongoing']
                if len(events) < 12:
                    raise ValueError('Too few independent ongoing events; increase --days')
                cache = []
                for e in events:
                    h, a = visible_event(e, models[0].policy)
                    cache.append([m.scores(h, a, e.times, e.condition, up, down, params) for m in models])
                if block == 'selection':
                    policies = (.1, .2, .3, .4, .5, .6, 'expected_iou')
                    control_policy = max(policies, key=lambda policy: np.mean([
                        proxy_queue_iou(e.truth, decode_queue(p[0], policy), e.valid) for e, p in zip(events, cache)]))
                    trials = []
                    for weight in (.25, .5, 1.):
                        for policy in policies:
                            score = np.mean([proxy_queue_iou(e.truth, decode_queue((1-weight)*p[0]+weight*p[1], policy), e.valid) for e, p in zip(events, cache)])
                            trials.append((score, weight, policy))
                    _, selection_weight, selection_policy = max(trials, key=lambda x: x[0])
                deltas = []
                for e, p in zip(events, cache):
                    old = proxy_queue_iou(e.truth, decode_queue(p[0], control_policy), e.valid)
                    new = proxy_queue_iou(e.truth, decode_queue((1-selection_weight)*p[0]+selection_weight*p[1], selection_policy), e.valid)
                    deltas.append(new-old)
                    records.append({'panel': panel, 'block': block, 'day': str(e.times[-1].date()), 'control': old, 'candidate': new})
                panel_scores[block] = float(np.mean(deltas))
            pd.DataFrame(records).to_csv(args.output / 'metrics.csv', index=False)
        threshold = .001 if args.task == 'state' else .01
        decisions.append({'panel': panel, 'selected_weight': selection_weight, 'selected_decoder': selection_policy,
                          'development_deltas': panel_scores,
                          'pilot_gate_passed': all(panel_scores[b] >= threshold for b in ('november', 'january')),
                          'production_promoted': False,
                          'remaining_gates': 'full directions, worst-family/regime retention, event uncertainty, deployed-policy comparison, exact reproduction'})
        (args.output / 'decisions.json').write_text(json.dumps(decisions, indent=2))
        print(json.dumps(decisions[-1]), flush=True)
        del models, data
        gc.collect()
    (args.output / 'runtime.json').write_text(json.dumps({'seconds': time.monotonic()-started}))


if __name__ == '__main__':
    main()
