"""Paired day uncertainty for a frozen state pilot policy, with fixed regime weights."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_state, graph
from src.holdout import official_cell_mask, _regimes
from src.physics import project_fd


def utility(sums):
    speed, flow, count = np.asarray(sums).sum(0)
    return .54*max(0., 1-np.sqrt(speed/count)/25)+.46*max(0., 1-np.sqrt(flow/count)/600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    decisions = json.loads((args.pilot/'decisions.json').read_text())
    decision = next(row for row in decisions if row['panel'] == args.panel)
    if not decision['pilot_gate_passed']:
        parser.error('Only frozen policies that passed the pilot gate can enter this audit.')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.pilot/f'{args.panel}_models.pkl'
    with checkpoint.open('rb') as source:
        payload = pickle.load(source)
    protocol = json.loads((args.pilot/'protocol.json').read_text())
    rows, checks = [], []
    for block, date in [('november', '2030-11-15'), ('january', '2031-01-15')]:
        begin = pd.Timestamp(date+'T00:00:00Z')
        data = load_state(args.release_root, args.panel, 'train', True, start=begin,
                          end=begin+pd.Timedelta(days=protocol['days']))
        if data.links != payload['links']:
            raise ValueError('Network ordering changed')
        up, down, _, params = graph(args.release_root, args.panel, data.links)
        mask = official_cell_mask(args.panel, data.times, data.links) & np.isfinite(data.values).all(-1)
        observed = data.values.copy(); observed[mask] = np.nan
        with threadpool_limits(limits=4):
            old = payload['control'].reconstruct(observed, data.times, up, down, params)
            candidate = payload['candidate'].reconstruct(observed, data.times, up, down, params)
        weight = decision['selected_weight']
        predictions = [project_fd(old, params), project_fd((1-weight)*old+weight*candidate, params)]
        dates = data.times.strftime('%Y-%m-%d')
        regimes = _regimes(args.panel, data.times)
        grouped = {}
        for regime in ('R1', 'R2', 'R3'):
            days = []
            for day in np.unique(dates[regimes == regime]):
                target = mask & (dates == day)[:, None]
                if not target.any():
                    continue
                values = []
                for name, pred in zip(('control', 'candidate'), predictions):
                    speed = float(np.square(data.values[..., 0][target]-pred[..., 0][target]).sum(dtype=np.float64))
                    flow = float(np.square(((data.values[..., 1]-pred[..., 1])/params.lanes.to_numpy()[None])[target]).sum(dtype=np.float64))
                    summary = [speed, flow, int(target.sum())]
                    values.append(summary)
                    rows.append({'block': block, 'regime': regime, 'day': day, 'model': name,
                                 'speed_sse': speed, 'flow_lane_sse': flow, 'count': summary[2]})
                days.append(values)
            if len(days) < 2:
                raise ValueError('Each regime needs at least two independent development days')
            grouped[regime] = np.asarray(days)
        rng = np.random.default_rng(771)
        differences = []
        for _ in range(1000):
            delta = []
            for days in grouped.values():
                sample = days[rng.integers(len(days), size=len(days))]
                delta.append(utility(sample[:, 1])-utility(sample[:, 0]))
            differences.append(np.mean(delta))
        regime_deltas = {regime: utility(days[:, 1])-utility(days[:, 0]) for regime, days in grouped.items()}
        checks.append({'block': block, 'delta': float(np.mean(list(regime_deltas.values()))),
                       'regime_deltas': regime_deltas,
                       'paired_day_p05': float(np.quantile(differences, .05)),
                       'paired_day_p95': float(np.quantile(differences, .95))})
        pd.DataFrame(rows).to_csv(args.output/'daily_errors.csv', index=False)
        print(json.dumps(checks[-1]), flush=True)
    result = {'panel': args.panel, 'selected_weight': decision['selected_weight'], 'checks': checks,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'state_audit_passed': all(row['paired_day_p05'] > 0 and min(row['regime_deltas'].values()) >= -.002 for row in checks),
              'production_promoted': False,
              'limitations': 'Previously inspected development months; conditional bootstrap on fixed candidate, not official physics or private uncertainty. Official state/physics trial and exact replay still required.'}
    (args.output/'audit.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
