"""Chronological queue ablations on training observations, never official labels."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state, graph, family
from src.evaluation import proxy_queue_iou
from src.queue_events import load_auxiliary, calibrate_occupancy_density, collect_events, visible_event
from src.queue_onset import onset_events
from src.latent_queue_v6 import (normalize_speed, estimate_noise, smooth_training_series,
    smooth_series, assert_events_before, horizon_targets, QueueArmModel)
from src.queue_scenarios_v6 import (binary_adjacency, OnsetSeedModel, apply_decoder,
    flip_rates, aggregate_macro, paired_day_block_bootstrap, topk_set_hit, set_mass_nll,
    directed_offsets, seed_footprint_decode, persist_latest, isolation_share, missed_false_by_horizon)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True)
    ap.add_argument('--rounds', type=int, default=180)
    ap.add_argument('--train-events', type=int, default=128)
    ap.add_argument('--eval-events', type=int, default=32)
    ap.add_argument('--fit-start', default='2030-06-01T00:00:00Z')
    ap.add_argument('--scenarios', type=int, default=128)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists():
        raise ValueError('Use a fresh output directory')
    protocol = dict(vars(args), release_root=str(args.release_root), output=str(args.output),
        fit_end='2030-09-30T22:30:00Z', selection='October', development=['November', 'January'],
        policy='origin_observed', target_arms=['raw', 'smoothed_hard', 'smoothed_soft'],
        decoder='raw arm selects October decoder separately per condition; all label arms share it',
        decoder_control='scenario versus expected-IoU surrogate on identical raw probabilities',
        selection_target='raw proxy only', seed_labels='smoothed hard T+30 on raw-selected onset windows',
        assessment='Matched raw-selected events; smoothed truth is a sensitivity proxy, not latent truth',
        gate='Smoothed gain >= .01, raw loss <= .005 in both development blocks; flip share >= .03',
        promotion='No production promotion from these previously inspected development months')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    records, diagnostics, seed_rows, error_rows = [], [], [], []
    started = time.monotonic()
    for panel in args.panel:
        print(panel, 'load fit', flush=True)
        d = load_state(args.release_root, panel, 'train', True, start=args.fit_start,
                       end=protocol['fit_end'])
        auxiliary = load_auxiliary(args.release_root, panel, d.times, d.links)
        up, down, _, params = graph(args.release_root, panel, d.links)
        adjacency = binary_adjacency(args.release_root, panel, d.links)
        offsets = directed_offsets(adjacency)
        density_scale = calibrate_occupancy_density(auxiliary)
        auxiliary[..., 1] = auxiliary[..., 0]*density_scale
        normalized = normalize_speed(d.values, params, auxiliary)
        noise = estimate_noise(normalized)
        smooth = smooth_training_series(normalized, noise, d.times, '2030-10-01T00:00:00Z')
        active = np.ones(len(d.times), bool)
        cutoff = .6*params.free_speed_kmh.to_numpy()
        events = [e for e in collect_events(d.values, auxiliary, d.times, active, cutoff,
            np.random.default_rng(606), max_per_condition=args.train_events) if e.condition == 'queue_ongoing']
        events += onset_events(d.values, auxiliary, d.times, active, cutoff,
                               max_events=args.train_events, seed=606)
        assert_events_before(events, d.times, '2030-10-01T00:00:00Z')
        counts = {c: sum(e.condition == c for e in events) for c in ('queue_onset', 'queue_ongoing')}
        if not all(counts.values()):
            raise ValueError(f'{panel}: missing training condition: {counts}')
        print(panel, 'fit events', counts, flush=True)
        targets = {arm: {} for arm in protocol['target_arms']}
        for event in events:
            for arm, target in horizon_targets(event, smooth.probability, smooth.hard).items():
                targets[arm][event.origin, event.condition] = target
        models = {}
        for arm in protocol['target_arms']:
            models[arm] = QueueArmModel(arm, rounds=args.rounds, rows_per_window=600).fit(
                events, up, down, params, noise, targets[arm])
        seed_events = [replace(e, truth=targets['smoothed_hard'][e.origin, e.condition].astype(bool))
                       for e in events if e.condition == 'queue_onset']
        ranker = OnsetSeedModel(rounds=args.rounds, min_groups=12).fit(
            seed_events, up, down, params, noise, adjacency)
        diagnostics.append({'panel': panel, 'block': 'fit', 'events': counts,
            'noise_assumption': noise.assumption, 'observation_variance_median': float(np.median(noise.observation_variance)),
            'flips': flip_rates((normalized <= .6).reshape(1, -1), smooth.hard.reshape(1, -1), np.isfinite(normalized).reshape(1, -1)),
            'seed_status': ranker.status, 'footprint': ranker.footprint})
        links = d.links
        del d, auxiliary, normalized, smooth, targets, events, seed_events
        gc.collect()
        decoders = {}
        for block, month in [('selection', '2030-10'), ('november', '2030-11'), ('january', '2031-01')]:
            begin = pd.Timestamp(month+'-01T00:00:00Z')
            d = load_state(args.release_root, panel, 'train', True, start=begin, end=begin+pd.offsets.MonthBegin(1))
            if d.links != links:
                raise ValueError('Network ordering changed')
            auxiliary = load_auxiliary(args.release_root, panel, d.times, d.links)
            auxiliary[..., 1] = auxiliary[..., 0]*density_scale
            normalized = normalize_speed(d.values, params, auxiliary)
            # Evaluation smoothing supplies sensitivity labels only, with the noise
            # parameters frozen before October. It never enters model features.
            smooth = smooth_series(normalized, noise)
            active = np.ones(len(d.times), bool)
            events = [e for e in collect_events(d.values, auxiliary, d.times, active, cutoff,
                np.random.default_rng(607), max_per_condition=args.eval_events) if e.condition == 'queue_ongoing']
            events += onset_events(d.values, auxiliary, d.times, active, cutoff,
                                   max_events=args.eval_events, seed=607)
            counts = {c: sum(e.condition == c for e in events) for c in ('queue_onset', 'queue_ongoing')}
            if not all(counts.values()):
                raise ValueError(f'{panel}/{block}: missing evaluation condition: {counts}')
            cache = [(e, {arm: model.probabilities(e, up, down, params) for arm, model in models.items()}) for e in events]
            if block == 'selection':
                for condition in counts:
                    trials = []
                    for policy in (.2, .35, .5, .65, 'expected_iou'):
                        score = np.mean([proxy_queue_iou(e.truth, apply_decoder(p['raw'], condition, policy), e.valid)
                                         for e, p in cache if e.condition == condition])
                        trials.append((float(score), policy))
                    decoders[condition] = max(trials, key=lambda pair: pair[0])[1]
                ranker.select_temperature(events, up, down, params, noise)
                (args.output/f'{panel}_selection.json').write_text(json.dumps({
                    'decoders': decoders, 'seed_temperature': ranker.temperature, 'seed_status': ranker.status}, indent=2))
                with (args.output/f'{panel}_models.pkl').open('wb') as f:
                    pickle.dump({'models': models, 'seed': ranker, 'links': links, 'protocol': protocol,
                                 'decoders': decoders, 'density_scale': density_scale}, f)
            diagnostics.append({'panel': panel, 'block': block, 'events': counts,
                'flips': flip_rates((normalized <= .6).reshape(1, -1), smooth.hard.reshape(1, -1), np.isfinite(normalized).reshape(1, -1))})
            for event, probabilities in cache:
                hard = smooth.hard[event.origin+1:event.origin+7].copy()
                if event.condition == 'queue_onset':
                    hard[:-1] = False
                predictions = {arm: apply_decoder(p, event.condition, decoders[event.condition])
                               for arm, p in probabilities.items()}
                predictions['raw_surrogate'] = apply_decoder(probabilities['raw'], event.condition, 'expected_iou')
                predictions['raw_scenario'] = apply_decoder(probabilities['raw'], event.condition, 'scenario',
                    n_scenarios=args.scenarios, seed=606+event.origin)
                history, _ = visible_event(event, protocol['policy'])
                predictions['persistence'] = persist_latest(history[..., 0]/params.free_speed_kmh.to_numpy())
                if event.condition == 'queue_onset':
                    predictions['persistence'][:-1] = False
                    seed = ranker.probabilities(event, up, down, params, noise)[-1]
                    decoded, joint = seed_footprint_decode(seed, probabilities['raw'][-1], offsets,
                        ranker.footprint, n_scenarios=args.scenarios, seed=606+event.origin)
                    predictions['seed_footprint'] = np.zeros_like(event.truth)
                    predictions['seed_footprint'][-1] = decoded
                    seed_rows.append({'panel': panel, 'block': block, 'origin': str(event.times[-1]),
                        'status': ranker.status, **joint})
                else:
                    predictions['seed_footprint'] = predictions['raw']
                error_rows.append({'panel': panel, 'block': block, 'condition': event.condition,
                    'origin': str(event.times[-1]), 'raw_errors': missed_false_by_horizon(event.truth, predictions['raw'], event.valid),
                    'raw_isolation': isolation_share(predictions['raw'], adjacency),
                    'truth_isolation': isolation_share(event.truth & event.valid, adjacency),
                    'label_flips': flip_rates(event.truth, hard, event.valid)})
                for target, truth in [('raw', event.truth), ('smoothed', hard)]:
                    row = {'panel': panel, 'family': family(panel), 'block': block, 'target': target,
                           'condition': event.condition, 'day': str(event.times[-1].date()), 'origin': str(event.times[-1])}
                    row.update({name: proxy_queue_iou(truth, pred, event.valid) for name, pred in predictions.items()})
                    records.append(row)
                if event.condition == 'queue_onset' and ranker.status == 'fitted':
                    seed = ranker.probabilities(event, up, down, params, noise)[-1]
                    positive = hard[-1] & event.valid[-1]
                    seed_rows.append({'panel': panel, 'block': block, 'origin': str(event.times[-1]),
                        'top1': topk_set_hit(seed, positive, 1), 'top3': topk_set_hit(seed, positive, 3),
                        'positive_links': int(positive.sum()), 'set_mass_nll': set_mass_nll(seed, positive)})
            pd.DataFrame(records).to_csv(args.output/'windows.csv', index=False)
            pd.DataFrame(seed_rows).to_csv(args.output/'seed_metrics.csv', index=False)
            (args.output/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
            (args.output/'queue_errors.json').write_text(json.dumps(error_rows, indent=2))
            print(panel, block, 'evaluated', counts, flush=True)
            del d, auxiliary, normalized, smooth, events, cache
            gc.collect()
        del models, ranker
    summary = []
    for block in ('selection', 'november', 'january'):
        for target in ('raw', 'smoothed'):
            rows = [r for r in records if r['block'] == block and r['target'] == target]
            for candidate, baseline in [('smoothed_hard', 'raw'), ('smoothed_soft', 'raw'), ('raw_scenario', 'raw_surrogate'), ('seed_footprint', 'raw'), ('persistence', 'raw')]:
                summary.append({'block': block, 'target': target, 'candidate': candidate, 'baseline': baseline,
                    'score': aggregate_macro(rows, candidate),
                    'paired_difference': paired_day_block_bootstrap(rows, candidate, baseline),
                    'production_promoted': False})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'runtime.json').write_text(json.dumps({'seconds': time.monotonic()-started}, indent=2))


if __name__ == '__main__':
    main()
