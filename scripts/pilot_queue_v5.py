"""Compare localized forecasts with v4 on chronological training-only episodes."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state, graph
from src.queue_events import (load_auxiliary, calibrate_occupancy_density,
    collect_events, EventQueueModel, visible_event, decode_queue)
from src.queue_onset import onset_events, decode_onset
from src.queue_v5 import LocalizedQueueModel
from src.evaluation import proxy_queue_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True)
    ap.add_argument('--events', type=int, default=256)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    event_manifest = json.loads(Path('artifacts/v4_delivery/models/events/model_manifest.json').read_text())
    selected_events = {r['panel']: r for r in event_manifest['models']}
    for panel in args.panel:
        if (args.output / f'{panel}_decision.json').exists():
            continue
        decision = json.loads(Path(f'artifacts/onset_v4/pilot/{panel}_decision.json').read_text())
        policy = selected_events.get(panel, {}).get('origin_policy', 'origin_observed')
        d = load_state(args.release_root, panel, 'train', True, end='2031-02-01T00:00:00Z')
        auxiliary = load_auxiliary(args.release_root, panel, d.times, d.links)
        up, down, _, params = graph(args.release_root, panel, d.links)
        fit = np.asarray(d.times < pd.Timestamp('2030-09-30T22:30:00Z'))
        density_scale = calibrate_occupancy_density(auxiliary[fit]); auxiliary[..., 1] = auxiliary[..., 0]*density_scale
        cutoff = params.free_speed_kmh.to_numpy()*.6
        groups = {'fit': fit}
        for name, date in [('selection','2030-10'),('november','2030-11'),('january','2031-01')]:
            start = pd.Timestamp(date+'-01T00:00:00Z')
            groups[name] = np.asarray((d.times >= start) & (d.times < start+pd.offsets.MonthBegin(1)))
        events = {}
        for name, active in groups.items():
            ongoing = [e for e in collect_events(d.values, auxiliary, d.times, active,
                cutoff, np.random.default_rng(851), max_per_condition=args.events if name=='fit' else 64)
                if e.condition == 'queue_ongoing']
            events[name] = ongoing + onset_events(d.values, auxiliary, d.times, active, cutoff,
                max_events=args.events if name=='fit' else 64, seed=851)
        print(panel, 'episodes', {k:len(v) for k,v in events.items()}, flush=True)
        baseline = EventQueueModel(policy=policy).fit(events['fit'], up, down, params)
        baseline.select_decoders(events['selection'], up, down, params)
        # The v4 onset prefix checkpoint was fitted before October, so it can
        # be evaluated on these later training months without refitting on them.
        kind = 'onset' if decision['accepted'] else 'baseline'
        with Path(f'artifacts/onset_v4/pilot/{panel}_{kind}.pkl').open('rb') as f:
            onset = pickle.load(f)['model']
        candidate = LocalizedQueueModel(policy=policy).fit(events['fit'], up, down, params)
        cache = {}
        for block in ('selection','november','january'):
            rows=[]
            for e in events[block]:
                h,a = visible_event(e,policy)
                new = candidate.scores(h,a,e.times,e.condition,up,down,params)
                if e.condition == 'queue_onset':
                    oh,oa = visible_event(e,onset.policy)
                    old = onset.scores(oh,oa,e.times,e.condition,up,down,params)
                    decoder = onset.decoder if kind=='onset' else decision['baseline_decoder']
                    old_pred = decode_onset(old,decoder)
                else:
                    old = baseline.scores(h,a,e.times,e.condition,up,down,params)
                    old_pred = decode_queue(old,baseline.decoders[e.condition])
                rows.append((e,old,new,old_pred))
            cache[block]=rows
        configurations={}
        records=[]
        for condition in ('queue_onset','queue_ongoing'):
            decode = decode_onset if condition=='queue_onset' else decode_queue
            trials=[]
            for weight in (.5,1.):
                for decoder in (.1,.2,.3,.4,.5,.6,'expected_iou'):
                    score=np.mean([proxy_queue_iou(e.truth,decode((1-weight)*old+weight*new,decoder),e.valid)
                        for e,old,new,_ in cache['selection'] if e.condition==condition])
                    trials.append((score,weight,decoder))
            best=max(trials,key=lambda x:x[0]); configurations[condition]={'weight':best[1],'decoder':best[2]}
            for block, rows in cache.items():
                for e,old,new,old_pred in rows:
                    if e.condition != condition: continue
                    pred=decode((1-best[1])*old+best[1]*new,best[2])
                    records.append({'panel':panel,'block':block,'condition':condition,'origin':str(d.times[e.origin]),
                        'baseline':proxy_queue_iou(e.truth,old_pred,e.valid),
                        'candidate':proxy_queue_iou(e.truth,pred,e.valid)})
        frame=pd.DataFrame(records)
        summary=frame.groupby(['condition','block'])[['baseline','candidate']].mean()
        accepted={c:bool(((summary.loc[c].loc[['november','january'],'candidate']-
            summary.loc[c].loc[['november','january'],'baseline'])>.005).all()) for c in configurations}
        report={'panel':panel,'policy':policy,'configurations':configurations,'accepted':accepted,
            'scores':{f'{c}/{b}':r.to_dict() for (c,b),r in summary.iterrows()},
            'fit_end':str(d.times[fit][-1]),'selection':'October only',
            'assessment':'Previously inspected November and January development months; observed-speed proxies, not official labels',
            'ongoing_baseline':'Matched-prefix v4 EventQueueModel, October-selected decoder'}
        frame.to_csv(args.output/f'{panel}_windows.csv',index=False)
        (args.output/f'{panel}_decision.json').write_text(json.dumps(report,indent=2))
        with (args.output/f'{panel}_candidate.pkl').open('wb') as f:
            pickle.dump({'model':candidate,'baseline':baseline,'links':d.links,'up':up,'down':down,
                'params':params,'density_scale':density_scale,'selection':report},f)
        print(json.dumps(report),flush=True)
        del d,auxiliary,events,baseline,candidate,cache;gc.collect()


if __name__=='__main__':
    main()
