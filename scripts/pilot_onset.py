"""Fit onset localization and complementary priors on train-only chronological folds."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state, graph
from src.queue_events import load_auxiliary, calibrate_occupancy_density, visible_event, decode_queue, EventQueueModel
from src.queue_onset import OnsetLocalizer, onset_events, decode_onset
from src.evaluation import proxy_queue_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True)
    ap.add_argument('--rounds', type=int, default=240)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for panel in args.panel:
        if (args.output/f'{panel}_baseline.pkl').exists():
            print('Keep completed', panel, flush=True); continue
        started = time.monotonic()
        d = load_state(args.release_root, panel, 'train', True, end='2031-02-01T00:00:00Z')
        aux = load_auxiliary(args.release_root, panel, d.times, d.links)
        up, down, _, params = graph(args.release_root, panel, d.links)
        fit = np.asarray(d.times < pd.Timestamp('2030-09-30T22:30:00Z'))
        density_scale = calibrate_occupancy_density(aux[fit]); aux[..., 1] = aux[..., 0]*density_scale
        cutoff = params.free_speed_kmh.to_numpy()*.6
        groups = {'fit':fit}
        for month in ('2030-10','2030-11','2031-01'):
            start = pd.Timestamp(month+'-01T00:00:00Z')
            groups[month] = np.asarray((d.times >= start)&(d.times < start+pd.offsets.MonthBegin(1)))
        events = {k:onset_events(d.values,aux,d.times,mask,cutoff) for k,mask in groups.items()}
        print(panel, 'onset episodes', {k:len(v) for k,v in events.items()}, flush=True)
        if min(map(len, events.values())) < 6:
            raise ValueError('Too few events for temporal selection and assessment')
        candidates, selected = {}, []
        for policy in ('history_only','origin_observed'):
            model = OnsetLocalizer(policy=policy,rounds=args.rounds).fit(events['fit'],up,down,params)
            cache = []
            for w in events['2030-10']:
                h,a = visible_event(w,policy)
                cache.append((w,model.components(h,a,w.times,up,down,params)))
            for tw in (0.,.5,1.):
                for pw in (0.,.25,.5):
                    for decoder in (.05,.1,.2,.3,.4,.5,'expected_iou'):
                        scores = []
                        for w,(tree,analogue,prior) in cache:
                            p = (1-pw)*(tw*tree+(1-tw)*analogue)+pw*prior
                            pred = np.zeros_like(w.truth); pred[-1] = decode_queue(p[None],decoder)[0]
                            scores.append(proxy_queue_iou(w.truth,pred,w.valid))
                        selected.append({'policy':policy,'tree_weight':tw,'prior_weight':pw,
                            'decoder':decoder,'selection_iou':float(np.mean(scores))})
            candidates[policy] = model
        best = max(selected,key=lambda r:r['selection_iou'])
        model = candidates[best['policy']]
        model.tree_weight,model.prior_weight,model.decoder = best['tree_weight'],best['prior_weight'],best['decoder']
        # Baseline uses the same fitting prefix, without the new localization features.
        baseline = EventQueueModel(policy=model.policy,rounds=180)
        # Only the onset head is required; avoid fitting an unrelated ongoing head.
        training = events['fit']
        xx,yy,ww = [],[],[]
        from src.queue_events import event_features
        from lightgbm import LGBMClassifier
        for w in training:
            h,a = visible_event(w,model.policy); valid=w.valid[-1]
            xx.append(event_features(h,a,w.times,up,down,params)[-len(params):][valid])
            yy.append(w.truth[-1,valid]);ww.extend([1/valid.sum()]*valid.sum())
        weights=np.asarray(ww);weights*=len(weights)/weights.sum()
        old=LGBMClassifier(n_estimators=180,num_leaves=15,min_child_samples=60,reg_lambda=6,
            learning_rate=.04,n_jobs=4,verbosity=-1,random_state=318).fit(np.concatenate(xx),np.concatenate(yy),sample_weight=weights)
        baseline.models={'queue_onset':old}
        policies=(.1,.2,.3,.4,.5,.6,'expected_iou')
        selection_cache=[]
        for w in events['2030-10']:
            h,a=visible_event(w,model.policy)
            selection_cache.append((w,baseline.scores(h,a,w.times,'queue_onset',up,down,params)))
        baseline_decoder=max(policies,key=lambda decoder:np.mean([
            proxy_queue_iou(w.truth,decode_onset(p,decoder),w.valid) for w,p in selection_cache]))
        baseline.decoders['queue_onset']=baseline_decoder
        assessment = []
        for block in ('2030-11','2031-01'):
            for w in events[block]:
                h,a=visible_event(w,model.policy)
                oldp=baseline.scores(h,a,w.times,'queue_onset',up,down,params)
                oldpred=decode_onset(oldp,baseline_decoder)
                pred=model.predict(w,up,down,params)
                assessment.append({'panel':panel,'block':block,'origin':str(d.times[w.origin]),
                    'baseline':proxy_queue_iou(w.truth,oldpred,w.valid),
                    'candidate':proxy_queue_iou(w.truth,pred,w.valid),'true_cells':int(w.truth.sum()),
                    'predicted_cells':int(pred.sum())})
        frame=pd.DataFrame(assessment); scores=frame.groupby('block')[['baseline','candidate']].mean()
        accepted=bool(((scores.candidate-scores.baseline)>.005).all())
        decision={'panel':panel,'selected':best,'baseline_decoder':baseline_decoder,'assessment':scores.to_dict('index'),
            'accepted':accepted,'training_episodes':len(training),'runtime_seconds':time.monotonic()-started,
            'assessment_note':'November and January are previously inspected development blocks; this refit excludes both.'}
        (args.output/f'{panel}_decision.json').write_text(json.dumps(decision,indent=2))
        frame.to_csv(args.output/f'{panel}_windows.csv',index=False)
        with (args.output/f'{panel}_onset.pkl').open('wb') as f:
            pickle.dump({'model':model,'links':d.links,'up':up,'down':down,'params':params,
                'density_scale':density_scale,'selected':best,'fit_end':str(d.times[fit][-1])},f)
        with (args.output/f'{panel}_baseline.pkl').open('wb') as f:
            pickle.dump({'model':baseline,'links':d.links,'up':up,'down':down,'params':params,
                'density_scale':density_scale,'fit_end':str(d.times[fit][-1])},f)
        records.append(decision)
        print(json.dumps(decision),flush=True)
        del d,aux,model,candidates,baseline,xx,yy,events
        gc.collect()
    print('Onset pilot completed',flush=True)


if __name__=='__main__':
    main()
