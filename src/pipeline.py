"""Train, validate, infer both scored months, and construct the official upload."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .data import panels,load_state,graph,read_csv,split_masks,split_manifest,family
from .features import artificial_mask,features,SCALE,sampled_features
from .models import Hybrid,booster,neural_reconstruct
from .baselines import historical_mean
from .physics import project_fd
from .od import build_od
from .submission import merge,merge_files
from .evaluation import state_metrics
from .queue_model import onset_slowdown

def fit_panel(root,panel,out,epochs,rounds,start=None,end=None,full=False,max_rows=60000,cv_days=7,reuse_full=False):
    data=load_state(root,panel,'train',truth=True,start=start,end=end)
    up,down,lap,params=graph(root,panel,data.links)
    fit,valid,cutoff=split_masks(data.times)
    if cv_days>0: valid &= np.asarray(data.times>=data.times[-1]-pd.Timedelta(days=cv_days))
    if not fit.any() or not valid.any(): raise ValueError('Date range too short for purged temporal split')
    print(f'  {len(data.links)} links, {len(data.times):,} time steps; fitting early block',flush=True)
    model=Hybrid(rounds=rounds).fit(data.values[fit],data.times[fit],up,down,params,max_rows=max_rows,neural_epochs=epochs)
    records=[]; rng=np.random.default_rng(2026)
    baseline=historical_mean(data.values[fit],data.times[fit],data.times[valid])
    for regime,rate in [('R1',.2),('R2',.3),('R3',.5)]:
        print(f'  Validate {regime}',flush=True)
        truth=data.values[valid]; masked,target=artificial_mask(truth,rng,rate)
        nn_prediction=(neural_reconstruct(model.neural,masked,up+down) if model.neural is not None else np.zeros_like(masked))
        candidates=[('tree_only',1.)]+([('hybrid',.8)] if model.neural is not None else [])
        for name,weight in candidates:
            model.tree_weight=weight
            raw=model.reconstruct(masked,data.times[valid],up,down,params,nn_prediction)
            for projection,pred in [('raw',raw),('fd',project_fd(raw,params))]:
                records.append({'panel':panel,'regime':regime,'model':name,'projection':projection,
                    'validation_type':'purged_temporal_artificial_mask','cutoff':str(cutoff),
                    **state_metrics(truth,pred,params.lanes,target)})
        references=[('historical_mean',baseline)]+([('neural_only',nn_prediction)] if model.neural is not None else [])
        for name,pred in references:
            records.append({'panel':panel,'regime':regime,'model':name,'projection':'raw',
                'validation_type':'purged_temporal_artificial_mask','cutoff':str(cutoff),
                **state_metrics(truth,pred,params.lanes,target)})
    selection=pd.DataFrame(records)
    candidate=selection[(selection.projection=='fd') & selection.model.isin(['hybrid','tree_only'])].groupby('model').S_state.mean()
    chosen=str(candidate.idxmax()); weight=.8 if chosen=='hybrid' else 1.
    print(f'  Selected {chosen} by temporal validation: {candidate.to_dict()}',flush=True)
    selection.to_csv(out/f'{panel}_holdout_metrics.csv',index=False)
    if full:
        existing=None
        if reuse_full and (out/f'{panel}.pkl').exists():
            with (out/f'{panel}.pkl').open('rb') as f: candidate_payload=pickle.load(f)
            if (candidate_payload.get('full_train') and candidate_payload['links']==data.links
                and candidate_payload['model'].rounds==rounds
                and candidate_payload['fit_end']==str(data.times[-1])):
                existing=candidate_payload['model']
        if existing is not None:
            print('  Keep existing full-train weights; apply validated blend choice',flush=True); model=existing
        else:
            print('  Refit all released train months',flush=True)
            model=None
            import gc; gc.collect()
            model=Hybrid(rounds=rounds).fit(data.values,data.times,up,down,params,max_rows=max_rows,
                                          neural_epochs=epochs if chosen=='hybrid' else 0)
    model.tree_weight=weight
    payload={'model':model,'links':data.links,'up':up,'down':down,'params':params,
             'fit_end':str(data.times[-1] if full else data.times[fit][-1]),'full_train':full,
             'selected_model':chosen,'tree_weight':weight,'epochs':epochs,'rounds':rounds,'max_train_rows':max_rows}
    for record in records: record['selected']=(record['model']==chosen and record['projection']=='fd')
    with (out/f'{panel}.pkl').open('wb') as f: pickle.dump(payload,f)
    return records

def predict_state(root,panel,split,payload,conservation_weight=0):
    data=load_state(root,panel,split)
    if data.links!=payload['links']: raise ValueError('Observed link set changed; retrain with full graph alignment')
    reconstructor=payload.get('reconstructor',payload['model'])
    pred=reconstructor.reconstruct(data.values,data.times,payload['up'],payload['down'],payload['params'])
    pred=project_fd(pred,payload['params'])
    if conservation_weight>0:
        from .conservation import project_public_blocks
        pred,_=project_public_blocks(root,panel,split,data,pred,payload['params'],conservation_weight)
    template=read_csv(Path(root)/'task1'/panel/split/'sample_submission_state.csv')
    ti=data.times.get_indexer(pd.to_datetime(template.timestamp,utc=True))
    ni=pd.Index(data.links).get_indexer(template.link_id)
    if (ti<0).any() or (ni<0).any(): raise ValueError('Template target absent from observations')
    template['speed_kmh']=pred[ti,ni,0]; template['flow_vph']=pred[ti,ni,1]
    return template,pred,data

def predict_queue(root,panel,split,payload,observations=None):
    folder=Path(root)/'task2'/panel/split
    if not (folder/'window_index.csv').exists():
        if panel in ('D12_I405_N','D12_I405_S'): return pd.DataFrame(columns=['window_id','timestamp','link_id','queue_pred'])
        raise FileNotFoundError(folder/'window_index.csv')
    windows=read_csv(folder/'window_index.csv')
    raw=pd.read_parquet(folder/'window_history.parquet')
    raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True); raw['link_id']=raw.link_id.astype(str)
    raw['window_id']=raw.window_id.astype(str)
    template=read_csv(folder/'sample_submission_queue.csv')
    template=template[template.window_id.isin(windows.window_id)].copy()
    params=payload['params']; cut=params.free_speed_kmh.to_numpy()*.6
    fd=read_csv(Path(root)/'corridors'/panel/'network/fd_parameters.csv')
    if 'v_cut' in fd:
        released=fd.groupby('link_id').v_cut.mean().reindex(payload['links']).to_numpy()
        cut=np.where(np.isfinite(released),released,cut)
    outputs=[]
    for wid,h in raw.groupby('window_id',sort=False):
        if str(wid) not in set(windows.window_id): continue
        target=template[template.window_id.eq(str(wid))].copy()
        target_times=pd.to_datetime(target.timestamp,utc=True)
        origin=target_times.min()-pd.Timedelta(minutes=5)
        h=h.copy()
        h=h[(h.timestamp>=origin-pd.Timedelta(minutes=60)) & (h.timestamp<=origin)]
        if observations is not None:
            at_origin=observations[observations.timestamp.eq(origin)].copy()
            h=pd.concat([h[h.timestamp<origin],at_origin],ignore_index=True)
        if 'pct_observed' in h: h.loc[h.pct_observed<75,['speed_kmh','flow_vph']]=np.nan
        if 'is_score_eligible' in h: h.loc[~h.is_score_eligible.astype(bool),['speed_kmh','flow_vph']]=np.nan
        # Only this window's released history is visible. No month reconstruction,
        # interpolation, future measurement, or other window can feed this call.
        times=pd.date_range(origin-pd.Timedelta(minutes=60),origin,freq='5min')
        v=np.stack([h.pivot_table(index='timestamp',columns='link_id',values=c).reindex(index=times,columns=payload['links']).to_numpy()
                    for c in ('speed_kmh','flow_vph')],-1)
        condition=str(windows.loc[windows.window_id.eq(str(wid)),'condition'].iloc[0])
        if 'queue_classifier' in payload:
            classifier=payload['queue_classifier']
            predicted_queue=classifier.classify(classifier.scores(v,times,payload['up'],payload['down'],params),condition)
        else:
            future=payload['model'].forecast(v,times,payload['up'],payload['down'],params)
            future=project_fd(future,params)
            predicted_queue=future[...,0]<=cut[None]
            # Onset history is free-flow, so this forecast stays above v_cut and the
            # mask is empty. Fill only that empty case; ongoing keeps the speed mask.
            if condition=='queue_onset' and not predicted_queue.any():
                predicted_queue=onset_slowdown(future[...,0],v[...,0])
        ordered=sorted(pd.to_datetime(target.timestamp,utc=True).unique())
        if len(ordered)!=6: raise ValueError(f'Window {wid} does not have six future steps')
        if list(pd.DatetimeIndex(ordered))!=list(pd.date_range(origin+pd.Timedelta(minutes=5),periods=6,freq='5min')):
            raise ValueError('Queue timestamps must be T+5 through T+30')
        ti=pd.Index(ordered).get_indexer(pd.to_datetime(target.timestamp,utc=True))
        ni=pd.Index(payload['links']).get_indexer(target.link_id)
        if (ni<0).any(): raise ValueError('Queue template contains unknown link')
        target['queue_pred']=predicted_queue[ti,ni].astype(int); outputs.append(target)
    if not outputs: raise ValueError(f'No queue forecasts for {panel}/{split}')
    return pd.concat(outputs,ignore_index=True)

def spatial_cv(root,out,rounds,start=None,end=None,max_rows=30000):
    """Family holdout with a shared tree; feature building never pools families."""
    cache={}; ps=panels(root); rng=np.random.default_rng(2026)
    for panel in ps:
        print('Spatial features',panel,flush=True)
        d=load_state(root,panel,'train',truth=True,start=start,end=end)
        up,down,_,params=graph(root,panel,d.links); fit,test,_=split_masks(d.times)
        cache[panel]=[]
        for rows in (fit,test):
            values=d.values[rows]; masked,target=artificial_mask(values,rng,.3)
            x,ix=sampled_features(masked,d.times[rows],up,down,params,target,rng,max_rows,True)
            lane=np.broadcast_to(params.lanes.to_numpy(),values.shape[:2]).ravel()[ix]
            cache[panel].append((x,values.reshape(-1,2)[ix],lane))
    results=[]
    for held in sorted({family(p) for p in ps}):
        print('Spatial holdout',held,flush=True)
        training=[cache[p][0] for p in ps if family(p)!=held]
        if not training: continue
        xx=np.concatenate([v[0] for v in training]); yy=np.concatenate([v[1] for v in training])
        models=[booster(rounds=rounds).fit(xx,yy[:,c]/SCALE[c]) for c in range(2)]
        for panel in [p for p in ps if family(p)==held]:
            x,y,lanes=cache[panel][1]; pred=np.stack([m.booster_.predict(x,num_threads=4)*SCALE[c] for c,m in enumerate(models)],-1)
            rmse=np.sqrt(np.mean((y[:,0]-pred[:,0])**2)); fq=np.sqrt(np.mean(((y[:,1]-pred[:,1])/lanes)**2))
            results.append({'panel':panel,'held_out_family':held,'model':'shared_tree_only',
                            'rmse_speed_kmh':float(rmse),'rmse_flow_vph_per_lane':float(fq),
                            'S_state':float(.54*max(0,1-rmse/25)+.46*max(0,1-fq/600))})
    pd.DataFrame(results).to_csv(out/'spatial_cv.csv',index=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['inspect','train','predict','spatial-cv'])
    ap.add_argument('--release-root',type=Path,default=Path('data/kaggle_public'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/run'))
    ap.add_argument('--model-root',type=Path,help='Read checkpoints from a separate training directory during prediction')
    ap.add_argument('--save-panel-states',action='store_true',help='Also write redundant per-panel state CSVs')
    ap.add_argument('--queue-history-only',action='store_true',help='Reproduce the selected queue export using window_history without adding the monthly origin reading')
    ap.add_argument('--panel',action='append'); ap.add_argument('--split',choices=['train','validation','private','both'],default='both')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--rounds',type=int,default=160)
    ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--full-train',action='store_true')
    ap.add_argument('--max-train-rows',type=int,default=60000)
    ap.add_argument('--cv-days',type=int,default=7,help='Evaluate the last N days of the purged held-out block; 0 uses the entire block')
    ap.add_argument('--resume',action='store_true',help='Keep completed panel models and their recorded validation rows')
    ap.add_argument('--force',action='store_true',help='Refit selected panels while preserving other panels\' recorded validation')
    ap.add_argument('--conservation-weight',type=float,default=0,
                    help='Optional topology-derived soft penalty; not validated against hidden boundary flows')
    args=ap.parse_args(); root=args.release_root.resolve(); ps=panels(root)
    if args.panel:
        if set(args.panel)-set(ps): raise ValueError('Unknown panel selection')
        ps=args.panel
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    split_manifest(root,out/'cv_manifest.json')
    if args.command=='inspect':
        summary={p:{'network_files':[x.name for x in (root/'corridors'/p/'network').iterdir()],
                     'train_parquet_files':len(list((root/'corridors'/p/'train').rglob('*.parquet')))} for p in ps}
        (out/'inventory.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); return
    if args.command=='spatial-cv': spatial_cv(root,out,args.rounds,args.start,args.end); return
    if args.command=='train':
        previous=out/'temporal_cv.csv'
        metrics=pd.read_csv(previous).to_dict('records') if args.resume and previous.exists() else []
        for p in ps:
            if args.resume and not args.force and (out/f'{p}.pkl').exists() and any(r['panel']==p and pd.notna(r.get('model')) for r in metrics):
                print(f'Keep completed model {p}',flush=True); continue
            print(f'Fit {p}',flush=True)
            new_records=fit_panel(root,p,out,args.epochs,args.rounds,args.start,args.end,args.full_train,args.max_train_rows,args.cv_days,args.resume)
            metrics=[r for r in metrics if r['panel']!=p]+new_records
            pd.DataFrame(metrics).to_csv(out/'temporal_cv.csv',index=False)
        return
    splits=['validation','private'] if args.split=='both' else [args.split]
    paths={k:out/f'{k}.csv' for k in ['state','queue','odme']}
    written={k:False for k in paths}; diagnostics=[]; choices=[]
    model_root=args.model_root.resolve() if args.model_root else out
    for p in ps:
        with (model_root/f'{p}.pkl').open('rb') as f: payload=pickle.load(f)
        if (model_root/f'{p}_refined.pkl').exists():
            with (model_root/f'{p}_refined.pkl').open('rb') as f: payload['reconstructor']=pickle.load(f)
        if (model_root/f'{p}_queue.pkl').exists():
            with (model_root/f'{p}_queue.pkl').open('rb') as f: payload['queue_classifier']=pickle.load(f)
        choices.append({'panel':p,'state_model':'residual' if 'reconstructor' in payload else payload['selected_model'],
                        'queue_model':'classifier' if 'queue_classifier' in payload else 'speed_regression',
                        'fit_end':payload['fit_end']})
        for split in splits:
            print(f'Predict {p}/{split}',flush=True)
            state,pred,data=predict_state(root,p,split,payload,args.conservation_weight)
            queue=predict_queue(root,p,split,payload,None if args.queue_history_only else data.rows)
            od,diag=build_od(root,p,split); diagnostics.extend(diag)
            for task,frame in [('state',state),('queue',queue),('odme',od)]:
                if not len(frame): continue
                frame.to_csv(paths[task],index=False,header=not written[task],mode='a' if written[task] else 'w')
                written[task]=True
            if args.save_panel_states: state.to_csv(out/f'{p}_{split}_state.csv',index=False)
            np.savez_compressed(out/f'{p}_{split}_states.npz',prediction=pred,
                               times=data.times.astype(str).to_numpy(dtype=str),links=np.asarray(data.links))
    if not all(written.values()): raise ValueError('All three task outputs required')
    (out/'od_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'prediction_models.json').write_text(json.dumps(choices,indent=2))
    if args.split=='both' and len(ps)==len(panels(root)):
        filename='fixture_submission.csv' if (root/'FIXTURE_ONLY.json').exists() else 'submission.csv'
        n=merge_files(root,paths,out/filename); print(f'Validated {filename}: {n:,} rows')
    else: print('Subset run: wrote task files only; full upload requires all panels and both months.')

if __name__=='__main__': main()
