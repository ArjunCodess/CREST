"""Evaluate cross-sensor completion on matched daily masks and temporal blocks."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state,graph
from src.state_v4 import EnhancedReconstructor
from src.refinement import ResidualReconstructor
from src.physics import project_fd
from src.evaluation import state_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--panel',action='append',required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for panel in args.panel:
        if (args.output/f'{panel}_decision.json').exists():continue
        started=time.monotonic()
        d=load_state(args.release_root,panel,'train',True)
        up,down,_,params=graph(args.release_root,panel,d.links)
        fit=np.asarray(d.times<pd.Timestamp('2030-10-01T00:00:00Z')-pd.Timedelta(hours=12))
        print('State baseline fit',panel,flush=True)
        baseline=ResidualReconstructor().fit(d.values[fit],d.times[fit],up,down,params)
        print('State factor/residual fit',panel,flush=True)
        candidate=EnhancedReconstructor().fit(d.values[fit],d.times[fit],up,down,params)
        rows=[]; predictions={}
        for block,start in [('selection','2030-10-15'),('november','2030-11-15'),('january','2031-01-15')]:
            begin=pd.Timestamp(start+'T00:00:00Z'); select=np.asarray((d.times>=begin)&(d.times<begin+pd.Timedelta(days=7)))
            truth=d.values[select]; times=d.times[select]
            rng=np.random.default_rng(642);daily=rng.choice([.2,.3,.5],size=int(np.ceil(len(truth)/288)))
            rates=np.repeat(daily,288)[:len(truth)]
            mask=(rng.random(truth.shape[:2])<rates[:,None])&np.isfinite(truth).all(-1)
            masked=truth.copy();masked[mask]=np.nan
            old=baseline.reconstruct(masked,times,up,down,params)
            new=candidate.reconstruct(masked,times,up,down,params)
            factor=candidate.factor.reconstruct(masked)
            predictions[block]=(truth,old,new,factor,mask,rates)
        selection=predictions['selection']
        configs=[]
        for nw in (.5,1.):
            for fw in (0.,.1,.25):
                truth,old,new,factor,mask,_=selection
                pred=project_fd((1-fw)*((1-nw)*old+nw*new)+fw*factor,params)
                configs.append((state_metrics(truth,pred,params.lanes,mask)['S_state'],nw,fw))
        _,nw,fw=max(configs)
        for block,(truth,old,new,factor,mask,rates) in predictions.items():
            predictions_by_name={'baseline':project_fd(old,params),
                'candidate':project_fd((1-fw)*((1-nw)*old+nw*new)+fw*factor,params)}
            for name,pred in predictions_by_name.items():
                for regime,rate in [('R1',.2),('R2',.3),('R3',.5)]:
                    target=mask&(rates[:,None]==rate)
                    if target.any():rows.append({'panel':panel,'block':block,'model':name,'regime':regime,
                        **state_metrics(truth,pred,params.lanes,target)})
        frame=pd.DataFrame(rows);scores=frame.groupby(['block','model']).S_state.mean().unstack()
        accepted=bool(((scores.loc[['november','january'],'candidate']-scores.loc[['november','january'],'baseline'])>.001).all())
        decision={'panel':panel,'new_weight':nw,'factor_weight':fw,'scores':scores.to_dict('index'),
            'accepted':accepted,'runtime_seconds':time.monotonic()-started,
            'fit_end':str(d.times[fit][-1]),'selection':'October 15-21 only','assessment':'November and January 15-21; previously examined months'}
        frame.to_csv(args.output/f'{panel}_metrics.csv',index=False)
        (args.output/f'{panel}_decision.json').write_text(json.dumps(decision,indent=2))
        with (args.output/f'{panel}_candidate.pkl').open('wb') as f:pickle.dump(candidate,f)
        print(json.dumps(decision),flush=True)
        del d,baseline,candidate,predictions
        gc.collect()


if __name__=='__main__':main()
