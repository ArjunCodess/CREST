"""Evaluate state/FD trade-offs without claiming access to hidden LWR fluxes."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_state,graph
from src.evaluation import state_metrics
from src.physics import project_fd
from src.fd_calibration import soft_fd,fd_score


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--panel',action='append',required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for panel in args.panel:
        d=load_state(args.release_root,panel,'train',True,end='2031-02-01T00:00:00Z')
        up,down,_,params=graph(args.release_root,panel,d.links)
        with Path(f'artifacts/state_v4/pilot/{panel}_candidate.pkl').open('rb') as f:
            model=pickle.load(f)
        rows=[]
        for block,date in [('selection','2030-10-15'),('november','2030-11-15'),('january','2031-01-15')]:
            start=pd.Timestamp(date+'T00:00:00Z')
            select=np.asarray((d.times>=start)&(d.times<start+pd.Timedelta(days=7)))
            truth=d.values[select];times=d.times[select]
            rng=np.random.default_rng(642)
            rates=np.repeat(rng.choice([.2,.3,.5],size=int(np.ceil(len(truth)/288))),288)[:len(truth)]
            mask=(rng.random(truth.shape[:2])<rates[:,None])&np.isfinite(truth).all(-1)
            masked=truth.copy();masked[mask]=np.nan
            with threadpool_limits(limits=4,user_api='blas'):
                raw=project_fd(model.reconstruct(masked,times,up,down,params),params)
            for weight in (0.,.05,.1,.2,.4):
                pred=soft_fd(raw,params,weight)
                metrics=state_metrics(truth,pred,params.lanes,mask)
                sfd=fd_score(pred,params,mask)
                rows.append({'panel':panel,'block':block,'weight':weight,**metrics,'S_FD':sfd,
                    'state_fd_utility':.35*metrics['S_state']+.05*sfd})
        frame=pd.DataFrame(rows)
        best=frame[frame.block.eq('selection')].sort_values('state_fd_utility').iloc[-1]
        baseline=frame[frame.weight.eq(0)].set_index('block')
        chosen=frame[frame.weight.eq(best.weight)].set_index('block')
        gain=chosen.state_fd_utility-baseline.state_fd_utility
        state_delta=chosen.S_state-baseline.S_state
        accepted=bool(best.weight>0 and (gain.loc[['november','january']]>.0001).all()
            and (state_delta.loc[['november','january']]>=-.002).all())
        report={'panel':panel,'selected_weight':float(best.weight),'accepted':accepted,
            'utility_gain':gain.to_dict(),'state_delta':state_delta.to_dict(),
            'limitation':'State plus FD surrogate only. Organizer-held LWR fluxes and latent targets are unavailable.'}
        frame.to_csv(args.output/f'{panel}_metrics.csv',index=False)
        (args.output/f'{panel}_decision.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        del d,model;gc.collect()


if __name__=='__main__':main()
