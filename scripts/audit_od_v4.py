"""Compare additive and entropy OD priors on held-out counts and synthetic corruptions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from src.data import read_csv
from src.od import solve_od


def entropy_od(A,counts,prior,regularization=.01):
    valid=np.isfinite(counts);operator=A[valid]
    scale=max(float(np.mean(counts[valid])),1.)
    base=np.maximum(prior/scale,1e-10);target=counts[valid]/scale
    def objective(u):
        exponent=np.asarray(operator.T@u)
        flows=base*np.exp(np.clip(exponent,-40,40))
        return float(flows.sum()-target@u+regularization*(u@u)/2),np.asarray(operator@flows)-target+regularization*u
    solved=minimize(objective,np.zeros(valid.sum()),jac=True,method='L-BFGS-B',
        options={'maxiter':800,'gtol':1e-8,'ftol':1e-12})
    return base*np.exp(np.clip(operator.T@solved.x,-40,40))*scale


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(714);records=[];rank_rows=[]
    for folder in sorted((args.release_root/'task4').iterdir()):
        panel=folder.name;net=args.release_root/'corridors'/panel/'network'
        inc=read_csv(net/'path_link_incidence.csv')[['link_id','path_id']].drop_duplicates()
        paths=read_csv(net/'path_set.csv').path_id.tolist();links=inc.link_id.drop_duplicates().tolist()
        pi={p:i for i,p in enumerate(paths)};li={p:i for i,p in enumerate(links)}
        A=coo_matrix((np.ones(len(inc)),(inc.link_id.map(li),inc.path_id.map(pi))),shape=(len(links),len(paths))).tocsr()
        prior=read_csv(folder/'train/synthetic_weak_prior.csv');counts=read_csv(folder/'train/synthetic_link_counts.csv')
        period=prior.departure_time.iloc[0]
        b=prior[prior.departure_time.eq(period)].set_index('path_id').path_flow.reindex(paths).to_numpy()
        count_period=counts[counts.departure_time.eq(period)] if 'departure_time' in counts else counts
        c=count_period.set_index('link_id')['count'].reindex(links).to_numpy()
        valid=np.flatnonzero(np.isfinite(c));measured=A[valid].toarray()
        rank=int(np.linalg.matrix_rank(measured));rank_rows.append({'panel':panel,'paths':len(paths),'measured_links':len(valid),'rank':rank,'minimum_nullity':len(paths)-rank})
        for fold in range(4):
            held=rng.choice(valid,max(1,len(valid)//5),replace=False);training=c.copy();training[held]=np.nan
            for name,flows in [('additive',solve_od(A,training,b)[0]),('entropy',entropy_od(A,training,b))]:
                records.append({'panel':panel,'experiment':'heldout_link','fold':fold,'method':name,
                    'error':float(np.abs(A[held]@flows-c[held]).sum()/np.maximum(c[held].sum(),1.)),
                    'prior_deviation':float(np.abs(flows-b).sum()/max(b.sum(),1.))})
        # This supplies known synthetic path truth, not hidden organizer labels.
        for fold in range(6):
            truth=np.maximum(b,1e-8)
            noisy_prior=truth*np.exp(rng.normal(-.5*.35**2,.35,len(b)))
            synthetic_counts=np.full(len(links),np.nan);synthetic_counts[valid]=A[valid]@truth
            for name,flows in [('additive',solve_od(A,synthetic_counts,noisy_prior)[0]),('entropy',entropy_od(A,synthetic_counts,noisy_prior))]:
                records.append({'panel':panel,'experiment':'synthetic_lognormal_prior','fold':fold,'method':name,
                    'error':float(np.abs(flows-truth).sum()/max(truth.sum(),1.)),
                    'prior_deviation':float(np.abs(flows-noisy_prior).sum()/max(noisy_prior.sum(),1.))})
        pd.DataFrame(records).to_csv(args.output/'comparisons.csv',index=False)
        (args.output/'rank_audit.json').write_text(json.dumps(rank_rows,indent=2))
        print(panel,'rank',rank,'of',len(paths),'paths',flush=True)
    frame=pd.DataFrame(records)
    summary=frame.groupby(['experiment','method']).error.mean().unstack()
    report={'scores':summary.to_dict('index'),'interpretation':'Held-out link counts and synthetic path priors are surrogate evidence, not official OD truth.',
        'promotion':'Keep additive baseline unless entropy improves both error measures by at least 10%.'}
    report['entropy_gate_passed']=bool((summary.entropy<.9*summary.additive).all())
    (args.output/'decision.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
