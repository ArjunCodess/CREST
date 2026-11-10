"""Sparse, nonnegative prior-regularized ODME using measured links only."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, eye, vstack
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from .data import read_csv

def solve_od(A,counts,prior,regularization=.05):
    if regularization<=0: raise ValueError('Positive prior regularization required')
    valid=np.isfinite(counts)
    if not valid.any(): raise ValueError('No observed link counts')
    measured=A[valid]
    scale=max(float(np.mean(np.abs(counts[valid]))),1.)
    target=counts[valid]/scale; base=prior/scale
    # The dual has one variable per measured link, rather than one per path.
    # Its gradient is continuous and its generalized Hessian is positive definite.
    def dual(u):
        z=base-np.asarray(measured.T@u)/regularization
        flows=np.maximum(z,0)
        value=.5*(u@u)+target@u+.5*regularization*(flows@flows)
        gradient=u+target-np.asarray(measured@flows)
        return value,gradient,z
    u=np.zeros(len(target)); converged=False
    with threadpool_limits(limits=4):
        for iteration in range(100):
            value,gradient,z=dual(u)
            if np.max(np.abs(gradient))<1e-8:
                converged=True; break
            active=measured[:,z>0]
            hessian=np.eye(len(target))+(active@active.T).toarray()/regularization
            step=-cho_solve(cho_factor(hessian,lower=True,check_finite=False),gradient,check_finite=False)
            slope=float(gradient@step); fraction=1.
            for backtrack in range(40):
                candidate=u+fraction*step
                if dual(candidate)[0]<=value+1e-4*fraction*slope: break
                fraction*=.5
            u=candidate
    if not converged:
        _,gradient,z=dual(u)
        if np.max(np.abs(gradient))>=1e-6: raise RuntimeError('OD dual residual did not converge')
    flows=np.maximum(base-np.asarray(measured.T@u)/regularization,0)*scale
    loaded=measured@flows
    deviation=flows-prior
    cost=.5*(np.sum((loaded-counts[valid])**2)+regularization*np.sum(deviation**2))
    return flows, {'S_link':float(max(0,1-np.abs(loaded-counts[valid]).sum()/max(counts[valid].sum(),1e-9))),
                   'count_rmse':float(np.sqrt(np.mean((loaded-counts[valid])**2))),
                   'n_measured_links':int(valid.sum()),'solver_cost':float(cost),
                   'solver_iterations':int(iteration+1)}

def build_od(root,panel,split,regularization=.05):
    net=Path(root)/'corridors'/panel/'network'; task=Path(root)/'task4'/panel/split
    paths=read_csv(net/'path_set.csv'); inc=read_csv(net/'path_link_incidence.csv')
    counts=read_csv(task/'synthetic_link_counts.csv'); prior=read_csv(task/'synthetic_weak_prior.csv')
    template=read_csv(task/'sample_submission_path_flow.csv')
    ids=paths.path_id.tolist(); links=inc.link_id.drop_duplicates().tolist()
    pi={p:i for i,p in enumerate(ids)}; li={p:i for i,p in enumerate(links)}
    pairs=inc[['link_id','path_id']].drop_duplicates()
    A=coo_matrix((np.ones(len(pairs)),(pairs.link_id.map(li),pairs.path_id.map(pi))),shape=(len(links),len(ids))).tocsr()
    output=[]; diagnostics=[]
    for period,t in template.groupby('departure_time',sort=False):
        p=prior[prior.departure_time.eq(period)] if 'departure_time' in prior else prior
        c=counts[counts.departure_time.eq(period)] if 'departure_time' in counts else counts
        pv=p.set_index('path_id').path_flow.reindex(ids)
        if pv.isna().any(): raise ValueError('Released prior does not cover every path')
        cv=c.set_index('link_id')['count'].reindex(links).to_numpy()
        flows,d=solve_od(A,cv,pv.to_numpy(),regularization)
        t=t.copy(); t['path_flow']=t.path_id.map(dict(zip(ids,flows)))
        if t.path_flow.isna().any(): raise ValueError('Unknown template path')
        output.append(t); diagnostics.append({'panel':panel,'split':split,'departure_time':period,**d})
    return pd.concat(output,ignore_index=True),diagnostics
