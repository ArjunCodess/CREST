"""FD feasibility and honest, optional conservation regularization.

Density is derived, never a separately submitted degree of freedom. Conservation
can conflict with noisy observations and unknown boundaries; residuals are
reported rather than claimed to vanish against hidden organizer fluxes.
"""
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def project_fd(states, params):
    x=np.asarray(states,dtype=float).copy()
    vf=params.free_speed_kmh.to_numpy(); cap=params.capacity_vph.to_numpy()
    kj=params.k_jam.to_numpy(); kc=cap/vf; w=cap/np.maximum(kj-kc,1e-6)
    v=np.clip(x[...,0],1,vf)
    bound=np.minimum(cap,w*kj*v/(v+w))
    q=np.clip(x[...,1],0,bound)
    return np.stack([v,q],-1)

def differentiable_fd(states, vf, cap, kj):
    """Piecewise differentiable fixed-speed triangular-FD projection."""
    v=torch.minimum(states[...,0].clamp_min(1),vf)
    w=cap/(kj-cap/vf).clamp_min(1e-6)
    q=torch.minimum(states[...,1].clamp_min(0),torch.minimum(cap,w*kj*v/(v+w)))
    return torch.stack([v,q],-1)

def conservation_residual(states, length, inflow, outflow, ramp_net):
    density=states[...,1]/np.maximum(states[...,0],1)
    return np.diff(density*length,axis=0)-(inflow[:-1]+ramp_net[:-1]-outflow[:-1])/12

def smooth_conservation(states, observed, params, incoming, outgoing, ramp_net,
                        valid, weight=.05, max_nfev=20):
    """Optimize missing cells only, with known ramp data and explicit flux operators.

    incoming/outgoing are raw sum operators, not normalized graph averages.
    valid excludes boundaries, latent neighbors, and unavailable ramps. This
    topology-based residual is a diagnostic, not the official Task 3 score.
    """
    initial=project_fd(states,params)
    missing=~np.isfinite(observed)
    if not missing.any() or not np.any(valid): return initial, {'optimized':False}
    shape=initial.shape; loc=np.flatnonzero(missing.ravel())
    base=initial.copy(); base[np.isfinite(observed)]=observed[np.isfinite(observed)]
    scale=np.broadcast_to(np.array([120.,10000.]),shape).ravel()[loc]
    start=base.ravel()[loc]/scale
    length=params.length_km.to_numpy()
    def unpack(z):
        x=base.copy(); x.ravel()[loc]=z*scale
        feasible=project_fd(x,params)
        return np.where(missing,feasible,observed)
    def mass(x):
        q=x[...,1]
        return conservation_residual(x,length,q@incoming.T,q@outgoing.T,ramp_net)[valid[:-1]]
    def residual(z):
        return np.r_[z-start,np.sqrt(weight)*mass(unpack(z))/10]
    # Sparse dependency pattern keeps numerical derivatives practical on daily blocks.
    nt,nn,_=shape
    cells=np.argwhere(valid[:-1]); pattern=lil_matrix((len(loc)+len(cells),len(loc)),dtype=int)
    lookup={int(flat):j for j,flat in enumerate(loc)}
    pattern[:len(loc),:]=__import__('scipy').sparse.eye(len(loc),format='lil')
    for ri,(t,i) in enumerate(cells):
        neighbors=np.flatnonzero((incoming[i]+outgoing[i])!=0)
        for tt,jj,channels in [(t,i,(0,1)),(t+1,i,(0,1))]+[(t,int(j),(0,1)) for j in neighbors]:
            for c in channels:
                col=lookup.get((tt*nn+jj)*2+c)
                if col is not None: pattern[len(loc)+ri,col]=1
    solved=least_squares(residual,start,bounds=(0,np.inf),jac_sparsity=pattern.tocsr(),max_nfev=max_nfev)
    result=unpack(solved.x)
    return result, {'optimized':True,'success':bool(solved.success),'nfev':solved.nfev,
                    'before_mass_rmse_vehicles':float(np.sqrt(np.mean(mass(base)**2))),
                    'after_mass_rmse_vehicles':float(np.sqrt(np.mean(mass(result)**2)))}
