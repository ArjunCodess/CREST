"""Soft calibration toward a triangular fundamental diagram."""
import numpy as np
from .physics import project_fd


def soft_fd(states, params, weight):
    x = project_fd(states, params)
    vf = params.free_speed_kmh.to_numpy()
    cap = params.capacity_vph.to_numpy()
    kj = params.k_jam.to_numpy()
    wave = cap / (kj-cap/vf)
    speed, flow = x[...,0], x[...,1]
    free = np.stack([np.broadcast_to(vf,speed.shape),flow],-1)
    congested_speed = np.maximum(1,flow/np.maximum(kj-flow/wave,1e-6))
    congested = np.stack([congested_speed,flow],-1)
    congested_flow = wave*kj*speed/(speed+wave)
    fixed_speed = np.stack([speed,congested_flow],-1)
    choices = np.stack([free,congested,fixed_speed])
    scale = np.stack([np.full(len(params),25.),600*params.lanes.to_numpy()],-1)
    error = np.sum(((choices-x[None])/scale[None,None])**2,axis=-1)
    take = np.argmin(error,axis=0)
    nearest = np.take_along_axis(choices,take[None,...,None],axis=0)[0]
    return project_fd((1-weight)*x+weight*nearest,params)


def fd_score(states, params, mask):
    speed, flow = states[...,0],states[...,1]
    vf,cap,kj = (params[c].to_numpy() for c in ('free_speed_kmh','capacity_vph','k_jam'))
    density = flow/np.maximum(speed,1)
    wave = cap/(kj-cap/vf)
    expected = np.where(density<=cap/vf,vf*density,wave*np.maximum(kj-density,0))
    lanes=params.lanes.to_numpy()
    return float(max(0,1-np.abs((flow-expected)/lanes)[mask].sum()/max((flow/lanes)[mask].sum(),1e-9)))
