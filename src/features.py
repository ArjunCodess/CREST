"""Causal forecast features; optional two-sided reconstruction features."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SCALE = np.array([120., 10000.], dtype=np.float32)

def shift(x, steps):
    y = np.full_like(x, np.nan, dtype=np.float32)
    if steps > 0: y[steps:] = x[:-steps]
    elif steps < 0: y[:steps] = x[-steps:]
    else: y[:] = x
    return y

def neighbors(x, operator):
    valid = np.isfinite(x)
    # Road graphs are sparse; avoid a dense T*N*N contraction for every block.
    op=csr_matrix(operator); t,n,c=x.shape
    numerator=(op@np.nan_to_num(x).transpose(1,0,2).reshape(n,-1)).reshape(n,t,c).transpose(1,0,2)
    denominator=(op@valid.astype(np.float32).transpose(1,0,2).reshape(n,-1)).reshape(n,t,c).transpose(1,0,2)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator>0)

def features(values, times, up, down, params, reconstruction=False, current_mask=None):
    x = values.astype(np.float32) / SCALE
    upstream, downstream = neighbors(x, up), neighbors(x, down)
    current=values if current_mask is None else np.where(current_mask[...,None],np.nan,values)
    current_x=x if current_mask is None else current/SCALE
    current_up=upstream if current_mask is None else neighbors(current_x,up)
    current_down=downstream if current_mask is None else neighbors(current_x,down)
    blocks = [current_x, np.isfinite(current_x).astype(np.float32), current_up, current_down]
    for lag in range(1, 13):
        blocks.extend([shift(x, lag), shift(upstream, lag), shift(downstream, lag)])
    if reconstruction:
        # Future released measurements are legal for Task 1, never for Task 2.
        for lag in (1, 2, 3, 6, 12): blocks.append(shift(x, -lag))
    speed_gradient = current_down[..., 0] - current_up[..., 0]
    k = np.divide(current[..., 1], current[..., 0], out=np.full(values.shape[:2], np.nan,dtype=np.float32), where=current[..., 0]>1)
    kd, ku = neighbors(k[..., None], down)[..., 0], neighbors(k[..., None], up)[..., 0]
    dq = (current_down[..., 1]-current_up[..., 1])*SCALE[1]
    wave = np.divide(dq, kd-ku, out=np.full_like(dq, np.nan), where=np.abs(kd-ku)>1)
    blocks.extend([speed_gradient[..., None], np.clip(wave/30, -5, 5)[..., None],
                   (current_x[..., 0]-shift(x, 1)[..., 0])[..., None]])
    t = pd.DatetimeIndex(times)
    phase = np.asarray((t.hour*60+t.minute)/1440*2*np.pi)
    calendar = np.stack([np.sin(phase), np.cos(phase), t.dayofweek/6], -1)
    blocks.append(np.broadcast_to(calendar[:, None], (*x.shape[:2], 3)))
    static = params[['length_km','lanes','free_speed_kmh','capacity_vph','k_jam']].to_numpy()/np.array([10,6,120,10000,600])
    blocks.append(np.broadcast_to(static[None], (*x.shape[:2],5)))
    return np.concatenate([np.asarray(block,dtype=np.float32) for block in blocks],-1)

def artificial_mask(values, rng, rate=.3):
    """Mask all channels together at a link-time cell, never one target only."""
    mask = (rng.random(values.shape[:2]) < rate) & np.isfinite(values).all(-1)
    x = values.copy(); x[mask] = np.nan
    return x, mask

def sampled_features(values,times,up,down,params,eligible,rng,max_rows=200000,reconstruction=False,current_mask=None):
    """Compute only sampled rows, in daily blocks, to bound feature RAM."""
    n=values.shape[1]; available=np.flatnonzero(eligible.ravel())
    ix=rng.choice(available,min(max_rows,len(available)),replace=False)
    if not len(ix): raise ValueError('No eligible training feature rows')
    blocks=[]; indices=[]
    for start in range(0,len(values),288):
        selected=ix[(ix//n>=start)&(ix//n<start+288)]
        if not len(selected): continue
        lo=max(0,start-12); hi=min(len(values),start+288+(12 if reconstruction else 0))
        x=features(values[lo:hi],times[lo:hi],up,down,params,reconstruction,
                   None if current_mask is None else current_mask[lo:hi])
        blocks.append(x[(selected//n)-lo,selected%n]); indices.append(selected)
    return np.concatenate(blocks),np.concatenate(indices)

def prediction_blocks(values,times,up,down,params,reconstruction=False):
    for start in range(0,len(values),288):
        end=min(start+288,len(values)); lo=max(0,start-12)
        hi=min(len(values),end+(12 if reconstruction else 0))
        x=features(values[lo:hi],times[lo:hi],up,down,params,reconstruction)[start-lo:end-lo]
        yield start,end,x
