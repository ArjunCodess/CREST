"""Residual reconstruction around observed temporal interpolation.

All interpolation operates on the current masked segment, so no held-out target
or future queue horizon can enter the reconstruction features.
"""
import gc
import numpy as np
import pandas as pd
from .features import features,SCALE,prediction_blocks
from .models import booster

def interpolation(values):
    base=np.empty_like(values)
    for c in range(2):
        x=pd.DataFrame(values[...,c])
        base[...,c]=x.interpolate(limit_direction='both').to_numpy()
    return base

def _feature_rows(values,times,up,down,params,indices=None,row_slice=None):
    """Materialize reconstruction features without a full-window tensor."""
    if indices is not None:
        n=values.shape[1]; ix=np.asarray(indices,dtype=np.int64); blocks=[]
        for block_start in range(0,len(values),288):
            block_end=min(block_start+288,len(values))
            sel=ix[(ix//n>=block_start)&(ix//n<block_end)]
            if not len(sel): continue
            lo=max(0,block_start-12); hi=min(len(values),block_end+12)
            x=features(values[lo:hi],times[lo:hi],up,down,params,True)
            t,nk=sel//n,sel%n
            blocks.append(x[t-lo,nk])
        return np.ascontiguousarray(np.concatenate(blocks))
    if row_slice is not None:
        rs,re=row_slice; parts=[]
        for start,end,x in prediction_blocks(values,times,up,down,params,True):
            a,b=max(start,rs),min(end,re)
            if a<b: parts.append(x[a-start:b-start])
        return np.concatenate(parts) if len(parts)>1 else parts[0]
    parts=[x for _,_,x in prediction_blocks(values,times,up,down,params,True)]
    return parts[0] if len(parts)==1 else np.concatenate(parts)

def reconstruction_rows(values,times,up,down,params,indices=None,row_slice=None):
    base=interpolation(values)
    fallback=np.stack([params.free_speed_kmh.to_numpy(),params.capacity_vph.to_numpy()*.65],-1).astype(np.float32)
    base=np.where(np.isfinite(base),base,fallback[None])
    x=_feature_rows(values,times,up,down,params,indices,row_slice)
    if indices is not None:
        selected_base=np.ascontiguousarray(base.reshape(-1,2)[indices])
        return np.concatenate([x,selected_base/SCALE],-1),selected_base
    if row_slice is not None:
        rs,re=row_slice
        base_slice=base[rs:re]
        return np.concatenate([x,base_slice/SCALE],-1),base_slice
    return np.concatenate([x,base/SCALE],-1),base

class ResidualReconstructor:
    def __init__(self,rounds=220,max_rows=120000):
        self.rounds=rounds; self.max_rows=max_rows
    def fit(self,values,times,up,down,params):
        rng=np.random.default_rng(47); n=values.shape[1]
        # Mix all official missingness rates rather than training only at 30%.
        masked=values.copy(); regime=rng.choice([.2,.3,.5],size=(len(values),1))
        hidden=(rng.random(values.shape[:2])<regime)&np.isfinite(values).all(-1)
        masked[hidden]=np.nan
        available=np.flatnonzero(hidden.ravel()); chosen=rng.choice(available,min(self.max_rows,len(available)),replace=False)
        xx=[]; yy=[]
        for start in range(0,len(values),288):
            selected=chosen[(chosen//n>=start)&(chosen//n<start+288)]
            if not len(selected): continue
            lo=max(0,start-144); hi=min(len(values),start+432)
            local=(selected//n-lo)*n+selected%n
            x,b=reconstruction_rows(masked[lo:hi],times[lo:hi],up,down,params,local)
            xx.append(x); yy.append((values.reshape(-1,2)[selected]-b)/SCALE)
            del x,b
            if len(xx)%14==0: gc.collect()
        x=np.concatenate(xx); y=np.concatenate(yy); del xx,yy; gc.collect()
        self.models=[booster(seed=47+c,rounds=self.rounds).fit(x,y[:,c]) for c in range(2)]
        return self
    def reconstruct(self,values,times,up,down,params):
        out=np.empty_like(values)
        for start in range(0,len(values),288):
            end=min(start+288,len(values)); lo=max(0,start-144); hi=min(len(values),end+144)
            x,b=reconstruction_rows(values[lo:hi],times[lo:hi],up,down,params,row_slice=(start-lo,end-lo))
            pred=np.stack([m.booster_.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(x.shape[:2])*SCALE[c]
                           for c,m in enumerate(self.models)],-1)+b
            out[start:end]=np.where(np.isfinite(values[start:end]),values[start:end],np.maximum(pred,0))
        return out
