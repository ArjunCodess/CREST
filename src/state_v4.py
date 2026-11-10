"""Low-rank cross-sensor completion and two-sided residual reconstruction."""
import gc
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.utils.extmath import randomized_svd
from .features import SCALE, neighbors, shift
from .models import booster
from .refinement import interpolation, reconstruction_rows


class CrossSensorFactor:
    def __init__(self, rank=10, ridge=.1):
        self.rank,self.ridge=rank,ridge

    def fit(self, values, params):
        self.scale=np.stack([params.free_speed_kmh.to_numpy(),params.capacity_vph.to_numpy()],-1)
        rng=np.random.default_rng(641)
        # Temporal filling supplies training covariance, never evaluation targets.
        filled=interpolation(values)/self.scale[None]
        sampled=filled[rng.choice(len(filled),min(len(filled),6000),replace=False)].reshape(-1,len(params)*2)
        self.mean=np.nanmean(sampled,0)
        self.mean=np.nan_to_num(self.mean,nan=0.)
        self.std=np.maximum(np.nanstd(sampled,0),.04)
        self.std=np.nan_to_num(self.std,nan=.04)
        standardized=np.nan_to_num((sampled-self.mean)/self.std)
        _,_,vt=randomized_svd(standardized,n_components=min(self.rank,len(params)*2-1),random_state=641)
        self.loading=vt.T.astype(np.float32)
        return self

    def reconstruct(self, values):
        n=values.shape[1]
        x=(values/self.scale[None]).reshape(len(values),-1)
        ok=np.isfinite(x); centered=np.nan_to_num((x-self.mean)/self.std)
        output=np.empty_like(x)
        load=self.loading
        for start in range(0,len(x),384):
            end=min(start+384,len(x))
            gram=np.einsum('ti,ij,ik->tjk',ok[start:end].astype(np.float32),load,load,optimize=True)
            gram+=self.ridge*np.eye(load.shape[1])[None]
            rhs=centered[start:end]@load
            coefficient=np.linalg.solve(gram,rhs[...,None])[...,0]
            output[start:end]=(coefficient@load.T)*self.std+self.mean
        result=output.reshape(values.shape)*self.scale[None]
        return np.where(np.isfinite(values),values,result).astype(np.float32)


def context_rows(values, times, up, down, params, factor, row_slice):
    x,base=reconstruction_rows(values,times,up,down,params,row_slice=row_slice)
    rs,re=row_slice
    filled=interpolation(values)
    filled=np.where(np.isfinite(filled),filled,np.stack([params.free_speed_kmh,params.capacity_vph*.65],-1)[None])
    normalized=filled/SCALE
    blocks=[x, factor[rs:re]/SCALE]
    for operator in (up,down):
        neighbor=neighbors(normalized,operator)
        blocks.extend([neighbor[rs:re],(neighbor-normalized)[rs:re]])
        for lag in (-2,2): blocks.append(shift(neighbor,lag)[rs:re])
    for sigma in (1.,3.):
        smooth=gaussian_filter1d(normalized,sigma,axis=0,mode='nearest')
        blocks.extend([smooth[rs:re],(normalized-smooth)[rs:re]])
    index=np.arange(len(values))[:,None,None]
    valid=np.isfinite(values)
    previous=np.maximum.accumulate(np.where(valid,index,-len(values)),axis=0)
    following=np.minimum.accumulate(np.where(valid,index,2*len(values))[::-1],axis=0)[::-1]
    age=np.minimum(index-previous,144); lead=np.minimum(following-index,144)
    blocks.extend([age[rs:re]/12,lead[rs:re]/12,
        np.divide(age,age+lead,out=np.zeros_like(age,dtype=float),where=age+lead>0)[rs:re]])
    identity=np.arange(len(params),dtype=np.float32)/max(len(params)-1,1)
    blocks.append(np.broadcast_to(identity[None,:,None],(re-rs,len(params),1)))
    return np.concatenate(blocks,-1,dtype=np.float32),base


class EnhancedReconstructor:
    def context(self, *args):
        return context_rows(*args)

    def __init__(self, rounds=300, max_rows=160000):
        self.rounds,self.max_rows=rounds,max_rows

    def fit(self, values, times, up, down, params):
        self.factor=CrossSensorFactor().fit(values,params)
        rng=np.random.default_rng(641)
        # One missingness regime per day, matching the released mask layout.
        daily=rng.choice([.2,.3,.5],size=(int(np.ceil(len(values)/288)),1))
        rates=np.repeat(daily,288,axis=0)[:len(values)]
        hidden=(rng.random(values.shape[:2])<rates)&np.isfinite(values).all(-1)
        masked=values.copy();masked[hidden]=np.nan
        factor=self.factor.reconstruct(masked)
        eligible=np.flatnonzero(hidden.ravel())
        chosen=rng.choice(eligible,min(self.max_rows,len(eligible)),replace=False)
        xx,yy=[],[];n=values.shape[1]
        for start in range(0,len(values),288):
            end=min(start+288,len(values)); selected=chosen[(chosen//n>=start)&(chosen//n<end)]
            if not len(selected): continue
            lo=max(0,start-144);hi=min(len(values),end+144)
            x,b=self.context(masked[lo:hi],times[lo:hi],up,down,params,factor[lo:hi],(start-lo,end-lo))
            ti,ni=selected//n-start,selected%n
            xx.append(x[ti,ni]);yy.append((values.reshape(-1,2)[selected]-b[ti,ni])/SCALE)
        x,y=np.concatenate(xx),np.concatenate(yy)
        del xx,yy,masked,factor;gc.collect()
        self.models=[booster(seed=641+c,rounds=self.rounds).fit(x,y[:,c]) for c in range(2)]
        return self

    def reconstruct(self, values, times, up, down, params):
        factor=self.factor.reconstruct(values)
        output=np.empty_like(values)
        for start in range(0,len(values),288):
            end=min(start+288,len(values));lo=max(0,start-144);hi=min(len(values),end+144)
            x,base=self.context(values[lo:hi],times[lo:hi],up,down,params,factor[lo:hi],(start-lo,end-lo))
            prediction=np.stack([m.booster_.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(x.shape[:2])*SCALE[c]
                for c,m in enumerate(self.models)],-1)+base
            output[start:end]=np.where(np.isfinite(values[start:end]),values[start:end],np.maximum(prediction,0))
        return output
