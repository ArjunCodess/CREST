"""LightGBM and masked graph attention with a temporal Transformer."""
import numpy as np
import torch
from torch import nn
from lightgbm import LGBMRegressor
from .features import SCALE, features, artificial_mask, sampled_features, prediction_blocks

def booster(seed=42, rounds=160):
    return LGBMRegressor(n_estimators=rounds, learning_rate=.04, num_leaves=31,
        min_child_samples=50, reg_lambda=2, subsample=.85, colsample_bytree=.85,
        random_state=seed, n_jobs=4, verbosity=-1)

class GraphTemporal(nn.Module):
    """Shared weights allow different graph sizes; graph mask includes both directions."""
    def __init__(self, width=32):
        super().__init__()
        self.embedding = nn.Linear(4, width)
        self.spatial = nn.MultiheadAttention(width, 4, dropout=.05, batch_first=True)
        layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=.05, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.randn(25, width)*.01)
        self.state = nn.Linear(width, 2)
        self.future = nn.Linear(width, 12)

    def encode_space(self,x,adjacency):
        b,t,n,_ = x.shape
        h = self.embedding(x).reshape(b*t,n,-1)
        mask = ~(adjacency.bool() | adjacency.T.bool() | torch.eye(n,device=x.device,dtype=torch.bool))
        z,_ = self.spatial(h,h,h,attn_mask=mask,need_weights=False)
        return (h+z).reshape(b,t,n,-1)

    def decode_time(self,encoded,center=None):
        b,t,n,_=encoded.shape
        h=encoded.permute(0,2,1,3).reshape(b*n,t,-1)
        h = self.temporal(h+self.position[:t])
        center = t-1 if center is None else center
        return (self.state(h[:,center]).reshape(b,n,2),
                self.future(h[:,-1]).reshape(b,n,6,2).permute(0,2,1,3))

    def forward(self,x,adjacency,center=None):
        return self.decode_time(self.encode_space(x,adjacency),center)

def tensor_input(x):
    valid = np.isfinite(x)
    return torch.tensor(np.concatenate([np.nan_to_num(x/SCALE),valid],-1),dtype=torch.float32)

def fit_neural(values, adjacency, epochs=3, batches=60, seed=42, params=None):
    """Self-supervised denoising and future speed/flow; queue truth is unavailable."""
    torch.manual_seed(seed); torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    model = GraphTemporal(); opt = torch.optim.AdamW(model.parameters(),lr=.002)
    a = torch.tensor(adjacency,dtype=torch.float32)
    if len(values)<40: raise ValueError('Need at least 40 five-minute steps for neural fit')
    for _ in range(epochs):
        model.train()
        for _ in range(batches):
            origins = rng.integers(12,len(values)-12,size=2)
            # Center denoising uses +/- 60 minutes; forecasting uses only past.
            clean = np.stack([values[t-12:t+13] for t in origins])
            masked = clean.copy()
            hidden = rng.random(clean.shape[:-1])<.3
            masked[hidden] = np.nan
            pred,_ = model(tensor_input(masked),a,center=12)
            target = torch.tensor(clean[:,12]/SCALE,dtype=torch.float32)
            ok = torch.isfinite(target) & torch.tensor(hidden[:,12,:,None])
            loss = ((pred-target.nan_to_num())**2)[ok].mean() if ok.any() else pred.sum()*0
            if params is not None:
                from .physics import differentiable_fd
                vf=torch.tensor(params.free_speed_kmh.to_numpy(),dtype=torch.float32)
                cap=torch.tensor(params.capacity_vph.to_numpy(),dtype=torch.float32)
                kj=torch.tensor(params.k_jam.to_numpy(),dtype=torch.float32)
                scales=torch.tensor(SCALE)
                feasible=differentiable_fd(pred*scales,vf,cap,kj)/scales
                loss=loss+.02*((pred-feasible)**2).mean()
            past = np.stack([values[t-12:t+1] for t in origins])
            future = np.stack([values[t+1:t+7] for t in origins])/SCALE
            _,fp = model(tensor_input(past),a)
            ft = torch.tensor(future,dtype=torch.float32); valid = torch.isfinite(ft)
            if valid.any(): loss = loss+((fp-ft.nan_to_num())**2)[valid].mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    model.eval()
    return model

def neural_reconstruct(model, values, adjacency):
    a=torch.tensor(adjacency,dtype=torch.float32)
    padded=np.pad(values,((12,12),(0,0),(0,0)),constant_values=np.nan)
    out=[]
    with torch.no_grad():
        # Spatial encoding depends only on the current observations. Cache once
        # instead of recomputing the same 25 overlapping timestamps per target.
        encoded=[]
        for start in range(0,len(padded),128):
            encoded.append(model.encode_space(tensor_input(padded[start:start+128][None]),a)[0].numpy())
        encoded=np.concatenate(encoded)
        for start in range(0,len(values),32):
            batch=np.stack([encoded[t:t+25] for t in range(start,min(start+32,len(values)))])
            out.append(model.decode_time(torch.tensor(batch),center=12)[0].numpy()*SCALE)
    return np.concatenate(out)

class Hybrid:
    def __init__(self, rounds=160, seed=42):
        self.rounds=rounds; self.seed=seed
    def fit(self, values, times, up, down, params, max_rows=200000, neural_epochs=3):
        rng=np.random.default_rng(self.seed)
        masked, hidden=artificial_mask(values,rng)
        x,ix=sampled_features(masked,times,up,down,params,hidden,rng,max_rows,True)
        if len(ix)<10: raise ValueError('Too few eligible masked training cells')
        self.trees=[]
        for c in range(2):
            m=booster(self.seed,self.rounds)
            m.fit(x,values[...,c].ravel()[ix]/SCALE[c]); self.trees.append(m)
        del x,ix,masked,hidden
        import gc; gc.collect()
        self.neural=(fit_neural(values,up+down,epochs=neural_epochs,seed=self.seed,params=params) if neural_epochs>0 else None)
        self.queue=[]
        for horizon in range(1,7):
            hmodels=[]
            eligible=np.isfinite(values[horizon:]).all(-1)
            xx,origins=sampled_features(values[:-horizon],times[:-horizon],up,down,params,eligible,rng,max_rows,False)
            for c in range(2):
                y=values[horizon:,:,c]/SCALE[c]
                m=booster(self.seed+horizon,self.rounds); m.fit(xx,y.ravel()[origins]); hmodels.append(m)
            self.queue.append(hmodels)
        return self
    def reconstruct(self, values,times,up,down,params,neural_prediction=None):
        current=values.copy()
        weight=getattr(self,'tree_weight',.8)
        nn_prediction=(neural_prediction if neural_prediction is not None else
            neural_reconstruct(self.neural,values,up+down) if weight<1 and self.neural is not None else np.zeros_like(values))
        for _ in range(2):
            pred=np.empty_like(values)
            for start,end,x in prediction_blocks(current,times,up,down,params,True):
                pred[start:end]=np.stack([m.booster_.predict(x.reshape(-1,x.shape[-1]),num_threads=4).reshape(x.shape[:2])*SCALE[c]
                                         for c,m in enumerate(self.trees)],-1)
            weight=getattr(self,'tree_weight',.8) if self.neural is not None else 1.
            pred=weight*pred+(1-weight)*nn_prediction
            current=np.where(np.isfinite(values),values,np.maximum(pred,0))
        return current
    def forecast(self, values,times,up,down,params):
        x=features(values,times,up,down,params,False)[-1]
        tree=np.stack([np.stack([m.booster_.predict(x,num_threads=4)*SCALE[c] for c,m in enumerate(ms)],-1) for ms in self.queue])
        weight=getattr(self,'tree_weight',.8) if self.neural is not None else 1.
        if weight>=1: return np.maximum(tree,0)
        with torch.no_grad():
            nn_future=self.neural(tensor_input(values[-13:][None]),torch.tensor(up+down,dtype=torch.float32))[1][0].numpy()*SCALE
        return np.maximum(weight*tree+(1-weight)*nn_future,0)
