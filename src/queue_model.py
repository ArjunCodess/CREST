"""Six-horizon classifiers trained only on observed-speed queue proxies."""
import numpy as np
from lightgbm import LGBMClassifier
from .features import features,sampled_features
from .models import booster

# Chosen on the February selection block only (k = 1..8). Untouched week scored once.
# The horizon is whichever step the speed forecast actually drops; step 6 is not pinned.
ONSET_DROPS = 2

def onset_slowdown(forecast_speed, history_speed, k=ONSET_DROPS):
    """Cells an onset window should mark when every forecast speed stays above v_cut.

    Rank is the drop from the last finite speed in the visible history. An empty
    history stays empty: a guessed link still scores IoU 0.
    """
    speed=np.asarray(history_speed,dtype=float); forecast=np.asarray(forecast_speed,dtype=float)
    pred=np.zeros(forecast.shape,dtype=bool)
    if speed.ndim!=2 or forecast.shape[-1]!=speed.shape[1]: return pred
    finite=np.isfinite(speed)
    if not finite.any(): return pred
    taken=speed.shape[0]-1-np.argmax(finite[::-1],axis=0); anyf=finite.any(0)
    last=np.full(speed.shape[1],np.nan); last[anyf]=speed[taken[anyf],np.flatnonzero(anyf)]
    drop=last[None,:]-forecast
    usable=np.isfinite(drop)
    if not usable.any(): return pred
    score=np.where(usable,drop,-np.inf)
    pred.ravel()[np.argsort(-score.ravel(),kind='mergesort')[:min(k,int(usable.sum()))]]=True
    return pred

class QueueClassifier:
    def fit(self,values,times,up,down,params):
        rng=np.random.default_rng(68); self.models=[]; self.prior_adjust=[]
        self.active_links=np.isfinite(values[...,0]).any(axis=0)
        current_mask=rng.random(values.shape[:2],dtype=np.float32)<rng.choice([.2,.3,.5],size=(len(values),1))
        cutoff=params.free_speed_kmh.to_numpy()*.6
        for horizon in range(1,7):
            future=values[horizon:,:,0]; valid=np.isfinite(future); target=future<=cutoff
            selected=np.zeros(valid.shape,dtype=bool)
            for label,limit in [(True,12000),(False,36000)]:
                available=np.flatnonzero(valid.ravel() & (target.ravel()==label))
                picked=rng.choice(available,min(limit,len(available)),replace=False)
                selected.ravel()[picked]=True
            x,ix=sampled_features(values[:-horizon],times[:-horizon],up,down,params,selected,rng,48000,False,current_mask[:-horizon])
            y=target.ravel()[ix].astype(int)
            population=float(target[valid].mean()); sampled=float(y.mean())
            self.prior_adjust.append((population/max(1-population,1e-9))/(sampled/max(1-sampled,1e-9)) if 0<sampled<1 else 1.)
            if len(np.unique(y))<2:
                self.models.append(float(y.mean())); continue
            m=LGBMClassifier(n_estimators=150,learning_rate=.05,num_leaves=31,
                min_child_samples=60,reg_lambda=3,random_state=68+horizon,n_jobs=4,verbosity=-1)
            m.fit(x,y); self.models.append(m)
        self.thresholds={'queue_onset':.5,'queue_ongoing':.5}; self.force_nonempty=False
        return self
    def scores(self,values,times,up,down,params):
        x=features(values,times,up,down,params,False)[-1]
        p=np.stack([np.full(len(x),m) if isinstance(m,float) else m.booster_.predict(x,num_threads=4) for m in self.models])
        adjustment=np.asarray(getattr(self,'prior_adjust',np.ones(6)))[:,None]
        return p*adjustment/np.maximum(1-p+p*adjustment,1e-12)
    def classify(self,probabilities,condition):
        active=getattr(self,'active_links',np.ones(probabilities.shape[1],dtype=bool))
        result=(probabilities>=self.thresholds.get(condition,.5)) & active[None]
        if self.force_nonempty and not result.any():
            result.ravel()[np.argmax(np.where(active[None],probabilities,-np.inf))]=True
        return result


class QueueSpeedRegressor:
    """Speed baseline retrained on the same prefix, with the origin masked."""
    def fit(self,values,times,up,down,params):
        rng=np.random.default_rng(42); self.models=[]
        current_mask=rng.random(values.shape[:2])<rng.choice([.2,.3,.5],size=(len(values),1))
        for horizon in range(1,7):
            eligible=np.isfinite(values[horizon:,:,0])
            x,ix=sampled_features(values[:-horizon],times[:-horizon],up,down,params,eligible,rng,60000,False,current_mask[:-horizon])
            model=booster(seed=42+horizon,rounds=80)
            model.fit(x,values[horizon:,:,0].ravel()[ix]/120)
            self.models.append(model)
        return self

    def scores(self,values,times,up,down,params):
        x=features(values,times,up,down,params,False)[-1]
        return np.stack([m.booster_.predict(x,num_threads=4)*120 for m in self.models])
