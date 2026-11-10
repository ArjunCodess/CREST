"""Fit-only time-of-week reference with per-link fallback."""
import numpy as np
import pandas as pd

def slots(times):
    t=pd.DatetimeIndex(times)
    return np.asarray(t.dayofweek*288+t.hour*12+t.minute//5)

def historical_mean(values,fit_times,target_times):
    n=values.shape[1]; ids=(slots(fit_times)[:,None]*n+np.arange(n)[None]).ravel()
    result=[]
    for c in range(2):
        y=values[...,c].ravel(); valid=np.isfinite(y)
        sums=np.bincount(ids[valid],weights=y[valid],minlength=2016*n)
        counts=np.bincount(ids[valid],minlength=2016*n)
        means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0).reshape(2016,n)
        with np.errstate(invalid='ignore'): fallback=np.nanmean(values[...,c],axis=0)
        if not np.isfinite(fallback).all(): raise ValueError('A link has no finite fit-only baseline observations')
        result.append(np.where(np.isfinite(means[slots(target_times)]),means[slots(target_times)],fallback[None]))
    return np.stack(result,-1)
