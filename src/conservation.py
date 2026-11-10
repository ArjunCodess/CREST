"""Construct topology diagnostics only where every required public input exists."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_csv
from .physics import smooth_conservation

def public_flux_inputs(root,panel,split,times,links):
    topo=read_csv(Path(root)/'corridors'/panel/'network/lwr_mainline_topology.csv').set_index('link_id')
    index={l:i for i,l in enumerate(links)}; n=len(links)
    incoming=np.zeros((n,n)); outgoing=np.zeros((n,n)); known=np.ones(n,dtype=bool)
    net=np.zeros((len(times),n)); valid=np.ones_like(net,dtype=bool)
    ramp_parts=[]
    paths=sorted((Path(root)/'corridors'/panel/split/'ramp_states').rglob('*.parquet'))
    if paths:
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema=pq.ParquetFile(paths[0]).schema_arrow
        columns=[c for c in ('timestamp','ramp_link_id','pct_observed','flow_vph','is_score_eligible') if c in schema.names]
        stamp_type=schema.field('timestamp').type
        text_stamps=pa.types.is_string(stamp_type) or pa.types.is_large_string(stamp_type)
        first,last=(pd.Timestamp(times[0]),pd.Timestamp(times[-1]))
        first=first.strftime('%Y-%m-%dT%H:%M:%SZ') if text_stamps else first.to_pydatetime()
        last=last.strftime('%Y-%m-%dT%H:%M:%SZ') if text_stamps else last.to_pydatetime()
        for path in paths:
            x=pd.read_parquet(path,columns=columns,filters=[('timestamp','>=',first),('timestamp','<=',last)])
            if not x.empty:
                x['timestamp']=pd.to_datetime(x.timestamp,utc=True)
                ramp_parts.append(x)
    ramps=pd.concat(ramp_parts,ignore_index=True) if ramp_parts else pd.DataFrame()
    if len(ramps):
        ramps['ramp_link_id']=ramps.ramp_link_id.astype(str)
        ramps=ramps[(ramps.pct_observed>=75)&ramps.flow_vph.notna()]
        if 'is_score_eligible' in ramps: ramps=ramps[ramps.is_score_eligible.astype(bool)]
    for link,i in index.items():
        if link not in topo.index: known[i]=False; continue
        row=topo.loc[link]
        for name,op in [('incoming_link_ids',incoming),('outgoing_link_ids',outgoing)]:
            ids=[s for s in str(row.get(name,'')).split(';') if s and s!='nan']
            if not ids or any(s not in index for s in ids): known[i]=False
            for s in ids:
                if s in index: op[i,index[s]]=1
        for name,sign in [('on_ramp_link_ids',1),('off_ramp_link_ids',-1)]:
            ids=[s for s in str(row.get(name,'')).split(';') if s and s!='nan']
            for rid in ids:
                if not len(ramps): valid[:,i]=False; continue
                series=ramps[ramps.ramp_link_id.eq(rid)].groupby('timestamp').flow_vph.mean().reindex(times)
                valid[:,i]&=series.notna().to_numpy()
                net[:,i]+=sign*series.fillna(0).to_numpy()
    valid &= known[None]
    return incoming,outgoing,net,valid

def project_public_blocks(root,panel,split,data,pred,params,weight):
    incoming,outgoing,net,valid=public_flux_inputs(root,panel,split,data.times,data.links)
    reports=[]; result=pred.copy()
    for start in range(0,len(pred),288):
        end=min(len(pred),start+288)
        block,report=smooth_conservation(pred[start:end],data.values[start:end],params,incoming,outgoing,
             net[start:end],valid[start:end],weight=weight)
        result[start:end]=block; reports.append({'start':str(data.times[start]),**report})
    return result,reports
