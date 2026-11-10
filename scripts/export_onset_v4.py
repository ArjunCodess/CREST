"""Replace only onset windows; preserve all ongoing predictions from the scored entry."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.queue_events import history_arrays
from src.queue_onset import decode_onset
from src.submission import keyed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--baseline-queue',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original=keyed(pd.read_csv(args.baseline_queue),'queue');output=original.copy()
    manifest=json.loads((args.model_root/'model_manifest.json').read_text());details=[]
    for row in manifest['models']:
        path=args.model_root/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with path.open('rb') as f:payload=pickle.load(f)
        model=payload['model'];panel=row['panel']
        for split in ('validation','private'):
            folder=args.release_root/'task2'/panel/split
            index=pd.read_csv(folder/'window_index.csv',dtype={'window_id':str})
            index=index[index.condition.eq('queue_onset')]
            history=pd.read_parquet(folder/'window_history.parquet')
            history['timestamp']=pd.to_datetime(history.timestamp,utc=True);history['link_id']=history.link_id.astype(str)
            for w in index.itertuples():
                origin=pd.Timestamp(w.forecast_origin)
                visible=history[history.window_id.eq(w.window_id)]
                at_origin=visible.iloc[:0].copy()
                if model.policy=='origin_observed':
                    parts=[]
                    for p in (args.release_root/'corridors'/panel/split/'mainline_states_masked').rglob('*'+origin.strftime('%Y_%m_%d')+'.parquet'):
                        raw=pd.read_parquet(p);raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True)
                        parts.append(raw[raw.timestamp.eq(origin)])
                    if not parts:raise ValueError('Missing permitted origin observations')
                    at_origin=pd.concat(parts,ignore_index=True);at_origin['link_id']=at_origin.link_id.astype(str)
                h,a,times=history_arrays(visible,at_origin,origin,payload['links'],model.policy,payload['density_scale'])
                scores=model.scores(h,a,times,'queue_onset',payload['up'],payload['down'],payload['params'])
                pred=decode_onset(scores,payload['decoder'])
                target=output.xs(w.window_id,level='window_id',drop_level=False)
                stamps=target.index.get_level_values('timestamp')
                links=target.index.get_level_values('link_id')
                steps=pd.date_range(origin+pd.Timedelta(minutes=5),periods=6,freq='5min')
                ti=steps.get_indexer(stamps);ni=pd.Index(payload['links']).get_indexer(links)
                if (ti<0).any() or (ni<0).any():raise ValueError('Unknown onset export key')
                output.loc[target.index,'queue_pred']=pred[ti,ni].astype(int)
                details.append({'panel':panel,'split':split,'window_id':w.window_id,'positive_cells':int(pred.sum()),'kind':payload['kind']})
            print('Exported onset',panel,split,flush=True)
    frame=output.reset_index();frame['timestamp']=frame.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame['queue_pred']=frame.queue_pred.astype(int);frame.to_csv(args.output/'queue.csv',index=False)
    audit={'rows':len(frame),'changed_cells':int((output.queue_pred!=original.queue_pred).sum()),
        'windows':details,'first_five_onset_steps_empty':True,'ongoing_preserved':True,
        'sha256':hashlib.sha256((args.output/'queue.csv').read_bytes()).hexdigest()}
    (args.output/'queue_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
