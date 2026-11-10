"""Replace promoted ongoing windows only, keeping the frozen v4 onset table."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.queue_events import predict_release_queue
from src.submission import keyed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--baseline-queue',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original=keyed(pd.read_csv(args.baseline_queue),'queue');result=original.copy()
    manifest=json.loads((args.models/'model_manifest.json').read_text());details=[]
    for row in manifest['models']:
        path=args.models/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Checkpoint hash mismatch')
        with path.open('rb') as f:payload=pickle.load(f)
        for split in ('validation','private'):
            windows=pd.read_csv(args.release_root/'task2'/row['panel']/split/'window_index.csv')
            ids=set(windows.loc[windows.condition.eq('queue_ongoing'),'window_id'])
            prediction=predict_release_queue(args.release_root,row['panel'],split,payload)
            replacement=keyed(prediction[prediction.window_id.isin(ids)],'queue')
            if len(replacement.index.difference(original.index)):raise ValueError('Unknown ongoing keys')
            result.loc[replacement.index,'queue_pred']=replacement.queue_pred.to_numpy()
            details.append({'panel':row['panel'],'split':split,'windows':len(ids),'rows':len(replacement)})
            print('Exported ongoing',row['panel'],split,flush=True)
    frame=result.reset_index();frame.timestamp=frame.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.queue_pred=frame.queue_pred.astype(int);dest=args.output/'queue.csv';frame.to_csv(dest,index=False)
    audit={'rows':len(frame),'changed_cells':int((result.queue_pred!=original.queue_pred).sum()),
        'onset_preserved':True,'windows':details,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (args.output/'queue_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
