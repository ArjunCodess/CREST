"""Stream a gated FD adjustment into a frozen, audited submission."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import network
from src.fd_calibration import soft_fd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--decisions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    decisions=[json.loads(p.read_text()) for p in args.decisions.glob('*decision.json')]
    chosen={r['panel']:r['selected_weight'] for r in decisions if r['accepted']}
    parameters={p:network(args.release_root,p).set_index('link_id') for p in chosen}
    dest=args.output/'submission.csv';partial=args.output/'submission.partial.csv'
    if dest.exists() or partial.exists():raise ValueError('Use a fresh candidate output directory')
    rows=changed=0
    sources=zip_longest(pd.read_csv(args.reference,dtype=str,chunksize=100000),
        pd.read_csv(args.release_root/'submission_key.csv',dtype=str,chunksize=100000))
    for frame,key in sources:
        if frame is None or key is None or not frame.submission_id.equals(key.submission_id):
            raise ValueError('Candidate/key order mismatch')
        if not frame.task.equals(key.task):raise ValueError('Candidate/key task mismatch')
        for panel,weight in chosen.items():
            mask=key.panel.eq(panel)&frame.task.eq('state')
            if not mask.any():continue
            params=parameters[panel].loc[key.loc[mask,'link_id']].reset_index()
            old=frame.loc[mask,['speed_kmh','flow_vph']].to_numpy(dtype=float)
            new=soft_fd(old[None],params,weight)[0]
            if not np.isfinite(new).all() or (new<0).any():raise ValueError('Invalid FD output')
            changed+=int(np.any(old!=new,axis=1).sum())
            frame.loc[mask,['speed_kmh','flow_vph']]=new.astype(str)
        frame.to_csv(partial,index=False,header=rows==0,mode='w' if rows==0 else 'a')
        rows+=len(frame)
    if rows!=6980503:raise ValueError('Incomplete submission')
    partial.replace(dest)
    with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    audit={'rows':rows,'changed_state_rows':changed,'promoted_weights':chosen,
        'queue_and_OD_preserved':True,'other_panels_preserved':True,
        'reference':str(args.reference),'sha256':digest}
    (args.output/'candidate_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
