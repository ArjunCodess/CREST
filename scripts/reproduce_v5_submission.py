"""Reproduce v5 from the release and packaged v4/v5 checkpoints."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--package-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--verified-base',type=Path,help='Optional earlier v4 reproduction with a matching audit and SHA-256')
    args=ap.parse_args()
    release,package,output=args.release_root.resolve(),args.package_root.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    actual=output/'submission.csv'
    if actual.exists():raise ValueError('Use a fresh v5 output directory')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    base=args.verified_base.resolve() if args.verified_base else output/'base'
    if args.verified_base:
        audit=json.loads((base/'reproduction_audit.json').read_text())
        with (base/'submission.csv').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if not audit['all_tasks_regenerated_from_checkpoints_and_release'] or digest!=audit['sha256']:
            raise ValueError('Verified base audit/hash mismatch')
    else:
        subprocess.run([sys.executable,str(package/'scripts/reproduce_v4_submission.py'),
            '--release-root',str(release),'--package-root',str(package),'--output',str(base)],
            cwd=package,env=env,check=True)
    subprocess.run([sys.executable,str(package/'scripts/export_queue_v5.py'),
        '--release-root',str(release),'--models',str(package/'models/ongoing'),
        '--baseline-queue',str(base/'onset/queue.csv'),'--output',str(output/'queue')],
        cwd=package,env=env,check=True)
    shutil.copyfile(base/'submission.csv',actual)
    subprocess.run([sys.executable,str(package/'scripts/splice_queue_rows.py'),
        '--submission',str(actual),'--queue',str(output/'queue/queue.csv'),
        '--key',str(release/'submission_key.csv'),'--apply'],cwd=package,env=env,check=True)
    fd=package/'models/fd_decisions'
    if fd.exists() and any(json.loads(p.read_text())['accepted'] for p in fd.glob('*decision.json')):
        subprocess.run([sys.executable,str(package/'scripts/build_fd_v5_candidate.py'),
            '--release-root',str(release),'--reference',str(actual),'--decisions',str(fd),
            '--output',str(output/'fd')],cwd=package,env=env,check=True)
        (output/'fd/submission.csv').replace(actual)
    rows=0;maximum=np.zeros(4)
    if args.reference:
        for x,y in zip_longest(pd.read_csv(actual,chunksize=100000),pd.read_csv(args.reference,chunksize=100000)):
            if x is None or y is None or not x.iloc[:,:2].equals(y.iloc[:,:2]):raise ValueError('Reference order mismatch')
            xx,yy=x.iloc[:,2:].to_numpy(),y.iloc[:,2:].to_numpy()
            np.testing.assert_allclose(xx,yy,atol=1e-6,rtol=1e-6)
            maximum=np.maximum(maximum,np.abs(xx-yy).max(0));rows+=len(x)
    else:
        rows=sum(len(x) for x in pd.read_csv(actual,usecols=['submission_id'],chunksize=100000))
    if rows!=6980503:raise ValueError('Incomplete reproduction')
    with actual.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    audit={'rows':rows,'sha256':digest,'all_tasks_regenerated_from_checkpoints_and_release':True,
        'reference_compared':bool(args.reference),'matches_selected_submission_within_1e_6':True if args.reference else None,
        'max_absolute_difference':maximum.tolist() if args.reference else None,
        'reference_used_only_after_prediction':True,'verified_v4_base_reused':bool(args.verified_base)}
    (output/'reproduction_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
