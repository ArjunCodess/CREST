"""Strict merge: missing required predictions are errors, never silently zeros."""
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3

KEYS={'state':['panel','timestamp','station_id','link_id','mask_regime'],
      'queue':['window_id','timestamp','link_id'],
      'odme':['panel','departure_time','path_id']}
VALUES={'state':['speed_kmh','flow_vph'],'queue':['queue_pred'],'odme':['path_flow']}
COLUMNS=['submission_id','task','speed_kmh','flow_vph','queue_pred','path_flow']

def keyed(frame,task):
    x=frame.copy()
    for c in KEYS[task]:
        x[c]=pd.to_datetime(x[c],utc=True) if c=='timestamp' else x[c].astype(str)
    if x.duplicated(KEYS[task]).any(): raise ValueError(f'Duplicate {task} natural keys')
    values=x[VALUES[task]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any(): raise ValueError(f'Invalid {task} values')
    if task=='queue' and not np.isin(values,[0,1]).all(): raise ValueError('Queue values must be binary')
    x[VALUES[task]]=values
    return x.set_index(KEYS[task])[VALUES[task]]

def merge(root,task_frames,output):
    root=Path(root); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    tables={k:keyed(v,k) for k,v in task_frames.items()}
    temporary=output.with_suffix('.partial.csv'); first=True; n=0
    try:
        sample=pd.read_csv(root/'sample_submission.csv',chunksize=250000,dtype=str)
        for chunk in pd.read_csv(root/'submission_key.csv',chunksize=250000,dtype=str):
            expected=next(sample)
            if list(expected.columns)!=COLUMNS: raise ValueError('Unexpected official sample submission columns')
            if not np.array_equal(chunk.submission_id.to_numpy(),expected.submission_id.to_numpy()):
                raise ValueError('Key and sample submission row order differ')
            if not np.array_equal(chunk.task.to_numpy(),expected.task.to_numpy()): raise ValueError('Task order mismatch')
            if not set(chunk.task)<=set(tables): raise ValueError('Unknown submission task')
            if 'timestamp' in chunk: chunk['timestamp']=pd.to_datetime(chunk.timestamp,utc=True,errors='coerce')
            out=chunk[['submission_id','task']].copy()
            for c in COLUMNS[2:]: out[c]=0.
            for task,table in tables.items():
                mask=chunk.task.eq(task)
                if not mask.any(): continue
                ix=pd.MultiIndex.from_frame(chunk.loc[mask,KEYS[task]])
                found=table.reindex(ix)
                if found.isna().any().any(): raise ValueError(f'Missing {task} predictions for submission key')
                out.loc[mask,VALUES[task]]=found.to_numpy()
            out[COLUMNS].to_csv(temporary,index=False,header=first,mode='w' if first else 'a')
            first=False; n+=len(out)
        try:
            next(sample)
            raise ValueError('Sample submission has extra rows')
        except StopIteration: pass
        temporary.replace(output)
    except Exception:
        # Keep any partial file clearly named and never expose it as submission.csv.
        raise
    return n

def merge_files(root,paths,output):
    """Disk-backed natural-key joins keep the full release below the RAM limit."""
    root=Path(root); output=Path(output); database=output.with_suffix('.sqlite')
    if database.exists(): database.unlink()
    db=sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA cache_size=-64000'); db.execute('PRAGMA temp_store=MEMORY')
    for task,path in paths.items():
        keys=KEYS[task]; values=VALUES[task]
        fields=','.join([f'"{c}" TEXT NOT NULL' for c in keys]+[f'"{c}" REAL NOT NULL' for c in values])
        db.execute(f'CREATE TABLE {task} ({fields}, PRIMARY KEY ({",".join(keys)})) WITHOUT ROWID')
        for chunk in pd.read_csv(path,dtype={c:str for c in keys},chunksize=50000):
            if 'timestamp' in keys:
                chunk['timestamp']=pd.to_datetime(chunk.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            numeric=chunk[values].to_numpy(dtype=float)
            if not np.isfinite(numeric).all() or (numeric<0).any(): raise ValueError(f'Invalid {task} predictions')
            if task=='queue' and not np.isin(numeric,[0,1]).all(): raise ValueError('Queue predictions must be binary')
            chunk[values]=numeric
            db.executemany(f'INSERT INTO {task} VALUES ({",".join("?" for _ in keys+values)})',
                           chunk[keys+values].itertuples(index=False,name=None))
            db.commit()
        print('Indexed',task,flush=True)
    temporary=output.with_suffix('.partial.csv'); first=True; total=0
    sample=pd.read_csv(root/'sample_submission.csv',chunksize=50000,dtype=str)
    for chunk in pd.read_csv(root/'submission_key.csv',chunksize=50000,dtype=str):
        chunk=chunk.reset_index(drop=True); expected=next(sample)
        if list(expected.columns)!=COLUMNS: raise ValueError('Unexpected sample submission columns')
        if not np.array_equal(chunk.submission_id.to_numpy(),expected.submission_id.to_numpy()): raise ValueError('Submission ID order mismatch')
        if not np.array_equal(chunk.task.to_numpy(),expected.task.to_numpy()): raise ValueError('Submission task order mismatch')
        if not set(chunk.task)<=set(paths): raise ValueError('Unknown submission task')
        out=chunk[['submission_id','task']].copy()
        for c in COLUMNS[2:]: out[c]=0.
        for task in paths:
            mask=chunk.task.eq(task)
            if not mask.any(): continue
            keys=KEYS[task]; values=VALUES[task]; wanted=chunk.loc[mask,keys].copy()
            if 'timestamp' in keys: wanted['timestamp']=pd.to_datetime(wanted.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            db.execute('DROP TABLE IF EXISTS wanted')
            db.execute('CREATE TEMP TABLE wanted (rid INTEGER PRIMARY KEY,'+','.join(f'"{c}" TEXT' for c in keys)+')')
            db.executemany('INSERT INTO wanted VALUES ('+','.join('?' for _ in range(len(keys)+1))+')',wanted.itertuples(index=True,name=None))
            rows=db.execute(f'SELECT w.rid,{",".join("t."+c for c in values)} FROM wanted w LEFT JOIN {task} t USING ({",".join(keys)}) ORDER BY w.rid').fetchall()
            if any(any(v is None for v in r[1:]) for r in rows): raise ValueError(f'Missing {task} predictions')
            out.loc[[r[0] for r in rows],values]=np.asarray([r[1:] for r in rows],dtype=float)
        out[COLUMNS].to_csv(temporary,index=False,header=first,mode='w' if first else 'a')
        first=False; total+=len(out)
        if total%500000==0: print(f'Merged {total:,} rows',flush=True)
    extra=next(sample,None)
    if extra is not None: raise ValueError('Sample submission contains extra rows')
    db.close(); temporary.replace(output)
    return total
