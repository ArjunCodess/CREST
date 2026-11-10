"""Stream the official competition ZIP into its own data folder without storing the ZIP.

Accepts either a new Kaggle API token or legacy kaggle.json via a private file.
Never logs token contents, signed redirect URLs, or response bodies.
"""
import argparse
import hashlib
import json
import shutil
import os
from pathlib import Path
import requests
from stream_unzip import stream_unzip

URL='https://www.kaggle.com/api/v1/competitions/data/download-all/2026-ieee-big-data-traffic-flow-bench'

def include_file(name,essential=False):
    normalized=name.replace('\\','/')
    if essential and '/train/mainline_states_masked/' in normalized: return False
    if essential and normalized.startswith('kaggle_public/task1/') and '/train/' in normalized: return False
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--secret-file',type=Path)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--essential',action='store_true',help='Skip redundant train masks and train Task 1 templates; retain every input used by the participant pipeline')
    args=ap.parse_args(); target=args.destination.resolve(); target.mkdir(parents=True,exist_ok=True)
    secret=(args.secret_file.read_text(encoding='utf-8').strip() if args.secret_file else os.environ.get('KAGGLE_API_TOKEN',''))
    if not secret: raise RuntimeError('Supply a private --secret-file or KAGGLE_API_TOKEN environment variable')
    session=requests.Session()
    if secret.startswith('{'):
        config=json.loads(secret); session.auth=(config['username'],config['key'])
    else: session.headers['Authorization']='Bearer '+secret
    response=session.get(URL,stream=True,timeout=(30,180))
    if response.status_code!=200:
        status=response.status_code
        reason=''
        if 'json' in response.headers.get('Content-Type',''):
            message=str(response.json().get('message',''))
            if 'accept' in message.lower() and 'rules' in message.lower():
                reason=' The token owner must accept the competition rules on Kaggle before downloading.'
        response.close(); session.close()
        raise RuntimeError(f'Kaggle download returned HTTP {status}.{reason}')
    digest=hashlib.sha256(); total=0; completed=0; excluded=0
    def chunks():
        nonlocal total
        for chunk in response.iter_content(chunk_size=1024*1024):
            digest.update(chunk); total+=len(chunk); yield chunk
    try:
        for name,size,parts in stream_unzip(chunks()):
            if not include_file(name.decode('utf-8'),args.essential):
                for _ in parts: pass
                excluded+=1; continue
            relative=Path(name.decode('utf-8'))
            path=(target/relative).resolve()
            if not path.is_relative_to(target): raise ValueError('Unsafe archive path')
            if str(name.decode('utf-8')).endswith('/'):
                for _ in parts: pass
                path.mkdir(parents=True,exist_ok=True); continue
            path.parent.mkdir(parents=True,exist_ok=True)
            skip=path.exists() and path.stat().st_size==size
            if not skip and shutil.disk_usage(target).free<size+128*1024**2:
                raise RuntimeError(f'Insufficient disk space to extract {relative}')
            temp=path.with_name(path.name+'.partial')
            if skip:
                for _ in parts: pass
            else:
                with temp.open('wb') as f:
                    for part in parts: f.write(part)
                temp.replace(path)
            completed+=1
            if completed%100==0: print(f'{completed} files extracted; {total/1024**3:.2f} GiB transferred',flush=True)
    finally: response.close(); session.close()
    (target/'download_manifest.json').write_text(json.dumps({'source':'official Kaggle competition API',
        'archive_sha256':digest.hexdigest(),'download_bytes':total,'files':completed,
        'essential_subset':args.essential,'excluded_files':excluded},indent=2))
    print(f'Complete: {completed} files at {target}')

if __name__=='__main__': main()
