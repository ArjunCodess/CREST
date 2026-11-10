"""Run checkpointed improvements sequentially to keep memory use bounded."""
import argparse
from pathlib import Path
import subprocess
import sys
import json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['all','reconstruction','queue'],default='all'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    panels=sorted(p.name for p in (args.release_root/'corridors').iterdir() if (p/'network/links.csv').exists())
    scripts={'reconstruction':['refine_reconstruction.py'],'queue':['train_queue.py'],
             'all':['refine_reconstruction.py','train_queue.py']}[args.stage]
    for script in scripts:
        print('Start',script,flush=True)
        for panel in panels:
            subprocess.run([sys.executable,'-u',str(root/'scripts'/script),'--release-root',str(args.release_root),
                            '--output',str(args.output),'--panel',panel],cwd=root,check=True)
    print('All comparison stages finished',flush=True)

if __name__=='__main__': main()
