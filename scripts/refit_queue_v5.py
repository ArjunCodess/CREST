"""Refit accepted ongoing heads; preserve v4 onset/state/OD predictions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import load_state,graph
from src.queue_events import load_auxiliary,calibrate_occupancy_density,collect_events
from src.queue_onset import onset_events
from src.queue_v5 import LocalizedQueueModel,BlendedQueueModel


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--decisions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    v4=Path('artifacts/v4_delivery/models/events')
    eligible={r['panel']:r for r in json.loads((v4/'model_manifest.json').read_text())['models']}
    manifest_path=args.output/'model_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'method':'Localized ongoing queues, training-only temporal promotion',
        'onset_preserved':True,'state_preserved':True,'od_preserved':True,'models':[]}
    for path in sorted(args.decisions.glob('*_decision.json')):
        decision=json.loads(path.read_text());panel=decision['panel']
        if not decision['accepted']['queue_ongoing'] or panel not in eligible:continue
        if any(r['panel']==panel for r in manifest['models']):continue
        print('Refit localized queues',panel,flush=True)
        with (v4/eligible[panel]['file']).open('rb') as f: baseline=pickle.load(f)['model']
        d=load_state(args.release_root,panel,'train',True)
        aux=load_auxiliary(args.release_root,panel,d.times,d.links)
        scale=calibrate_occupancy_density(aux);aux[...,1]=aux[...,0]*scale
        up,down,_,params=graph(args.release_root,panel,d.links)
        active=np.ones(len(d.times),bool);cutoff=.6*params.free_speed_kmh.to_numpy()
        events=[e for e in collect_events(d.values,aux,d.times,active,cutoff,
            np.random.default_rng(851),max_per_condition=256) if e.condition=='queue_ongoing']
        events+=onset_events(d.values,aux,d.times,active,cutoff,max_events=256,seed=851)
        candidate=LocalizedQueueModel(policy=decision['policy']).fit(events,up,down,params)
        config=decision['configurations']['queue_ongoing']
        model=BlendedQueueModel(candidate,baseline,config['weight'],config['decoder'])
        payload={'model':model,'links':d.links,'up':up,'down':down,'params':params,
            'density_scale':scale,'fit_start':str(d.times[0]),'fit_end':str(d.times[-1]),
            'selection':decision,'training_events':len(events)}
        dest=args.output/f'{panel}_ongoing.pkl'
        with dest.open('wb') as f:pickle.dump(payload,f)
        manifest['models'].append({'panel':panel,'file':dest.name,
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'policy':model.policy,
            'decoder':config['decoder'],'weight':config['weight']})
        manifest_path.write_text(json.dumps(manifest,indent=2))
        del d,aux,candidate,model,events,payload,baseline;gc.collect()


if __name__=='__main__':main()
