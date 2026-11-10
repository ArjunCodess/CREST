"""Refit temporal-gated onset localization, keeping corrected onset baselines elsewhere."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lightgbm import LGBMClassifier
from src.data import load_state,graph
from src.queue_events import load_auxiliary,calibrate_occupancy_density,visible_event,event_features,EventQueueModel
from src.queue_onset import OnsetLocalizer,onset_events


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'model_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'models':[]}
    for decision_path in sorted(args.pilot.glob('*_decision.json')):
        decision=json.loads(decision_path.read_text());panel=decision['panel']
        if any(r['panel']==panel for r in manifest['models']):continue
        print('Full onset refit',panel,'localizer' if decision['accepted'] else 'corrected baseline',flush=True)
        d=load_state(args.release_root,panel,'train',True)
        aux=load_auxiliary(args.release_root,panel,d.times,d.links)
        density_scale=calibrate_occupancy_density(aux);aux[...,1]=aux[...,0]*density_scale
        up,down,_,params=graph(args.release_root,panel,d.links)
        events=onset_events(d.values,aux,d.times,np.ones(len(d.times),bool),.6*params.free_speed_kmh.to_numpy())
        selected=decision['selected'];policy=selected['policy']
        if decision['accepted']:
            model=OnsetLocalizer(policy=policy).fit(events,up,down,params)
            model.tree_weight,model.prior_weight,model.decoder=selected['tree_weight'],selected['prior_weight'],selected['decoder']
            decoder=model.decoder;kind='localizer'
        else:
            xx,yy,weights=[],[],[]
            for w in events:
                h,a=visible_event(w,policy);valid=w.valid[-1]
                xx.append(event_features(h,a,w.times,up,down,params)[-len(params):][valid])
                yy.append(w.truth[-1,valid]);weights.extend([1/valid.sum()]*valid.sum())
            weight=np.asarray(weights);weight*=len(weight)/weight.sum()
            classifier=LGBMClassifier(n_estimators=180,num_leaves=15,min_child_samples=60,reg_lambda=6,
                learning_rate=.04,n_jobs=4,verbosity=-1,random_state=318)
            classifier.fit(np.concatenate(xx),np.concatenate(yy),sample_weight=weight)
            model=EventQueueModel(policy=policy);model.models={'queue_onset':classifier}
            decoder=decision['baseline_decoder'];model.decoders['queue_onset']=decoder;kind='corrected_baseline'
        path=args.output/f'{panel}_onset.pkl'
        payload={'model':model,'links':d.links,'up':up,'down':down,'params':params,'density_scale':density_scale,
            'decoder':decoder,'kind':kind,'fit_start':str(d.times[0]),'fit_end':str(d.times[-1]),'training_events':len(events)}
        with path.open('wb') as f:pickle.dump(payload,f)
        manifest['models'].append({'panel':panel,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'kind':kind,'policy':policy,'decoder':decoder,'training_events':len(events)})
        manifest['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/queue_onset.py'),Path('src/queue_events.py'),Path(__file__)]}
        manifest_path.write_text(json.dumps(manifest,indent=2))
        del d,aux,model,events;gc.collect()
    print('Full onset refits complete',flush=True)


if __name__=='__main__':main()
