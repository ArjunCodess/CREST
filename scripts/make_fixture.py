"""Small schema fixture for integration tests. Never a competition dataset."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def make(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    (root/'FIXTURE_ONLY.json').write_text(json.dumps({'warning':'Software integration fixture. Not official TrafficFlowBench data; do not submit.'}))
    rng=np.random.default_rng(2026); combined=[]; key=[]; sid=0
    for panel in ['D12_I5_N','D12_I5_S','D7_I10_E']:
        links=[str(i) for i in range(4)]
        net=root/'corridors'/panel/'network'; net.mkdir(parents=True,exist_ok=True)
        params=pd.DataFrame({'link_id':links,'length_km':1.,'lanes':4.,'free_speed_kmh':110.,'capacity_vph':7200.,'k_jam':400.,'v_cut':66.})
        params.to_csv(net/'links.csv',index=False); params.to_csv(net/'fd_parameters.csv',index=False)
        pd.DataFrame({'link_id':links,'incoming_link_ids':['','0','1','2'],'outgoing_link_ids':['1','2','3',''],
                      'on_ramp_link_ids':'','off_ramp_link_ids':''}).to_csv(net/'lwr_mainline_topology.csv',index=False)
        pd.DataFrame({'path_id':['p0','p1'],'origin_zone':['a','b'],'destination_zone':['z','z']}).to_csv(net/'path_set.csv',index=False)
        pd.DataFrame({'path_id':['p0']*4+['p1']*2,'link_id':links+links[2:]}).to_csv(net/'path_link_incidence.csv',index=False)
        for split,start,steps in [('train','2030-06-01',720),('validation','2031-03-01',72),('private','2031-04-01',72)]:
            times=pd.date_range(start,periods=steps,freq='5min',tz='UTC')
            rows=[]
            for t,time in enumerate(times):
                for i,link in enumerate(links):
                    speed=np.clip(86+24*np.sin(t/13-i*.5)+rng.normal(0,2),25,109)
                    flow=4500+500*np.cos(t/15-i*.5)+rng.normal(0,30)
                    rows.append({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ'),'date':time.strftime('%Y-%m-%d'),
                                 'link_id':link,'station_id':'s'+link,'speed_kmh':speed,'flow_vph':flow,
                                 'pct_observed':100.,'is_score_eligible':True,'mask_regime':'R2'})
            raw=pd.DataFrame(rows); masked=raw.copy(); hidden=rng.random(len(raw))<.3
            masked.loc[hidden,['speed_kmh','flow_vph']]=np.nan
            folder=root/'corridors'/panel/split
            maskdir=folder/'mainline_states_masked/mask_regime=R2'; maskdir.mkdir(parents=True,exist_ok=True)
            masked.to_parquet(maskdir/'states.parquet',index=False)
            if split=='train':
                plain=folder/'mainline_states'; plain.mkdir(parents=True,exist_ok=True); raw.to_parquet(plain/'states.parquet',index=False)
            target=raw.loc[hidden,['timestamp','station_id','link_id','mask_regime']].copy(); target.insert(0,'panel',panel)
            target['speed_kmh']=0.; target['flow_vph']=0.
            t1=root/'task1'/panel/split; t1.mkdir(parents=True,exist_ok=True); target.to_csv(t1/'sample_submission_state.csv',index=False)
            origin=times[20]; wid=panel+'_'+split
            hist=raw[pd.to_datetime(raw.timestamp,utc=True).between(times[8],origin)].copy(); hist['window_id']=wid
            t2=root/'task2'/panel/split; t2.mkdir(parents=True,exist_ok=True)
            hist.to_parquet(t2/'window_history.parquet',index=False)
            pd.DataFrame({'window_id':[wid],'panel':[panel],'split':[split],'condition':['queue_onset'],
                          'T':[origin.strftime('%Y-%m-%dT%H:%M:%SZ')]}).to_csv(t2/'window_index.csv',index=False)
            queue=pd.DataFrame([{'window_id':wid,'timestamp':(origin+pd.Timedelta(minutes=5*h)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                 'link_id':l,'queue_pred':0} for h in range(1,7) for l in links])
            queue.to_csv(t2/'sample_submission_queue.csv',index=False)
            t4=root/'task4'/panel/split; t4.mkdir(parents=True,exist_ok=True)
            od=pd.DataFrame({'panel':panel,'departure_time':'PUBLIC-'+split.upper()+'-PM','path_id':['p0','p1'],
                             'origin_zone':['a','b'],'destination_zone':'z','path_flow':[4000.,1000.]})
            od.to_csv(t4/'synthetic_weak_prior.csv',index=False); od.to_csv(t4/'sample_submission_path_flow.csv',index=False)
            pd.DataFrame({'link_id':links,'count':[4100.,4100.,5100.,5100.]}).to_csv(t4/'synthetic_link_counts.csv',index=False)
            if split!='train':
                for task,frame in [('state',target),('queue',queue),('odme',od)]:
                    for row in frame.to_dict('records'):
                        key.append({'submission_id':sid,'task':task,'split':split,**{k:v for k,v in row.items() if k not in ['speed_kmh','flow_vph','queue_pred','path_flow']}})
                        combined.append({'submission_id':sid,'task':task,'speed_kmh':0.,'flow_vph':0.,'queue_pred':0.,'path_flow':0.}); sid+=1
    pd.DataFrame(key).to_csv(root/'submission_key.csv',index=False)
    pd.DataFrame(combined).to_csv(root/'sample_submission.csv',index=False)
    return root

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('artifacts/fixture/release'))
    make(ap.parse_args().output)
