import numpy as np
import pandas as pd
from src.conservation import public_flux_inputs


def test_public_flux_reads_requested_time_span_and_keeps_missing_ramps_invalid(tmp_path):
    folder=tmp_path/'corridors'/'panel'
    (folder/'network').mkdir(parents=True)
    (folder/'train/ramp_states').mkdir(parents=True)
    pd.DataFrame({'link_id':['L1'], 'incoming_link_ids':['L1'], 'outgoing_link_ids':['L1'],
        'on_ramp_link_ids':['R1'], 'off_ramp_link_ids':['']}).to_csv(folder/'network/lwr_mainline_topology.csv',index=False)
    pd.DataFrame({'timestamp':['2030-09-01T00:00:00Z','2030-09-01T00:05:00Z','2030-12-01T00:00:00Z'],
        'ramp_link_id':['R1']*3,'pct_observed':[100,50,100], 'flow_vph':[100.,900.,10000.],
        'is_score_eligible':[True,False,True], 'unused_large_field':['ignored']*3}).to_parquet(folder/'train/ramp_states/ramps.parquet')
    times=pd.date_range('2030-09-01',periods=2,freq='5min',tz='UTC')
    incoming,outgoing,net,valid=public_flux_inputs(tmp_path,'panel','train',times,['L1'])
    np.testing.assert_array_equal(incoming,[[1.]])
    np.testing.assert_array_equal(outgoing,[[1.]])
    np.testing.assert_array_equal(net[:,0],[100.,0.])
    np.testing.assert_array_equal(valid[:,0],[True,False])
