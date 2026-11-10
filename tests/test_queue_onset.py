import numpy as np
import pandas as pd
from src.queue_onset import onset_events, decode_onset


def test_published_onset_contract_keeps_only_final_step_and_whole_block():
    values=np.ones((200,3,2))*[100.,3000.]
    values[50:70,:2,0]=30
    values[130:145,1,0]=35
    times=pd.date_range('2030-01-01',periods=200,freq='5min',tz='UTC')
    active=np.ones(200,bool); active[:40]=False
    result=onset_events(values,np.ones((200,3,3)),times,active,np.full(3,60.))
    assert len(result)==1 and result[0].origin==124
    assert result[0].truth[-1].sum()==1 and not result[0].truth[:-1].any()
    assert result[0].first_queue_step==6


def test_onset_decoding_never_spends_predictions_on_known_empty_steps():
    p=np.ones((6,4))*.9
    p[-1]=[.95,.8,.01,.01]
    output=decode_onset(p)
    assert not output[:-1].any()
    np.testing.assert_array_equal(output[-1],[True,True,False,False])
    assert decode_onset(np.zeros((6,4))).sum()==1
