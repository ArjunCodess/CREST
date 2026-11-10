import numpy as np
import pandas as pd
import pytest
from src.queue_events import QueueEvent, visible_event
from src.queue_v5 import localized_features, LocalizedQueueModel


def test_localized_features_cannot_see_hidden_origin():
    times = pd.date_range('2030-01-01', periods=13, freq='5min', tz='UTC')
    h = np.ones((13,4,2))*[100,3000]
    a = np.ones((13,4,3))
    e = QueueEvent(12,'queue_ongoing',h,a,times,np.zeros((6,4),bool),np.ones((6,4),bool),1)
    params = pd.DataFrame({'length_km':[1.]*4,'lanes':[3.]*4,
        'free_speed_kmh':[100.]*4,'capacity_vph':[6000.]*4,'k_jam':[300.]*4})
    up = np.eye(4,k=-1)
    x = localized_features(*visible_event(e,'history_only'),times,up,up.T,params)
    e.history[-1] = -1e9; e.auxiliary[-1] = 1e9
    y = localized_features(*visible_event(e,'history_only'),times,up,up.T,params)
    np.testing.assert_allclose(x,y,equal_nan=True)
    with pytest.raises(ValueError):
        localized_features(h[:-1],a[:-1],times[:-1],up,up.T,params)


def test_localized_onset_scores_obey_published_horizon():
    times = pd.date_range('2030-01-01',periods=13,freq='5min',tz='UTC')
    params = pd.DataFrame({'length_km':[1.]*4,'lanes':[3.]*4,
        'free_speed_kmh':[100.]*4,'capacity_vph':[6000.]*4,'k_jam':[300.]*4})
    model = LocalizedQueueModel()
    model.models = {'queue_onset':.8,'queue_ongoing':.8}
    args=(np.ones((13,4,2))*[100,3000],np.ones((13,4,3)),times)
    up=np.eye(4,k=-1)
    onset=model.scores(*args,'queue_onset',up,up.T,params)
    assert not onset[:-1].any()
    np.testing.assert_allclose(onset[-1],.8)
    np.testing.assert_allclose(model.scores(*args,'queue_ongoing',up,up.T,params),.8)
