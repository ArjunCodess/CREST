import numpy as np
import pandas as pd
from src.fd_calibration import soft_fd,fd_score
from src.physics import project_fd


def test_soft_fd_zero_weight_preserves_baseline_and_full_weight_matches_diagram():
    params=pd.DataFrame({'free_speed_kmh':[110.,115.],'capacity_vph':[6000.,7000.],
        'k_jam':[300.,400.],'lanes':[3.,4.]})
    states=np.array([[[100.,4000.],[50.,3500.]],[[60.,2500.],[90.,5000.]]])
    np.testing.assert_allclose(soft_fd(states,params,0),project_fd(states,params))
    projected=soft_fd(states,params,1)
    assert fd_score(projected,params,np.ones(states.shape[:2],bool))>1-1e-10
    assert np.isfinite(projected).all() and (projected>=0).all()
