import numpy as np
import pandas as pd
from src.state_v4 import CrossSensorFactor


def test_cross_sensor_factor_preserves_observations_and_ignores_removed_values():
    rng=np.random.default_rng(82)
    t=np.arange(160)
    values=np.zeros((160,6,2),dtype=np.float32)
    values[...,0]=100+np.sin(t[:,None]/9)*np.arange(1,7)[None]
    values[...,1]=3000+np.cos(t[:,None]/13)*np.arange(1,7)[None]*100
    params=pd.DataFrame({'free_speed_kmh':[110.]*6,'capacity_vph':[7000.]*6})
    model=CrossSensorFactor(rank=4).fit(values,params)
    mask=rng.random(values.shape[:2])<.3
    masked=values.copy();masked[mask]=np.nan
    output=model.reconstruct(masked)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[~mask],values[~mask])
    # The factor is fitted on a separate source, and this call consumes only its masked input.
    np.testing.assert_array_equal(model.reconstruct(masked),output)
    assert np.sqrt(np.mean((output[...,1][mask]-values[...,1][mask])**2))<150
