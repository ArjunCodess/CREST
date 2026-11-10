import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix
from src.data import split_masks,family,load_state,graph
from src.features import features,sampled_features,prediction_blocks
from src.physics import project_fd,differentiable_fd,conservation_residual,smooth_conservation
from src.models import GraphTemporal,tensor_input
from src.od import solve_od
from src.submission import merge,merge_files
from scripts.make_fixture import make

@pytest.fixture(scope='module')
def release(tmp_path_factory): return make(tmp_path_factory.mktemp('release'))

def test_purged_family_split():
    times=pd.date_range('2030-01-01',periods=720,freq='5min',tz='UTC')
    train,test,cut=split_masks(times)
    assert times[train].max()<cut-pd.Timedelta(minutes=90)
    assert times[test].min()>=cut+pd.Timedelta(minutes=90)
    assert family('D12_I5_N')==family('D12_I5_S')

def test_essential_download_keeps_all_pipeline_inputs():
    from scripts.download_data import include_file
    assert not include_file('kaggle_public/corridors/D12_I5_N/train/mainline_states_masked/mask_regime=R1/day.parquet',True)
    assert not include_file('kaggle_public/task1/D12_I5_N/train/sample_submission_state.csv',True)
    for name in ['corridors/D12_I5_N/train/mainline_states/day.parquet',
                 'corridors/D12_I5_N/private/mainline_states_masked/day.parquet',
                 'task1/D12_I5_N/private/sample_submission_state.csv',
                 'task2/D12_I5_N/train/window_history.parquet',
                 'task4/D12_I5_N/private/synthetic_weak_prior.csv','submission_key.csv','sample_submission.csv']:
        assert include_file('kaggle_public/'+name,True)

def test_forecast_features_are_causal(release):
    d=load_state(release,'D12_I5_N','train',True); up,down,lap,p=graph(release,'D12_I5_N',d.links)
    original=features(d.values,d.times,up,down,p)
    changed=d.values.copy(); changed[100:]=1e8
    after=features(changed,d.times,up,down,p)
    np.testing.assert_allclose(original[:100],after[:100],equal_nan=True)
    assert np.isfinite(lap).all()

def test_forecast_origin_mask_preserves_visible_history(release):
    d=load_state(release,'D12_I5_N','train',True); up,down,_,p=graph(release,'D12_I5_N',d.links)
    values=d.values[:30]; mask=np.random.default_rng(77).random(values.shape[:2])<.4
    augmented=features(values,d.times[:30],up,down,p,current_mask=mask)
    for t in [12,20,29]:
        history=values[:t+1].copy(); history[-1,mask[t]]=np.nan
        expected=features(history,d.times[:t+1],up,down,p)[-1]
        np.testing.assert_allclose(augmented[t],expected,equal_nan=True)

def test_block_features_match_full_history(release):
    d=load_state(release,'D12_I5_N','train',True); up,down,_,p=graph(release,'D12_I5_N',d.links)
    for reconstruct in [False,True]:
        full=features(d.values,d.times,up,down,p,reconstruct)
        blocks=np.concatenate([x for _,_,x in prediction_blocks(d.values,d.times,up,down,p,reconstruct)])
        np.testing.assert_allclose(blocks,full,equal_nan=True)
        sampled,indices=sampled_features(d.values,d.times,up,down,p,np.ones(d.values.shape[:2],dtype=bool),np.random.default_rng(1),100,reconstruct)
        np.testing.assert_allclose(sampled,full.reshape(-1,full.shape[-1])[indices],equal_nan=True)

def test_refinement_samples_stay_float32(release):
    from src.refinement import reconstruction_rows
    d=load_state(release,'D12_I5_N','train',True); up,down,_,p=graph(release,'D12_I5_N',d.links)
    values=d.values[:48].copy(); values[10]=np.nan
    all_rows,base=reconstruction_rows(values,d.times[:48],up,down,p)
    indices=np.array([40,41,60])
    sampled,sampled_base=reconstruction_rows(values,d.times[:48],up,down,p,indices)
    assert all_rows.dtype==np.float32 and sampled.dtype==np.float32
    np.testing.assert_allclose(sampled,all_rows.reshape(-1,all_rows.shape[-1])[indices],equal_nan=True)
    np.testing.assert_allclose(sampled_base,base.reshape(-1,2)[indices])

def test_queue_alignment_and_future_isolation(release):
    from src.pipeline import predict_queue
    from src.data import network
    class FakeModel:
        seen=None
        def forecast(self,values,times,*args):
            self.seen=(values.copy(),times.copy())
            return np.broadcast_to([80.,4000.],(6,len(args[-1]),2)).copy()
    panel='D12_I5_N'; d=load_state(release,panel,'validation'); up,down,_,params=graph(release,panel,d.links)
    model=FakeModel(); payload={'model':model,'links':d.links,'params':params,'up':up,'down':down}
    original=predict_queue(release,panel,'validation',payload,d.rows)
    expected=pd.to_datetime(original.timestamp,utc=True).min()-pd.Timedelta(minutes=5)
    assert model.seen[1][-1]==expected
    assert len(model.seen[1])==13
    changed=d.rows.copy(); changed.loc[changed.timestamp>expected,['speed_kmh','flow_vph']]=1e9
    result=predict_queue(release,panel,'validation',payload,changed)
    pd.testing.assert_frame_equal(original,result)
    # Flat 80 km/h stays above the fixture v_cut, so the onset rule marks the
    # two largest history-to-forecast drops and nothing else.
    assert int(original.queue_pred.sum())==2
    from src.queue_model import onset_slowdown
    slowdown=onset_slowdown(np.full((6,model.seen[0].shape[1]),80.),model.seen[0][...,0])
    ordered=pd.date_range(expected+pd.Timedelta(minutes=5),periods=6,freq='5min',tz='UTC')
    ti=pd.DatetimeIndex(ordered).get_indexer(pd.to_datetime(original.timestamp,utc=True))
    ni=pd.Index(d.links).get_indexer(original.link_id)
    np.testing.assert_array_equal(original.queue_pred.to_numpy(),slowdown[ti,ni].astype(int))
    class Classifier:
        seen_times=None
        def scores(self,values,times,*args):
            self.seen_times=times
            return np.full((6,values.shape[1]),.6)
        def classify(self,scores,condition):
            assert condition=='queue_onset'
            return scores>=.5
    classifier=Classifier(); payload['queue_classifier']=classifier
    classified=predict_queue(release,panel,'validation',payload,changed)
    assert classifier.seen_times[-1]==expected
    assert classified.queue_pred.eq(1).all()

def test_projection_and_gradient(release):
    d=load_state(release,'D12_I5_N','train',True); _,_,_,p=graph(release,'D12_I5_N',d.links)
    x=np.full((2,4,2),1e6); x[...,0]=10
    y=project_fd(x,p); k=y[...,1]/y[...,0]
    vf=p.free_speed_kmh.to_numpy(); cap=p.capacity_vph.to_numpy(); kj=p.k_jam.to_numpy(); w=cap/(kj-cap/vf)
    assert np.all(y[...,1]<=cap+1e-8)
    assert np.all(y[...,1]<=w*(kj-k)+1e-8)
    tx=torch.tensor(x,requires_grad=True)
    out=differentiable_fd(tx,*[torch.tensor(v) for v in [vf,cap,kj]])
    out.sum().backward(); assert torch.isfinite(tx.grad).all()
    np.testing.assert_allclose(out.detach(),y)

def test_od_does_not_make_missing_count_zero():
    A=csr_matrix([[1.,0.],[0.,1.],[1.,1.]])
    f,d=solve_od(A,np.array([100.,np.nan,np.nan]),np.array([100.,60.]))
    assert f[1]==pytest.approx(60,rel=1e-5)
    assert d['n_measured_links']==1

def test_dual_od_matches_primal_nnls():
    from scipy.optimize import nnls
    rng=np.random.default_rng(19); matrix=(rng.random((12,40))<.25).astype(float)
    prior=rng.uniform(0,100,size=40); counts=matrix@rng.uniform(0,150,size=40)
    expected,_=nnls(np.vstack([matrix,np.sqrt(.05)*np.eye(40)]),np.r_[counts,np.sqrt(.05)*prior])
    actual,_=solve_od(csr_matrix(matrix),counts,prior)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-3)

def test_od_matches_official_link_scorer(release):
    from src.od import build_od
    # src.data.network made the preserved official scorer packages importable.
    from task4.score_task4 import score_panel
    submission,diagnostics=build_od(release,'D12_I5_N','validation')
    official=score_panel('D12_I5_N',release,'validation',submission)
    assert official['S_link']==pytest.approx(diagnostics[0]['S_link'])
    assert official['invalid_rows']==0 and official['missing_paths']==0

def test_graph_transformer_arbitrary_size():
    m=GraphTemporal().eval()
    for n in [2,4]:
        x=np.ones((2,13,n,2)); a=torch.eye(n)
        state,future=m(tensor_input(x),a)
        assert state.shape==(2,n,2) and future.shape==(2,6,n,2)

@pytest.mark.parametrize('seed',[4,42])
def test_cached_neural_inference_matches_windows(seed):
    from src.models import neural_reconstruct
    from src.features import SCALE
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model=GraphTemporal().eval()
    rng=np.random.default_rng(4)
    values=rng.random((35,3,2))*SCALE; values[2,1]=np.nan
    a=np.eye(3,dtype=np.float32)
    padded=np.pad(values,((12,12),(0,0),(0,0)),constant_values=np.nan)
    windows=np.stack([padded[t:t+25] for t in range(len(values))])
    with torch.no_grad(): expected=model(tensor_input(windows),torch.tensor(a),center=12)[0].numpy()*SCALE
    actual=neural_reconstruct(model,values,a)
    # Different batching changes float32 reduction order. Compare normalized
    # channels so the tolerance does not depend on the flow's physical units.
    np.testing.assert_allclose(actual/SCALE,expected/SCALE,rtol=1e-5,atol=2e-7)

def test_strict_submission_missing_and_order(release,tmp_path):
    frames={}
    for task,filename in [('state','sample_submission_state.csv'),('queue','sample_submission_queue.csv'),('odme','sample_submission_path_flow.csv')]:
        number={'state':1,'queue':2,'odme':4}[task]
        frames[task]=pd.concat([pd.read_csv(p,dtype=str) for p in release.glob(f'task{number}/*/*/{filename}') if p.parent.name!='train'],ignore_index=True)
    output=tmp_path/'submission.csv'; n=merge(release,frames,output)
    expected=pd.read_csv(release/'sample_submission.csv'); got=pd.read_csv(output)
    assert n==len(expected)
    assert list(got)==list(expected)
    assert got.submission_id.equals(expected.submission_id)
    files={}
    for task,frame in frames.items():
        files[task]=tmp_path/f'{task}.csv'; frame.to_csv(files[task],index=False)
    merge_files(release,files,tmp_path/'disk_submission.csv')
    pd.testing.assert_frame_equal(got,pd.read_csv(tmp_path/'disk_submission.csv'))
    frames['state']=frames['state'].iloc[1:]
    with pytest.raises(ValueError,match='Missing state'): merge(release,frames,tmp_path/'bad.csv')
    assert not (tmp_path/'bad.csv').exists()

def test_derived_density_does_not_prove_conservation():
    states=np.array([[[50.,2000.]],[[50.,4000.]]])
    residual=conservation_residual(states,np.array([1.]),np.zeros((2,1)),np.zeros((2,1)),np.zeros((2,1)))
    assert residual[0,0]==pytest.approx(40)

def test_missing_observations_stay_fixed_in_mass_fit(release):
    d=load_state(release,'D12_I5_N','train',True); _,_,_,p=graph(release,'D12_I5_N',d.links)
    x=np.broadcast_to([90.,4000.],(4,4,2)).copy(); observed=x.copy(); observed[1,1]=np.nan
    incoming=np.zeros((4,4)); outgoing=np.zeros((4,4))
    result,report=smooth_conservation(x,observed,p,incoming,outgoing,np.zeros((4,4)),np.ones((4,4),dtype=bool),max_nfev=3)
    np.testing.assert_allclose(result[np.isfinite(observed)],observed[np.isfinite(observed)])
