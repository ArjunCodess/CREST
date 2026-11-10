"""Release-only readers and purged temporal/family split contracts."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

CHANNELS = ['speed_kmh', 'flow_vph']

def read_csv(path):
    return pd.read_csv(path, dtype={k: str for k in ['panel', 'link_id', 'station_id',
        'path_id', 'origin_zone', 'destination_zone', 'departure_time', 'window_id']})

def panels(root):
    root = Path(root)
    if not (root / 'corridors').is_dir():
        raise FileNotFoundError(f'{root} is not a release root; expected corridors/, task1/, task2/, task4/. Download the public Kaggle release first.')
    return sorted(p.name for p in (root / 'corridors').iterdir() if (p / 'network/links.csv').exists())

def family(panel):
    return panel.rsplit('_', 1)[0]

def split_masks(times, fraction=.8, embargo_steps=18):
    """Purge a full 60-min lookback plus 30-min target around the cutoff."""
    t = pd.DatetimeIndex(times)
    if len(t) < 2:
        raise ValueError('At least two timestamps required')
    cutoff = t[min(int(len(t) * fraction), len(t)-1)]
    gap = pd.Timedelta(minutes=5 * embargo_steps)
    return np.asarray(t < cutoff-gap), np.asarray(t >= cutoff+gap), cutoff

def split_manifest(root, output):
    ps = panels(root)
    folds = [{'held_out_family': f, 'fit_panels': [p for p in ps if family(p) != f],
              'test_panels': [p for p in ps if family(p) == f]} for f in sorted({family(p) for p in ps})]
    result = {'spatial_folds': folds, 'temporal': {'fit_fraction': .8, 'embargo_steps_each_side': 18},
              'policy': 'Fit preprocessing and models on fit rows only. Never fit on validation/private or withheld families.'}
    Path(output).write_text(json.dumps(result, indent=2))
    return result

def network(root, panel):
    """Use the evaluator's calibrated FD aggregation rather than OSM defaults."""
    import sys
    official = Path(__file__).resolve().parents[1] / 'starter/src'
    if str(official) not in sys.path:
        sys.path.insert(0, str(official))
    from task3.score_task3 import network_parameters
    return network_parameters(Path(root) / 'corridors' / panel)

@dataclass
class State:
    times: pd.DatetimeIndex
    links: list
    values: np.ndarray
    rows: pd.DataFrame

def load_state(root, panel, split, truth=False, start=None, end=None):
    if truth and split != 'train':
        raise ValueError('Only train truth is public')
    folder = Path(root) / 'corridors' / panel / split / ('mainline_states' if truth else 'mainline_states_masked')
    paths = sorted(folder.rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError(folder)
    links=sorted(read_csv(Path(root)/'corridors'/panel/'network/links.csv').link_id.unique())
    parts=[]
    for path in paths:
        import pyarrow.parquet as pq
        available=pq.ParquetFile(path).schema_arrow.names
        needed=['timestamp','link_id','speed_kmh','flow_vph','pct_observed']
        x = pd.read_parquet(path,columns=[c for c in needed if c in available])
        x['timestamp'] = pd.to_datetime(x.timestamp, utc=True)
        if start is not None: x = x[x.timestamp >= pd.Timestamp(start)]
        if end is not None: x = x[x.timestamp < pd.Timestamp(end)]
        if len(x):
            if 'pct_observed' in x: x.loc[x.pct_observed<75,CHANNELS]=np.nan
            x['link_id']=x.link_id.astype(str)
            if not set(x.link_id)<=set(links): raise ValueError('Observation link absent from released network')
            day_times=pd.DatetimeIndex(sorted(x.timestamp.unique()))
            arrays=[x.pivot_table(index='timestamp',columns='link_id',values=c,aggfunc='mean')
                    .reindex(index=day_times,columns=links).to_numpy(dtype=np.float32) for c in CHANNELS]
            parts.append((day_times,np.stack(arrays,-1)))
    if not parts: raise ValueError(f'No state records in requested date range: {folder}')
    times=pd.date_range(min(t.min() for t,_ in parts),max(t.max() for t,_ in parts),freq='5min')
    values=np.full((len(times),len(links),2),np.nan,dtype=np.float32)
    for day_times,day_values in parts: values[times.get_indexer(day_times)]=day_values
    rows=pd.DataFrame()
    if not truth:
        rows=pd.DataFrame({'timestamp':times.repeat(len(links)),
                           'link_id':pd.Categorical(np.tile(links,len(times)),categories=links),
                           'speed_kmh':values[...,0].ravel(),'flow_vph':values[...,1].ravel()})
    return State(times,links,values,rows)

def graph(root, panel, link_ids):
    path = Path(root) / 'corridors' / panel / 'network/lwr_mainline_topology.csv'
    n = len(link_ids); index = {s: i for i, s in enumerate(link_ids)}
    adjacency = np.zeros((n, n), dtype=np.float32)
    topo = read_csv(path)
    for row in topo.itertuples():
        dest = index.get(str(row.link_id))
        if dest is None: continue
        for source in str(getattr(row, 'incoming_link_ids', '')).split(';'):
            if source in index: adjacency[index[source], dest] = 1
        for dest_id in str(getattr(row, 'outgoing_link_ids', '')).split(';'):
            if dest_id in index: adjacency[dest, index[dest_id]] = 1
    params = network(root, panel).set_index('link_id').reindex(link_ids)
    if params[['length_km','free_speed_kmh','capacity_vph','k_jam']].isna().any().any():
        raise ValueError('Missing calibrated network parameters for observed links')
    travel = params.length_km.to_numpy() / params.free_speed_kmh.to_numpy() * 60
    weighted = adjacency * np.exp(-travel[:, None]/5)
    def normalize(a): return a / np.maximum(a.sum(1, keepdims=True), 1)
    up = normalize(weighted.T); down = normalize(weighted)
    sym = weighted + weighted.T
    degree = np.maximum(sym.sum(1), 1e-8)
    laplacian = np.eye(n) - sym / np.sqrt(degree[:, None]*degree[None, :])
    return up, down, laplacian.astype(np.float32), params
