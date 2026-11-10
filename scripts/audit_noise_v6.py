"""Train-only noise sensitivity by traffic level and observation quality.

Short-lag variogram intercepts depend on the assumed latent dynamics. They are
not identified measurement-noise variances or observed bounds on model error.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import graph, read_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True)
    ap.add_argument('--start', default='2030-09-01T00:00:00Z')
    ap.add_argument('--end', default='2030-09-08T00:00:00Z')
    args = ap.parse_args()
    if pd.Timestamp(args.end) > pd.Timestamp('2030-10-01T00:00:00Z'):
        raise ValueError('Noise fitting must precede October selection')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use an empty output directory')
    protocol = {'start': args.start, 'end': args.end, 'panels': args.panel,
        'split': 'train', 'selection': 'No later records, no fitted model selection.',
        'assumptions': 'Zero-mean white measurement noise, independent of sufficiently smooth latent dynamics. Linear-lag and quadratic-lag fits are alternatives, not identified truth.',
        'conditioning_caveat': 'Both endpoints must lie in the same bin. Conditioning on observed traffic levels can bias noise estimates. Within-file pairs only.',
        'variables': ['speed_kmh', 'flow_per_lane'], 'lags_steps': [1, 2, 3]}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []; sources = []
    for panel in args.panel:
        folder = args.release_root/'corridors'/panel
        links = sorted(read_csv(folder/'network/links.csv').link_id.unique())
        _, _, _, params = graph(args.release_root, panel, links)
        stats = {}
        for path in sorted((folder/'train/mainline_states').rglob('*.parquet')):
            frame = pd.read_parquet(path, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'pct_observed'],
                filters=[('timestamp', '>=', args.start), ('timestamp', '<', args.end)])
            if frame.empty:
                continue
            frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
            frame['link_id'] = frame.link_id.astype(str)
            times = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq='5min')
            arrays = [frame.pivot_table(index='timestamp', columns='link_id', values=c)
                .reindex(index=times, columns=links).to_numpy() for c in ('speed_kmh', 'flow_vph', 'pct_observed')]
            speed, flow, quality = arrays
            eligible = quality >= 75
            ratio = speed/params.free_speed_kmh.to_numpy()
            load = flow/params.capacity_vph.to_numpy()
            groups = {'all_eligible': eligible,
                'speed_below_0.6vf': eligible & (ratio <= .6),
                'speed_0.6_to_0.8vf': eligible & (ratio > .6) & (ratio < .8),
                'speed_at_least_0.8vf': eligible & (ratio >= .8),
                'flow_below_0.4capacity': eligible & (load < .4),
                'flow_0.4_to_0.8capacity': eligible & (load >= .4) & (load < .8),
                'flow_at_least_0.8capacity': eligible & (load >= .8),
                'quality_75_to_90': (quality >= 75) & (quality < 90),
                'quality_90_to_100': (quality >= 90) & (quality < 100),
                'quality_100': quality == 100}
            sources.append({'panel': panel, 'file': str(path), 'n_rows': len(frame), 'bytes': path.stat().st_size})
            for variable, x in [('speed_kmh', speed), ('flow_per_lane', flow/params.lanes.to_numpy())]:
                for group, mask in groups.items():
                    mask = mask & np.isfinite(x)
                    for lag in (1, 2, 3):
                        use = mask[lag:] & mask[:-lag]
                        difference = (x[lag:]-x[:-lag])[use]
                        key = (variable, group, lag)
                        old = stats.setdefault(key, np.zeros(3))
                        old += [len(difference), difference.sum(), difference@difference]
        for variable in ('speed_kmh', 'flow_per_lane'):
            for group in sorted({k[1] for k in stats}):
                semivariance = []; counts = []
                for lag in (1, 2, 3):
                    n, total, squared = stats[(variable, group, lag)]
                    counts.append(int(n))
                    semivariance.append(max(0, (squared/n-(total/n)**2)/2) if n >= 20 else None)
                for power, dynamics in [(1, 'diffusive_linear_lag'), (2, 'smooth_quadratic_lag')]:
                    if any(v is None for v in semivariance):
                        intercept = None
                    else:
                        design = np.stack([np.ones(3), np.arange(1, 4, dtype=float)**power], -1)
                        intercept = float(np.linalg.lstsq(design, semivariance, rcond=None)[0][0])
                    rows.append({'panel': panel, 'variable': variable, 'group': group, 'latent_dynamics': dynamics,
                        'sigma_assumed': np.sqrt(max(intercept, 0)) if intercept is not None else None,
                        'unclipped_variance_intercept': intercept, 'negative_intercept': intercept is not None and intercept < 0,
                        **{f'gamma_lag{i+1}': v for i, v in enumerate(semivariance)},
                        **{f'pairs_lag{i+1}': n for i, n in enumerate(counts)}})
        pd.DataFrame(rows).to_csv(args.output/'noise_by_level_quality.csv', index=False)
        (args.output/'sources.json').write_text(json.dumps(sources, indent=2))
        print(panel, 'noise diagnostic complete', flush=True)


if __name__ == '__main__':
    main()
