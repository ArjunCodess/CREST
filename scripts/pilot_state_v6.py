"""Training-only state/physics ablation. Never modifies a production checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_state, graph
from src.conservation import public_flux_inputs
from src.evaluation import state_metrics
from src.refinement import interpolation
from src.physics import project_fd
from src.state_v6 import RampAwareReconstructor, mass_diagnostic, uncertainty_projection
from src.research_diagnostics import noise_nugget, missing_run_lengths
from src.holdout import official_cell_mask, _regimes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True)
    ap.add_argument('--rounds', type=int, default=220)
    ap.add_argument('--max-rows', type=int, default=120000)
    ap.add_argument('--days', type=int, default=7)
    ap.add_argument('--fit-start', default='2030-06-01T00:00:00Z')
    ap.add_argument('--mask', choices=['published_hash', 'iid'], default='published_hash')
    ap.add_argument('--blocks', nargs='+', default=['selection', 'november', 'january'],
                    choices=['selection', 'november', 'january'])
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists():
        raise ValueError('Use a fresh output directory to preserve the registered experiment')
    protocol = {'fit_start': args.fit_start, 'fit_end': '2030-09-30T22:30:00Z',
        'selection': 'October 15 onward', 'development_only': ['november', 'january'],
        'sealed_assessment': 'No month is certified untouched across prior experiments; no clean-test claim.',
        'projection_weights': [0., .05, .2], 'rounds': args.rounds, 'max_rows': args.max_rows,
        'mask': args.mask,
        'promotion_gate': 'Selection S_state gain >= .003, replicated on both development blocks; no regime loses > .005. Not an official score.',
        'limitations': 'Public connector and ramp coverage may leave no valid mass equations. Raw observations are a diagnostic reference, not a physics ceiling.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    all_metrics, mass_rows, noise_rows, gap_rows = [], [], [], []
    for panel in args.panel:
        train = load_state(args.release_root, panel, 'train', True,
                           start=args.fit_start, end='2030-09-30T22:30:00Z')
        up, down, _, params = graph(args.release_root, panel, train.links)
        stable = train.values[..., 0] > .8*params.free_speed_kmh.to_numpy()
        nuggets = {}
        for channel, variable in enumerate(('speed_kmh', 'flow_per_lane')):
            values = train.values[..., channel]
            if channel == 1:
                values = values/params.lanes.to_numpy()
            nuggets[variable] = noise_nugget(values, stable)
            noise_rows.append({'panel': panel, 'variable': variable, **nuggets[variable]})
        (args.output/'noise_diagnostics.json').write_text(json.dumps(noise_rows, indent=2))
        flux = public_flux_inputs(args.release_root, panel, 'train', train.times, train.links)
        models = {}
        # Matched architecture and masks isolate weighting from added features.
        with threadpool_limits(limits=4):
            for name, weighted, ramps in [('unweighted_fd_features', False, False),
                                           ('weighted_fd_features', True, False),
                                           ('weighted_ramp_features', True, True)]:
                print(panel, 'fit', name, flush=True)
                models[name] = RampAwareReconstructor(args.rounds, args.max_rows, ramps, weighted).fit(
                    train.values, train.times, up, down, params, flux)
        report = {'panel': panel, 'n_fit_cells': len(train.times)*len(train.links),
                  'known_flux_fraction': float(np.mean(flux[-1])), 'training_rmse': models['weighted_ramp_features'].training_rmse}
        # Persist small fitted estimators before evaluation so failed runs retain
        # their provenance and never require a silent production refit.
        with (args.output/f'{panel}_prefix_models.pkl').open('wb') as f:
            pickle.dump({'models': models, 'protocol': protocol, 'links': train.links}, f)
        fit_links = train.links
        del train, flux, stable, values
        gc.collect()
        baseline_file = Path(f'artifacts/state_v4/pilot/{panel}_candidate.pkl')
        with baseline_file.open('rb') as f:
            baseline = pickle.load(f)
        for block, date in [('selection', '2030-10-15'), ('november', '2030-11-15'), ('january', '2031-01-15')]:
            if block not in args.blocks:
                continue
            begin = pd.Timestamp(date+'T00:00:00Z')
            data = load_state(args.release_root, panel, 'train', True,
                              start=begin, end=begin+pd.Timedelta(days=args.days))
            block_flux = public_flux_inputs(args.release_root, panel, 'train', data.times, data.links)
            if data.links != fit_links:
                raise ValueError('Network ordering changed')
            rng = np.random.default_rng(664)
            days, inverse = np.unique(np.asarray(data.times.normalize()), return_inverse=True)
            # Cycle regimes before shuffling so even a short pilot covers all.
            daily = np.resize([.2, .3, .5], len(days)); rng.shuffle(daily)
            rates = daily[inverse]
            if args.mask == 'published_hash':
                rates = np.array([{'R1': .2, 'R2': .3, 'R3': .5}[r] for r in _regimes(panel, data.times)])
                mask = official_cell_mask(panel, data.times, data.links)
            else:
                mask = rng.random(data.values.shape[:2]) < rates[:, None]
            mask &= np.isfinite(data.values).all(-1)
            masked = data.values.copy(); masked[mask] = np.nan
            gaps = missing_run_lengths(~np.isfinite(masked).all(-1))
            incoming, outgoing, net, known = block_flux
            complete = np.isfinite(data.values).all(-1)
            complete &= np.isfinite(data.values[..., 1]).astype(int) @ ((incoming+outgoing) != 0).T == ((incoming+outgoing) != 0).sum(1)
            complete[:-1] &= np.isfinite(data.values[1:]).all(-1)
            report_flux = (incoming, outgoing, net, known & complete)
            predictions = {'interpolation': interpolation(masked)}
            with threadpool_limits(limits=4):
                predictions['v4_prefix'] = project_fd(baseline.reconstruct(masked, data.times, up, down, params), params)
                for name, model in models.items():
                    predictions[name] = project_fd(model.reconstruct(masked, data.times, up, down, params, block_flux), params)
            # Observations remain fixed in projection and in all mass diagnostics.
            # FD projection here matches the deployed v4 postprocessing for state.
            raw = predictions['weighted_ramp_features']
            for weight in protocol['projection_weights'][1:]:
                pred, diagnostic = uncertainty_projection(raw, masked, params, block_flux,
                    models['weighted_ramp_features'].error_scale, weight=weight)
                if diagnostic.get('success', False):
                    predictions[f'projection_{weight}'] = pred
                mass_rows.append({'panel': panel, 'block': block, 'model': f'projection_{weight}', **diagnostic})
            for name, pred in {'observed_reference': data.values, **predictions}.items():
                frame = np.where(np.isfinite(masked), masked, pred) if name != 'observed_reference' else pred
                mass_rows.append({'panel': panel, 'block': block, 'model': name, **mass_diagnostic(frame, params, report_flux)})
                if name == 'observed_reference':
                    continue
                for regime in (.2, .3, .5):
                    selected = mask & (rates[:, None] == regime)
                    if selected.any():
                        all_metrics.append({'panel': panel, 'block': block, 'model': name,
                            'regime': regime, **state_metrics(data.values, pred, params.lanes, selected)})
                        for gap_name, gap_select in [('one_step', gaps == 1), ('two_three', (gaps >= 2) & (gaps <= 3)), ('four_plus', gaps >= 4)]:
                            cells = selected & gap_select
                            if not cells.any():
                                continue
                            metrics = state_metrics(data.values, pred, params.lanes, cells)
                            for variable, key in [('speed_kmh', 'rmse_speed_kmh'), ('flow_per_lane', 'rmse_flow_vph_per_lane')]:
                                sigma = nuggets[variable]['sigma_linear']
                                gap_rows.append({'panel': panel, 'block': block, 'model': name,
                                    'regime': regime, 'gap': gap_name, 'variable': variable,
                                    'rmse': metrics[key], 'n_cells': int(cells.sum()), 'sigma_assumed': sigma,
                                    'rmse_over_sigma': metrics[key]/sigma if sigma is not None and sigma > 0 else None})
                    else:
                        all_metrics.append({'panel': panel, 'block': block, 'model': name,
                            'regime': regime, 'rmse_speed_kmh': None, 'rmse_flow_vph_per_lane': None,
                            'S_state': 0., 'n_targets': 0})
            pd.DataFrame(all_metrics).to_csv(args.output/'state_metrics.csv', index=False)
            pd.DataFrame(gap_rows).to_csv(args.output/'state_noise_gap_metrics.csv', index=False)
            (args.output/'mass_diagnostics.json').write_text(json.dumps(mass_rows, indent=2))
            print(panel, block, 'evaluated', flush=True)
        (args.output/f'{panel}_fit.json').write_text(json.dumps(report, indent=2))
        del models, baseline; gc.collect()
    frame = pd.DataFrame(all_metrics)
    scores = frame.groupby(['panel', 'block', 'model']).S_state.mean().reset_index()
    scores.to_csv(args.output/'macro_regime_scores.csv', index=False)
    decisions = []
    for panel in args.panel:
        local = scores[scores.panel.eq(panel)].pivot(index='block', columns='model', values='S_state')
        choices = [c for c in local if c not in ('interpolation', 'v4_prefix')]
        best = local.loc['selection', choices].idxmax()
        delta = local[best]-local.v4_prefix
        regime_scores = frame[frame.panel.eq(panel)].pivot(index=['block', 'regime'], columns='model', values='S_state')
        worst = (regime_scores[best]-regime_scores.v4_prefix).min()
        replicated = all(b in delta.index and delta[b] >= .003 for b in ('november', 'january'))
        decisions.append({'panel': panel, 'selected': best, 'development_gain': delta.to_dict(),
            'worst_regime_delta': float(worst), 'passes_development_gate': bool(delta['selection'] >= .003 and replicated and worst >= -.005),
            'production_promoted': False, 'reason': 'Experimental proxy evaluation only; no sealed final assessment or submission replacement.'})
    (args.output/'decisions.json').write_text(json.dumps(decisions, indent=2))


if __name__ == '__main__':
    main()
