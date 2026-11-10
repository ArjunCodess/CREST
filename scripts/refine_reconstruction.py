"""Compare a residual imputer on identical masks, then fit only winning panels.

The acceptance trial stays on the last seven days. A separate earlier-February
week, which that trial never sees, is scored from the same prefix-only fit.
"""
import argparse
import gc
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import panels, load_state, graph, split_masks
from src.features import artificial_mask
from src.holdout import fit_trees, score_masked_block, selection_and_untouched
from src.refinement import ResidualReconstructor, interpolation
from src.physics import project_fd
from src.evaluation import state_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('artifacts/run'))
    ap.add_argument('--panel', action='append')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    previous = a.output / 'refinement_cv.csv'
    results = pd.read_csv(previous).to_dict('records') if previous.exists() else []
    untouched_path = a.output / 'untouched_state.csv'
    untouched_rows = pd.read_csv(untouched_path).to_dict('records') if untouched_path.exists() else []
    done_untouched = {r['panel'] for r in untouched_rows}
    for panel in a.panel or panels(a.release_root):
        completed = [r for r in results if r['panel'] == panel]
        refined_done = bool(completed) and (not completed[0]['accepted'] or (a.output / f'{panel}_refined.pkl').exists())
        if refined_done and panel in done_untouched:
            print('Keep completed refinement', panel, flush=True)
            continue
        print('Refinement', panel, flush=True)
        data = load_state(a.release_root, panel, 'train', True)
        up, down, _, params = graph(a.release_root, panel, data.links)
        fit, valid, _ = split_masks(data.times)
        valid &= np.asarray(data.times >= data.times[-1] - pd.Timedelta(days=7))
        _, untouched = selection_and_untouched(data.times)
        if data.times[fit].max() >= data.times[untouched].min():
            raise ValueError(f'{panel}: fit prefix overlaps the untouched block')
        model = ResidualReconstructor().fit(data.values[fit], data.times[fit], up, down, params)
        if not refined_done:
            rng = np.random.default_rng(2026)
            trial = []
            for regime, rate in [('R1', .2), ('R2', .3), ('R3', .5)]:
                truth = data.values[valid]
                masked, target = artificial_mask(truth, rng, rate)
                forecasts = [
                    ('interpolation', interpolation(masked)),
                    ('residual', model.reconstruct(masked, data.times[valid], up, down, params)),
                ]
                for name, pred in forecasts:
                    for projection, state in [('raw', pred), ('fd', project_fd(pred, params))]:
                        trial.append({'panel': panel, 'regime': regime, 'model': name, 'projection': projection,
                                      **state_metrics(truth, state, params.lanes, target)})
            baseline = pd.read_csv(a.output / 'temporal_cv.csv')
            baseline_score = float(baseline[(baseline.panel == panel) & baseline.selected].S_state.mean())
            candidate = float(np.mean([r['S_state'] for r in trial if r['model'] == 'residual' and r['projection'] == 'fd']))
            accepted = candidate > baseline_score + .001
            print(f'  residual={candidate:.6f}, previous={baseline_score:.6f}, accepted={accepted}', flush=True)
            for r in trial:
                r['accepted'] = accepted
            results = [r for r in results if r['panel'] != panel] + trial
            pd.DataFrame(results).to_csv(a.output / 'refinement_cv.csv', index=False)
        else:
            accepted = bool(completed[0]['accepted'])
            print(f'  refinement already recorded, accepted={accepted}', flush=True)
        if panel not in done_untouched:
            trees = fit_trees(data.values[fit], data.times[fit], up, down, params, rounds=80)
            rows, stamps, tree, resid = score_masked_block(
                panel, data.values, data.times, data.links, up, down, params, model, trees, untouched)
            for r in rows:
                r['accepted_residual'] = accepted
            untouched_rows = [r for r in untouched_rows if r['panel'] != panel] + rows
            pd.DataFrame(untouched_rows).to_csv(untouched_path, index=False)
            done_untouched.add(panel)
            folder = a.output / 'untouched'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(
                folder / f'{panel}.npz', times=stamps, links=np.asarray(data.links),
                tree=tree.astype(np.float32), residual=resid.astype(np.float32),
                accepted=np.array([accepted]))
            del trees, tree, resid
            print(f'  untouched block scored, accepted_residual={accepted}', flush=True)
        gc.collect()
        if accepted and not (a.output / f'{panel}_refined.pkl').exists():
            model = ResidualReconstructor().fit(data.values, data.times, up, down, params)
            temporary = a.output / f'{panel}_refined.partial.pkl'
            with temporary.open('wb') as f:
                pickle.dump(model, f)
            temporary.replace(a.output / f'{panel}_refined.pkl')
        del data, model
        gc.collect()
    print('Refinement comparisons saved', flush=True)

if __name__ == '__main__':
    main()
