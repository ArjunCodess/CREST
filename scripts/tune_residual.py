"""Compare higher-capacity residual reconstruction on purged training blocks."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_state, graph, split_masks
from src.holdout import selection_and_untouched
from src.features import artificial_mask
from src.refinement import ResidualReconstructor
from src.physics import project_fd
from src.evaluation import state_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = load_state(a.release_root, a.panel, 'train', True)
    up, down, _, params = graph(a.release_root, a.panel, d.links)
    fit, _, _ = split_masks(d.times)
    selection, untouched = selection_and_untouched(d.times)
    if d.times[fit].max() >= d.times[selection].min():
        raise ValueError('Training overlaps selection')
    results = []
    models = {}
    for label, rounds, sample in [('residual_220', 220, 120000), ('residual_600', 600, 300000)]:
        start = time.monotonic()
        print(f'Fit {a.panel} {label}', flush=True)
        model = ResidualReconstructor(rounds=rounds, max_rows=sample).fit(d.values[fit], d.times[fit], up, down, params)
        models[label] = model
        print(f'Fit took {time.monotonic()-start:.1f}s', flush=True)
        rng = np.random.default_rng(8128)
        for block, mask in [('selection', selection), ('assessment', untouched)]:
            for regime, rate in [('R1', .2), ('R2', .3), ('R3', .5)]:
                truth = d.values[mask]
                masked, target = artificial_mask(truth, rng, rate)
                pred = project_fd(model.reconstruct(masked, d.times[mask], up, down, params), params)
                result = {'panel': a.panel, 'model': label, 'block': block, 'regime': regime,
                    **state_metrics(truth, pred, params.lanes, target)}
                results.append(result)
                print(json.dumps(result), flush=True)
        pd.DataFrame(results).to_csv(a.output / f'{a.panel}_capacity_cv.csv', index=False)
    cv = pd.DataFrame(results)
    scores = cv.groupby(['block', 'model']).S_state.mean().unstack()
    accepted = bool((scores.residual_600 > scores.residual_220 + .001).all())
    print(scores.to_string(), flush=True)
    (a.output / f'{a.panel}_capacity_decision.json').write_text(json.dumps({'accepted': accepted,
        'minimum_gain_each_block': .001, 'scores': scores.to_dict(),
        'note': 'Assessment block has been examined in prior experiments; model comparison, not a fresh independent test.'}, indent=2))
    if accepted:
        del models, model
        gc.collect()
        print(f'Refit full train {a.panel}', flush=True)
        model = ResidualReconstructor(rounds=600, max_rows=300000).fit(d.values, d.times, up, down, params)
        with (a.output / f'{a.panel}_refined.pkl').open('wb') as f: pickle.dump(model, f)


if __name__ == '__main__':
    main()
