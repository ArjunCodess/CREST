"""Leak-free queue comparison. Official queue labels stay hidden.

Both the classifier and the speed baseline are fit on the purged prefix only.
Thresholds are chosen on a middle February block. The following week is scored
once, with the forecast-origin observation masked, and is not used to fit or
to pick thresholds.
"""
import argparse
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import panels, load_state, graph, split_masks, family, read_csv
from src.queue_model import QueueClassifier, QueueSpeedRegressor
from src.evaluation import proxy_queue_iou
from src.holdout import selection_and_untouched

EXCLUDED = ('D12_I405_N', 'D12_I405_S')

def cutoffs(root, panel, links, params):
    cut = params.free_speed_kmh.to_numpy() * .6
    fd = read_csv(Path(root) / 'corridors' / panel / 'network' / 'fd_parameters.csv')
    if 'v_cut' in fd:
        released = fd.groupby('link_id').v_cut.mean().reindex(links).to_numpy()
        cut = np.where(np.isfinite(released), released, cut)
    return cut

def collect(values, times, active, cutoff, rng):
    windows = []
    counts = {'queue_onset': 0, 'queue_ongoing': 0}
    last = {'queue_onset': -1000, 'queue_ongoing': -1000}
    for t in range(12, len(values) - 6):
        if not active[t]:
            continue
        future = values[t + 1:t + 7, :, 0]
        valid = np.isfinite(future)
        truth = future <= cutoff
        if not np.any(truth & valid):
            continue
        visible = values[t - 12:t, :, 0]
        condition = 'queue_ongoing' if np.any(np.isfinite(visible) & (visible <= cutoff)) else 'queue_onset'
        if counts[condition] >= 5 or t - last[condition] < 72:
            continue
        history = values[t - 12:t + 1].copy()
        history[-1, rng.random(history.shape[1]) < float(rng.choice([.2, .3, .5]))] = np.nan
        windows.append({'t': t, 'condition': condition, 'truth': truth, 'valid': valid,
                        'history': history, 'times': times[t - 12:t + 1]})
        counts[condition] += 1
        last[condition] = t
    return windows, counts

def mean_iou(windows, predict):
    grouped = {}
    for window in windows:
        grouped.setdefault(window['condition'], []).append(
            proxy_queue_iou(window['truth'], predict(window), window['valid']))
    if not grouped:
        return float('nan')
    return float(np.mean([np.mean(scores) for scores in grouped.values()]))

def choose_thresholds(model, windows, up, down, params):
    """Pick onset and ongoing thresholds on the selection block only."""
    best = (-1.0, {'queue_onset': .5, 'queue_ongoing': .5}, False)
    for force in (False, True):
        thresholds = {}
        model.force_nonempty = force
        for condition in ('queue_onset', 'queue_ongoing'):
            group = [w for w in windows if w['condition'] == condition]
            if not group:
                thresholds[condition] = .5
                continue
            choice = []
            for threshold in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                model.thresholds[condition] = threshold
                scores = [proxy_queue_iou(
                    w['truth'],
                    model.classify(model.scores(w['history'], w['times'], up, down, params), condition),
                    w['valid']) for w in group]
                choice.append((float(np.mean(scores)), threshold))
            thresholds[condition] = max(choice)[1]
        model.thresholds = dict(thresholds)
        score = mean_iou(windows, lambda w: model.classify(
            model.scores(w['history'], w['times'], up, down, params), w['condition']))
        if score > best[0]:
            best = (score, dict(thresholds), force)
    model.thresholds = best[1]
    model.force_nonempty = best[2]
    return best

def attach(model, up, down, params):
    model._up, model._down, model._params = up, down, params
    return model

def emit(panel, links, times, windows, cutoff, classifier, regressor, submission, truth, index_rows):
    for number, window in enumerate(windows, start=1):
        wid = f"Q2_{panel}_untouched_{window['condition']}_{number:03d}"
        origin = pd.DatetimeIndex([times[window['t']]])[0]
        future_times = pd.DatetimeIndex(times[window['t'] + 1:window['t'] + 7])
        clf = classifier.classify(classifier.scores(window['history'], window['times'], classifier._up, classifier._down, classifier._params), window['condition'])
        reg = regressor.scores(window['history'], window['times'], classifier._up, classifier._down, classifier._params) <= cutoff
        speed = window['truth']  # boolean queue proxy; eligibility is window['valid']
        for step, stamp in enumerate(future_times):
            for link_i, link in enumerate(links):
                if not window['valid'][step, link_i]:
                    continue
                row = {'window_id': wid, 'timestamp': stamp, 'link_id': link}
                submission['classifier'].append({**row, 'queue_pred': int(clf[step, link_i])})
                submission['regressor'].append({**row, 'queue_pred': int(reg[step, link_i])})
                truth.append({**row, 'panel': panel, 'queue_true': bool(speed[step, link_i]),
                              'is_score_eligible': True})
        index_rows.append({'window_id': wid, 'panel': panel, 'family_id': family(panel), 'split': 'train',
                           'condition': window['condition'], 'forecast_origin': origin})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('artifacts/run'))
    ap.add_argument('--panel', action='append')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    previous = a.output / 'queue_proxy_cv.csv'
    results = pd.read_csv(previous).to_dict('records') if previous.exists() else []
    done = {r['panel'] for r in results if r.get('block') == 'untouched'}
    submission = {'classifier': [], 'regressor': []}
    truth, index_rows = [], []
    # Reload rows already emitted by earlier panels so a resume stays complete.
    for kind in ('classifier', 'regressor'):
        path = a.output / f'queue_untouched_{kind}.csv'
        if path.exists():
            submission[kind] = pd.read_csv(path).to_dict('records')
    truth_path = a.output / 'queue_untouched_truth.parquet'
    if truth_path.exists():
        truth = pd.read_parquet(truth_path).to_dict('records')
    index_path = a.output / 'queue_untouched_windows.csv'
    if index_path.exists():
        index_rows = pd.read_csv(index_path).to_dict('records')
    for panel in a.panel or panels(a.release_root):
        if panel in EXCLUDED:
            continue
        if panel in done:
            print('Keep completed queue comparison', panel, flush=True)
            continue
        print('Queue proxy training', panel, flush=True)
        d = load_state(a.release_root, panel, 'train', True)
        up, down, _, params = graph(a.release_root, panel, d.links)
        fit, _, _ = split_masks(d.times)
        selection, untouched = selection_and_untouched(d.times)
        if d.times[fit].max() >= d.times[selection].min():
            raise ValueError(f'{panel}: fit prefix overlaps the selection block')
        cutoff = cutoffs(a.release_root, panel, d.links, params)
        classifier = attach(QueueClassifier().fit(d.values[fit], d.times[fit], up, down, params), up, down, params)
        regressor = QueueSpeedRegressor().fit(d.values[fit], d.times[fit], up, down, params)
        rng = np.random.default_rng(2026)
        selection_windows, selection_counts = collect(d.values, d.times, selection, cutoff, rng)
        untouched_windows, untouched_counts = collect(d.values, d.times, untouched, cutoff, rng)
        if not selection_windows or not untouched_windows:
            print('  Missing windows on one block; keep the speed baseline', flush=True)
            row = {'panel': panel, 'block': 'untouched', 'accepted': False,
                   'selection_windows': len(selection_windows), 'untouched_windows': len(untouched_windows)}
            results = [r for r in results if r['panel'] != panel] + [row]
            pd.DataFrame(results).to_csv(previous, index=False)
            continue
        chosen = choose_thresholds(classifier, selection_windows, up, down, params)
        classifier.thresholds, classifier.force_nonempty = chosen[1], chosen[2]
        def clf_pred(window):
            return classifier.classify(classifier.scores(window['history'], window['times'], up, down, params), window['condition'])
        def reg_pred(window):
            return regressor.scores(window['history'], window['times'], up, down, params) <= cutoff
        selection_clf, selection_reg = mean_iou(selection_windows, clf_pred), mean_iou(selection_windows, reg_pred)
        untouched_clf, untouched_reg = mean_iou(untouched_windows, clf_pred), mean_iou(untouched_windows, reg_pred)
        accepted = bool(untouched_clf > untouched_reg + .02)
        rows = []
        for block, clf_score, reg_score, counts, n in (
                ('selection', selection_clf, selection_reg, selection_counts, len(selection_windows)),
                ('untouched', untouched_clf, untouched_reg, untouched_counts, len(untouched_windows))):
            rows.append({'panel': panel, 'block': block, 'windows': n,
                         'onset_windows': counts['queue_onset'], 'ongoing_windows': counts['queue_ongoing'],
                         'regressor_proxy_IoU': reg_score, 'classifier_proxy_IoU': clf_score,
                         'accepted': accepted, 'thresholds': str(chosen[1]), 'force_nonempty': chosen[2]})
        print(f'  untouched IoU classifier={untouched_clf:.4f}, regressor={untouched_reg:.4f}, accepted={accepted}', flush=True)
        prefix = f'Q2_{panel}_'
        submission['classifier'] = [r for r in submission['classifier'] if not str(r['window_id']).startswith(prefix)]
        submission['regressor'] = [r for r in submission['regressor'] if not str(r['window_id']).startswith(prefix)]
        truth[:] = [r for r in truth if not str(r['window_id']).startswith(prefix)]
        index_rows[:] = [r for r in index_rows if r.get('panel') != panel]
        emit(panel, d.links, d.times, untouched_windows, cutoff, classifier, regressor, submission, truth, index_rows)
        results = [r for r in results if r['panel'] != panel] + rows
        pd.DataFrame(results).to_csv(previous, index=False)
        pd.DataFrame(submission['classifier']).to_csv(a.output / 'queue_untouched_classifier.csv', index=False)
        pd.DataFrame(submission['regressor']).to_csv(a.output / 'queue_untouched_regressor.csv', index=False)
        pd.DataFrame(truth).to_parquet(truth_path, index=False)
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        if accepted:
            deployed = QueueClassifier().fit(d.values, d.times, up, down, params)
            deployed.thresholds, deployed.force_nonempty = chosen[1], chosen[2]
            temporary = a.output / f'{panel}_queue.partial.pkl'
            with temporary.open('wb') as f:
                pickle.dump(deployed, f)
            temporary.replace(a.output / f'{panel}_queue.pkl')
    print('Queue comparisons saved', flush=True)

if __name__ == '__main__':
    main()
