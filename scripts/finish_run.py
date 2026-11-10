"""Finish the official run sequentially; each stage has its own durable log."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def find_compile_latex():
    bundled = Path.home() / '.codex' / 'plugins' / 'cache'
    if bundled.is_dir():
        for path in bundled.glob('**/scripts/compile_latex.py'):
            if path.is_file():
                return path
    return None


def list_panels(release_root):
    corridors = release_root / 'corridors'
    if not corridors.is_dir():
        return []
    return sorted(
        p.name for p in corridors.iterdir()
        if (p / 'network/links.csv').exists()
    )


def missing_experiment_checkpoints(release_root, output):
    missing_refined, missing_queue = [], []
    for panel in list_panels(release_root):
        if not (output / f'{panel}_refined.pkl').exists():
            missing_refined.append(panel)
        if not (output / f'{panel}_queue.pkl').exists():
            missing_queue.append(panel)
    return missing_refined, missing_queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--skip-experiments', action='store_true')
    ap.add_argument('--reuse-predictions', action='store_true', help='Use an already completed upload and task files')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'completed': []}
    status = output / 'completion_status.json'

    def stage(name, command):
        state['active_stage'] = name
        status.write_text(json.dumps(state, indent=2))
        print('Start', name, flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state['failed_stage'] = name
            state['returncode'] = result.returncode
            status.write_text(json.dumps(state, indent=2))
            raise RuntimeError(f'{name} failed: see {output / (name + ".log")}')
        state['completed'].append(name)
        status.write_text(json.dumps(state, indent=2))

    release = str(args.release_root.resolve())
    common = ['--release-root', release, '--output', str(output)]
    if (output / 'submission.csv').exists() and not (output / 'submission_initial.csv').exists():
        shutil.copy2(output / 'submission.csv', output / 'submission_initial.csv')
    release_path = args.release_root.resolve()
    if args.skip_experiments:
        missing_refined, missing_queue = missing_experiment_checkpoints(release_path, output)
        state['skipped_experiments'] = True
        state['missing_refined_pkl'] = missing_refined
        state['missing_queue_pkl'] = missing_queue
        status.write_text(json.dumps(state, indent=2))
    else:
        stage('experiments', [sys.executable, '-u', 'scripts/run_experiments.py', *common])
    if args.reuse_predictions:
        if not args.skip_experiments:
            raise ValueError('Reusing predictions requires skipping model changes')
        if not (output / 'submission.csv').exists():
            raise FileNotFoundError('No completed upload to reuse')
    else:
        stage('final_prediction', [sys.executable, '-u', '-m', 'src.pipeline', 'predict', *common, '--split', 'both'])
    for split in ['validation', 'private']:
        stage(f'official_link_score_{split}', [sys.executable, '-u', 'starter/src/task4/score_task4.py',
              '--submission', str(output / 'odme.csv'), '--release-root', release, '--split', split,
              '--output', str(output / f'official_link_score_{split}.csv')])
    stage('spatial_validation', [sys.executable, '-u', '-m', 'src.pipeline', 'spatial-cv', *common, '--rounds', '80'])
    stage('experiment_report', [sys.executable, '-u', '-m', 'src.report', '--run', str(output), '--output', str(output / 'report')])
    tex = root / 'paper/working_notes.tex'
    compiler = find_compile_latex()
    latexmk = shutil.which('latexmk')
    if compiler:
        compile_command = [sys.executable, str(compiler), str(tex), '--json']
        stage('paper_compile', compile_command)
    elif latexmk:
        stage('paper_compile', [latexmk, '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-cd', str(tex)])
    else:
        state['active_stage'] = 'paper_compile'
        status.write_text(json.dumps(state, indent=2))
        print('Skip paper_compile: no compile_latex.py bundle and latexmk not on PATH', flush=True)
        state['completed'].append('paper_compile')
        state['paper_compile'] = {'status': 'skipped', 'reason': 'no bundled compile_latex.py and latexmk not on PATH'}
        status.write_text(json.dumps(state, indent=2))
    stage('tests', [sys.executable, '-m', 'pytest', '-q'])
    stage('final_audit', [sys.executable, '-u', 'scripts/audit_run.py', *common])
    with (output / 'submission.csv').open('rb') as submission:
        digest = hashlib.file_digest(submission, 'sha256').hexdigest()
    state.update(active_stage=None, finished_utc=datetime.now(timezone.utc).isoformat(), submission_sha256=digest)
    status.write_text(json.dumps(state, indent=2))
    print('Completed submission, public scoring, validation, experiment report, manuscript build and tests', flush=True)


if __name__ == '__main__':
    main()
