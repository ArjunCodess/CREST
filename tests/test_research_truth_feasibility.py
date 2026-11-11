import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.research.experiments import run_one_scenario
from src.research.partial_identification import aggregate_bounds, observability_diagnostics
from src.research.protocol import bootstrap_registry, record_run_finish, record_run_start
from src.research.synthetic_sim import generate_scenario
from src.research.truth_feasibility import evaluate_truth_feasibility


DEV_SEEDS = [11, 22, 33, 44, 55]


@pytest.mark.parametrize('seed', DEV_SEEDS)
def test_baseline_uniform_truth_joint_feasible_all_dev_seeds(seed: int) -> None:
    s = generate_scenario(seed=seed, obs_noise_model='uniform_bounded', obs_noise_scale=12.0)
    ev = evaluate_truth_feasibility(s)
    assert ev['flags']['joint_within_assumption']
    r = run_one_scenario('development_baseline', seed, {'obs_noise_model': 'uniform_bounded'})
    assert r['eval_truth_feasibility']['joint_within_assumption']
    assert r['failure_flags']['joint_within_assumption']
    assert r['metrics']['bounded_status'] == 'optimal'
    assert r['failure_flags']['od_lp_status'] == 'optimal'


def test_bad_observation_stress_preserves_generative_physics() -> None:
    s = generate_scenario(seed=8801, truth_outside_interval=True, obs_noise_model='uniform_bounded')
    resid = s.storage_true[1:] - s.storage_true[:-1] - s.dt_hours * (
        s.z_true @ s.B.T + s.u_true @ s.R.T
    )
    assert np.max(np.abs(resid)) < 1e-9
    assert not s.boundary_truth_in_interval()


def test_gaussian_misspecification_can_exclude_truth_from_tubes() -> None:
    rng = np.random.default_rng(0)
    scale = 12.0
    tube = scale * 2.5
    failures = 0
    for seed in range(200, 260):
        s = generate_scenario(seed=seed, obs_noise_model='gaussian_misspecification', noise_std=scale, missing_rate=0.0)
        ev = evaluate_truth_feasibility(s)
        if not ev['flags']['observation_tubes']:
            failures += 1
    assert failures >= 5
    assert tube > scale


def test_rank_deficient_od_feasible_with_wide_bounds() -> None:
    A = np.array([[1.0, 1.0, 0.0]])
    caps = np.array([50.0, 50.0, 50.0])
    diag = observability_diagnostics(A, np.array([True]))
    assert diag.nullity == 2
    bounds = aggregate_bounds(
        A,
        np.array([20.0]),
        np.array([80.0]),
        np.array([1.0, 1.0, 1.0]),
        flow_upper=caps,
    )
    assert bounds['status'] == 'optimal'
    assert bounds['upper'] - bounds['lower'] > 10.0


def test_inconsistent_od_counts_stress() -> None:
    s = generate_scenario(seed=7701, inconsistent_od_counts=True)
    ev = evaluate_truth_feasibility(s)
    assert not ev['flags']['od_counts']
    r = run_one_scenario('development_inconsistent_od_counts', 7701, {'inconsistent_od_counts': True})
    assert r['failure_flags']['od_lp_status_reason'] == 'infeasible_deliberate_count_stress'


def test_registry_concurrent_subprocess_race(tmp_path: Path) -> None:
    registry = tmp_path / 'registry.json'
    bootstrap_registry(registry)
    ok_a = tmp_path / 'ok_a'
    err_a = tmp_path / 'err_a'
    ok_b = tmp_path / 'ok_b'
    err_b = tmp_path / 'err_b'
    crest = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        '''
        import sys
        from pathlib import Path
        from src.research.protocol import record_run_start
        reg, out, okp, errp = map(Path, sys.argv[1:5])
        try:
            record_run_start(
                reg,
                protocol_id='artificial_race_fixture',
                fingerprint='racefp',
                code_hash='worker',
                output_dir=out,
                mode='final',
            )
            okp.write_text('ok', encoding='utf-8')
        except RuntimeError:
            errp.write_text('rejected', encoding='utf-8')
        '''
    )
    worker = tmp_path / 'race_worker.py'
    worker.write_text(script, encoding='utf-8')
    env = {**dict(**__import__('os').environ), 'PYTHONPATH': str(crest)}
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), str(registry), str(tmp_path / 'out_a'), str(ok_a), str(err_a)],
            cwd=str(crest),
            env=env,
        ),
        subprocess.Popen(
            [sys.executable, str(worker), str(registry), str(tmp_path / 'out_b'), str(ok_b), str(err_b)],
            cwd=str(crest),
            env=env,
        ),
    ]
    codes = [p.wait(timeout=60) for p in procs]
    assert codes == [0, 0]
    winners = int(ok_a.exists()) + int(ok_b.exists())
    rejections = int(err_a.exists()) + int(err_b.exists())
    assert winners == 1
    assert rejections == 1
    runs = bootstrap_registry(registry).get('runs', [])
    assert len(runs) == 1
    assert runs[0]['status'] == 'in_progress'
    record_run_finish(
        registry,
        protocol_id='artificial_race_fixture',
        fingerprint='racefp',
        status='failed',
        message='race test cleanup',
    )
    failed = [r for r in bootstrap_registry(registry).get('runs', []) if r.get('status') == 'failed']
    assert failed
