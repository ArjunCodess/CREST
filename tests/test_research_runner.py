from pathlib import Path

import pytest

from src.research.config_paths import REPO_ROOT, load_raw_config
from src.research.config_validation import RETIRED_EXPOSED_V2_SEEDS, validate_and_normalize_protocol
from src.research.experiments import run_experiment_matrix


MINI_DEV_PROTOCOL = {
    'protocol_id': 'test_fixture_dev',
    'protocol_role': 'development',
    'seeds': [8801, 8802, 8803, 8804, 8805],
    'scenario_params': {
        'n_links': 5,
        'n_steps': 6,
        'demand_vph': 700.0,
        'noise_std': 10.0,
        'missing_rate': 0.1,
        'capacity_vehicles': 200.0,
    },
    'stress_variants': ['baseline'],
}


def test_dev_config_loads():
    raw, _ = load_raw_config(REPO_ROOT / 'configs' / 'research' / 'dev_pilot.json', default_bundled='dev_pilot.json')
    cfg = validate_and_normalize_protocol(raw, cli_mode='development', final_reserved_seeds=RETIRED_EXPOSED_V2_SEEDS)
    assert len(cfg['seeds']) >= 5
    assert cfg['protocol_role'] == 'development'


def test_experiment_matrix_runs_mini_fixture():
    cfg = validate_and_normalize_protocol(
        MINI_DEV_PROTOCOL,
        cli_mode='development',
        development_seed_allowlist=frozenset(MINI_DEV_PROTOCOL['seeds']),
        final_reserved_seeds=RETIRED_EXPOSED_V2_SEEDS,
    )
    cfg['scenario_set'] = 'test'
    payload = run_experiment_matrix(cfg, REPO_ROOT)
    assert len(payload['rows']) == 5
    assert payload['rows'][0]['metrics']['oracle_state_rmse_all'] < 1e-6


def test_summarize_excludes_failed_point_estimates():
    from src.research.experiments import summarize_by_scenario, summarize_matrix

    rows = [
        {
            'scenario_key': 'development_baseline',
            'metrics': {
                'bounded_status': 'optimal',
                'bounded_primal_violation': 0.0,
                'bounded_state_rmse_all': 10.0,
                'oracle_state_rmse_all': 0.0,
            },
        },
        {
            'scenario_key': 'development_baseline',
            'metrics': {
                'bounded_status': 'numerical_failure',
                'bounded_primal_violation': 0.5,
                'bounded_state_rmse_all': None,
                'oracle_state_rmse_all': 5.0,
            },
        },
    ]
    m = summarize_matrix(rows)
    assert m['n_point_estimate_valid'] == 1
    assert m['n_point_estimate_failed'] == 1
    s = summarize_by_scenario(rows)[0]
    assert s['bounded_state_rmse_all_mean_conditional'] == 10.0
    assert 'bounded_state_rmse_all_mean' not in s


def test_runner_bundled_defaults(tmp_path, monkeypatch):
    from src.research import runner

    monkeypatch.chdir(tmp_path)
    rc = runner.main(['--mode', 'development', '--output', str(tmp_path / 'out'), '--force'])
    assert rc == 0
    assert (tmp_path / 'out' / 'pilot_results.json').exists()
