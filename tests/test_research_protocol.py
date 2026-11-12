import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

from src.research.config_paths import bundled_config_path, load_raw_config
from src.research.config_validation import (
    ConfigValidationError,
    FINAL_HOLDOUT_V3_ARCHIVE_SEEDS,
    FINAL_HOLDOUT_V4_ACTIVE_SEEDS,
    RETIRED_EXPOSED_V2_SEEDS,
    canonical_protocol_content,
    validate_and_normalize_protocol,
)
from src.research.protocol import (
    assert_final_allowed,
    assert_output_writable,
    bootstrap_registry,
    config_fingerprint,
    record_run_start,
    registry_file_lock,
    write_final_receipt,
    _stale_registry_lock,
)

TEST_FINAL_SEEDS = frozenset({9901, 9902, 9903, 9904, 9905})


def _final_cfg_protocol(seeds: list[int] | None = None) -> dict:
    return {
        'protocol_id': 'test_final_fixture',
        'protocol_role': 'final',
        'seeds': sorted(seeds or TEST_FINAL_SEEDS),
        'scenario_params': {
            'n_links': 6,
            'n_steps': 10,
            'demand_vph': 820.0,
            'noise_std': 18.0,
            'missing_rate': 0.18,
            'capacity_vehicles': 200.0,
        },
        'stress_variants': ['baseline'],
    }


@pytest.fixture
def active_final_holdout(monkeypatch):
    monkeypatch.setattr(
        'src.research.config_validation.FINAL_HOLDOUT_ACTIVE_SEEDS',
        TEST_FINAL_SEEDS,
    )


def test_final_receipt_blocks_repeat(tmp_path, active_final_holdout):
    receipt = tmp_path / 'final_receipt.json'
    registry = tmp_path / 'registry.json'
    cfg = validate_and_normalize_protocol(_final_cfg_protocol(), cli_mode='final')
    write_final_receipt(receipt, cfg, {'ok': True}, code_hash='abc', registry_path=registry)
    with pytest.raises(RuntimeError):
        assert_final_allowed(registry, receipt, cfg, output_dir=tmp_path / 'out')


def test_override_allows_rerun(tmp_path, active_final_holdout):
    receipt = tmp_path / 'final_receipt.json'
    registry = tmp_path / 'registry.json'
    cfg = validate_and_normalize_protocol(_final_cfg_protocol(), cli_mode='final')
    write_final_receipt(receipt, cfg, {'ok': True}, code_hash='abc', registry_path=registry)
    assert_final_allowed(
        registry, receipt, cfg, output_dir=tmp_path / 'out', override=True, override_reason='documented rerun'
    )
    write_final_receipt(
        receipt,
        cfg,
        {'ok': True},
        code_hash='abc',
        registry_path=registry,
        override=True,
        override_reason='documented rerun',
    )
    assert json.loads(receipt.read_text())['override'] is True


def test_fingerprint_ignores_config_path():
    base = {
        'protocol_id': 'dev_pilot',
        'protocol_role': 'development',
        'seeds': [11, 22, 33, 44, 55],
        'scenario_params': {
            'n_links': 5,
            'n_steps': 8,
            'demand_vph': 750.0,
            'noise_std': 12.0,
            'missing_rate': 0.15,
            'capacity_vehicles': 180.0,
        },
        'stress_variants': ['baseline'],
    }
    a = config_fingerprint({**base, '_config_path': '/a/dev.json'})
    b = config_fingerprint({**base, '_config_path': 'C:\\b\\dev.json'})
    assert a == b
    c = canonical_protocol_content(base)
    assert '_config_path' not in c


def test_role_mode_mismatch_rejected():
    raw, _ = load_raw_config(None, default_bundled='dev_pilot.json')
    with pytest.raises(ConfigValidationError):
        validate_and_normalize_protocol(raw, cli_mode='final')


def test_v1_seeds_rejected_in_final_config(active_final_holdout):
    with pytest.raises(ConfigValidationError):
        validate_and_normalize_protocol(
            {
                'protocol_id': 'final_bad',
                'protocol_role': 'final',
                'seeds': [101, 202, 303, 404, 505],
                'scenario_params': _final_cfg_protocol()['scenario_params'],
                'stress_variants': ['baseline'],
            },
            cli_mode='final',
        )


def test_retired_v2_seeds_rejected_in_dev():
    with pytest.raises(ConfigValidationError):
        validate_and_normalize_protocol(
            {
                'protocol_id': 'dev_bad',
                'protocol_role': 'development',
                'seeds': sorted(RETIRED_EXPOSED_V2_SEEDS),
                'scenario_params': _final_cfg_protocol()['scenario_params'],
                'stress_variants': ['baseline'],
            },
            cli_mode='development',
        )


def test_on_disk_registry_v3_retired_failed_not_active():
    registry = Path(__file__).resolve().parents[1] / '.crest_research_protocol_registry.json'
    raw = json.loads(registry.read_text(encoding='utf-8'))
    v3 = raw['protocols']['final_holdout_v3']
    assert v3['status'] == 'retired_failed'
    assert v3.get('exposure_record', {}).get('protocol_id') == 'final_holdout_v3'
    failed = [r for r in raw.get('runs', []) if r.get('protocol_id') == 'final_holdout_v3']
    assert len(failed) == 1
    assert failed[0]['status'] == 'failed'
    assert failed[0]['code_hash'] == '07a399f8e9550149'


def test_final_v3_on_disk_archived_not_active():
    """On-disk v3 config remains frozen archive; final CLI must use v4."""
    v3_path = Path(__file__).resolve().parents[1] / 'configs' / 'research' / 'final_protocol.json'
    raw = json.loads(v3_path.read_text(encoding='utf-8'))
    assert raw['protocol_id'] == 'final_holdout_v3'
    assert raw.get('holdout_status') == 'retired_failed'
    assert set(raw['seeds']) == set(FINAL_HOLDOUT_V3_ARCHIVE_SEEDS)
    with pytest.raises(ConfigValidationError, match='must not use archived v3'):
        validate_and_normalize_protocol(raw, cli_mode='final')


def test_bundled_final_v4_matches_active_reserve_metadata_only():
    """Bundled default final config is amended v4; unrelated fixture seeds still used in other tests."""
    raw, _ = load_raw_config(None, default_bundled='final_protocol.json')
    assert raw['protocol_id'] == 'final_holdout_v4'
    assert set(raw['seeds']) == set(FINAL_HOLDOUT_V4_ACTIVE_SEEDS)
    cfg = validate_and_normalize_protocol(raw, cli_mode='final')
    assert cfg['protocol_id'] == 'final_holdout_v4'


def test_registry_v4_single_attempt_cap():
    registry = Path(__file__).resolve().parents[1] / '.crest_research_protocol_registry.json'
    raw = json.loads(registry.read_text(encoding='utf-8'))
    v4_runs = [r for r in raw.get('runs', []) if r.get('protocol_id') == 'final_holdout_v4']
    assert len(v4_runs) <= 1


def test_final_fixture_repeat_rejection_unrelated_protocol(tmp_path, active_final_holdout):
    """Repeat guard uses an unrelated mini fixture (not production v3 seeds)."""
    receipt = tmp_path / 'final_receipt.json'
    registry = tmp_path / 'registry.json'
    mini = {
        'protocol_id': 'wheel_smoke_final_fixture',
        'protocol_role': 'final',
        'seeds': [9901, 9902, 9903, 9904, 9905],
        'scenario_params': _final_cfg_protocol()['scenario_params'],
        'stress_variants': ['baseline'],
    }
    cfg = validate_and_normalize_protocol(mini, cli_mode='final')
    write_final_receipt(receipt, cfg, {'ok': True}, code_hash='abc', registry_path=registry)
    with pytest.raises(RuntimeError, match='receipt already exists|blocks repeat'):
        assert_final_allowed(registry, receipt, cfg, output_dir=tmp_path / 'out')


def test_in_progress_blocks_second_final(tmp_path, active_final_holdout):
    registry = tmp_path / 'registry.json'
    receipt = tmp_path / 'final_receipt.json'
    cfg = validate_and_normalize_protocol(_final_cfg_protocol(), cli_mode='final')
    fp = config_fingerprint(cfg)
    record_run_start(
        registry,
        protocol_id=cfg['protocol_id'],
        fingerprint=fp,
        code_hash='x',
        output_dir=tmp_path / 'out_a',
        mode='final',
    )
    with pytest.raises(RuntimeError):
        assert_final_allowed(registry, receipt, cfg, output_dir=tmp_path / 'out_b')


def test_registry_lock_serializes_starts(tmp_path, active_final_holdout):
    registry = tmp_path / 'registry.json'
    cfg = validate_and_normalize_protocol(_final_cfg_protocol(), cli_mode='final')
    fp = config_fingerprint(cfg)
    with registry_file_lock(registry):
        record_run_start(
            registry,
            protocol_id=cfg['protocol_id'],
            fingerprint=fp,
            code_hash='a',
            output_dir=tmp_path / 'out1',
            mode='final',
        )
        with pytest.raises(RuntimeError):
            record_run_start(
                registry,
                protocol_id=cfg['protocol_id'],
                fingerprint=fp,
                code_hash='b',
                output_dir=tmp_path / 'out2',
                mode='final',
            )


def test_dev_output_requires_force(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'pilot_results.json').write_text('{}')
    with pytest.raises(RuntimeError):
        assert_output_writable(out, force=False, role='development')


def test_bundled_config_exists():
    path = bundled_config_path('dev_pilot.json')
    assert path.exists()


def test_corrupt_lock_not_treated_stale(tmp_path):
    lock = tmp_path / 'registry.lock'
    lock.write_text('not-a-pid', encoding='utf-8')
    assert _stale_registry_lock(lock) is False


def test_dead_pid_lock_is_stale(tmp_path):
    import subprocess

    from src.research.protocol import _lock_owner_process_liveness

    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    pid = proc.pid
    proc.wait(timeout=30)
    assert _lock_owner_process_liveness(pid) == 'dead'
    lock = tmp_path / 'registry.lock'
    lock.write_text(str(pid), encoding='utf-8')
    assert _stale_registry_lock(lock) is True


def test_unknown_pid_lock_not_stale(tmp_path):
    lock = tmp_path / 'registry.lock'
    lock.write_text('4000000000', encoding='utf-8')
    assert _stale_registry_lock(lock) is False


@pytest.mark.parametrize('pid', [2**31, 4000000000, 10**100])
def test_posix_pid_overflow_preserves_lock(tmp_path, monkeypatch, pid):
    from src.research import protocol

    def overflow_kill(owner: int, signal: int) -> None:
        assert (owner, signal) == (pid, 0)
        raise OverflowError('signed integer is greater than maximum')

    monkeypatch.setattr(os, 'kill', overflow_kill)
    monkeypatch.setattr(protocol, '_lock_owner_process_liveness', protocol._posix_process_liveness)
    lock = tmp_path / 'registry.lock'
    lock.write_text(str(pid), encoding='utf-8')

    assert _stale_registry_lock(lock) is False
    assert lock.read_text(encoding='utf-8') == str(pid)
    with pytest.raises(TimeoutError):
        with registry_file_lock(tmp_path / 'registry.json', timeout_s=0):
            pytest.fail('An unprobeable owner must not lose its lock')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only stale-lock probe guard')
def test_stale_lock_windows_never_calls_os_kill(tmp_path, monkeypatch):
    calls: list[tuple[int, int]] = []

    def forbid_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        raise AssertionError('os.kill must not be used for registry lock staleness on Windows')

    monkeypatch.setattr(os, 'kill', forbid_kill)
    import subprocess

    from src.research.protocol import _lock_owner_process_liveness

    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    pid = proc.pid
    proc.wait(timeout=30)
    assert _lock_owner_process_liveness(pid) == 'dead'
    lock = tmp_path / 'registry.lock'
    lock.write_text(str(pid), encoding='utf-8')
    assert _stale_registry_lock(lock) is True
    assert calls == []


def _registry_lock_hold(registry: Path, hold_s: float, flag: Path) -> None:
    with registry_file_lock(registry, timeout_s=10.0):
        flag.write_text('ok', encoding='utf-8')
        time.sleep(hold_s)


def test_registry_lock_handoff_two_workers_both_complete(tmp_path):
    """Two cooperative workers serialize on the lock; neither is evicted mid-hold."""
    registry = tmp_path / 'registry.json'
    bootstrap_registry(registry)
    lock_path = registry.with_suffix('.lock')
    flag_a = tmp_path / 'a.done'
    flag_b = tmp_path / 'b.done'
    ctx = mp.get_context('spawn')
    p1 = ctx.Process(target=_registry_lock_hold, args=(registry, 0.12, flag_a))
    p2 = ctx.Process(target=_registry_lock_hold, args=(registry, 0.04, flag_b))
    p1.start()
    time.sleep(0.02)
    p2.start()
    p1.join(15)
    p2.join(15)
    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert flag_a.read_text(encoding='utf-8') == 'ok'
    assert flag_b.read_text(encoding='utf-8') == 'ok'
    assert not lock_path.exists()
