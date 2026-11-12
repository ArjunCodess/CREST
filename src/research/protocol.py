"""Development vs final protocol guards, registry ledger, and receipts."""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config_validation import (
    FINAL_HOLDOUT_ACTIVE_SEEDS,
    FINAL_HOLDOUT_V3_ARCHIVE_SEEDS,
    FINAL_HOLDOUT_V4_ACTIVE_SEEDS,
    RETIRED_EXPOSED_V2_SEEDS,
    V2_EXPOSURE_RECORD,
    V3_EXPOSURE_RECORD,
    canonical_protocol_content,
)
from .provenance import research_code_content_hash  # noqa: F401 — re-exported for runner/tests

LEDGER_VERSION = 1
METADATA_KEYS = frozenset({'_config_path', '_config_resolved', '_source_path'})

# Retired after independent review exposure (development-mode probe of v1 holdout).
FINAL_HOLDOUT_V1 = {
    'protocol_id': 'final_holdout_v1',
    'seeds': [101, 202, 303, 404, 505],
    'status': 'retired_exposed',
    'exposure_note': (
        'Reviewer executed original final_protocol.json seeds 101-505 in development mode '
        '(40 runs). Holdout no longer pristine; v1 retired 2026-11-11.'
    ),
}

def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def config_fingerprint(config: dict) -> str:
    import hashlib

    content = canonical_protocol_content(config)
    return hashlib.sha256(_canonical_json(content).encode()).hexdigest()[:16]


def default_registry_path() -> Path:
    env = os.environ.get('CREST_RESEARCH_REGISTRY')
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve() / '.crest_research_protocol_registry.json'


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, allow_nan=False)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as f:
        return json.loads(f.read())


def bootstrap_registry(registry_path: Path) -> dict:
    reg = load_json(registry_path)
    if reg:
        return reg
    return {
        'ledger_version': LEDGER_VERSION,
        'protocols': {
            FINAL_HOLDOUT_V1['protocol_id']: {
                'status': FINAL_HOLDOUT_V1['status'],
                'seeds': FINAL_HOLDOUT_V1['seeds'],
                'exposure_note': FINAL_HOLDOUT_V1['exposure_note'],
            },
            'final_holdout_v2': {
                'status': V2_EXPOSURE_RECORD['status'],
                'seeds': V2_EXPOSURE_RECORD['retired_seeds'],
                'exposure_record': V2_EXPOSURE_RECORD,
            },
            'final_holdout_v3': {
                'status': 'retired_failed',
                'seeds': sorted(FINAL_HOLDOUT_V3_ARCHIVE_SEEDS),
                'note': (
                    'Archived v3 reserve (failed generative attempt); manifest '
                    'configs/research/final_holdout_v3_seed_manifest.json — not rerun.'
                ),
                'exposure_record': V3_EXPOSURE_RECORD,
            },
            'final_holdout_v4': {
                'status': 'active' if FINAL_HOLDOUT_ACTIVE_SEEDS else 'inactive',
                'seeds': sorted(FINAL_HOLDOUT_V4_ACTIVE_SEEDS),
                'manifest': 'configs/research/final_holdout_v4_seed_manifest.json',
                'amendment_note': 'Transparent amended reserve after generative repair; v3 failure preserved.',
            },
            'dev_pilot': {
                'status': 'development',
                'seeds': [11, 22, 33, 44, 55],
            },
        },
        'runs': [],
    }


def load_receipt(path: Path) -> dict | None:
    return load_json(path)


REGISTRY_SCOPE_NOTE = (
    'Local project registry JSON is a development guard against accidental repeat final runs '
    'in the same workspace; it is not a security boundary and does not coordinate untrusted parties.'
)

_LOG = logging.getLogger(__name__)

# Windows: CPython os.kill maps signal 0 to TerminateProcess (not a portable existence probe).
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000


def _windows_process_liveness(pid: int) -> str:
    """Read-only liveness probe. Returns alive | dead | unknown (fail-closed for lock removal)."""
    if pid <= 0:
        return 'unknown'
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    access = _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
    handle = kernel32.OpenProcess(access, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        if err == 87:  # ERROR_INVALID_PARAMETER — no such process
            return 'dead'
        return 'unknown'
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return 'unknown'
        if exit_code.value == _STILL_ACTIVE:
            return 'alive'
        return 'dead'
    finally:
        kernel32.CloseHandle(handle)


def _posix_process_liveness(pid: int) -> str:
    if pid <= 0:
        return 'unknown'
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return 'dead'
    except PermissionError:
        return 'alive'
    except (OSError, OverflowError):
        return 'unknown'
    return 'alive'


def _lock_owner_process_liveness(pid: int) -> str:
    if sys.platform == 'win32':
        return _windows_process_liveness(pid)
    return _posix_process_liveness(pid)


def _stale_registry_lock(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        owner = int(lock_path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return False
    status = _lock_owner_process_liveness(owner)
    if status == 'dead':
        _LOG.info(
            'Removing stale registry lock %s (owner pid %s confirmed not running)',
            lock_path,
            owner,
        )
        return True
    return False


@contextmanager
def registry_file_lock(registry_path: Path, *, timeout_s: float = 30.0) -> Iterator[None]:
    """Best-effort exclusive lock for registry mutations (same machine, cooperative users)."""
    lock_path = registry_path.with_suffix('.lock')
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pid = str(os.getpid())
    reentrant = False
    if lock_path.exists():
        try:
            reentrant = lock_path.read_text(encoding='utf-8').strip() == pid
        except OSError:
            reentrant = False
    if not reentrant:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, pid.encode())
                os.close(fd)
                break
            except FileExistsError:
                if _stale_registry_lock(lock_path):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Could not acquire registry lock at {lock_path}')
                time.sleep(0.05)
    try:
        yield
    finally:
        if not reentrant:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def _find_active_run(registry: dict, fingerprint: str, protocol_id: str) -> dict | None:
    for run in reversed(registry.get('runs', [])):
        if run.get('protocol_id') == protocol_id and run.get('fingerprint') == fingerprint:
            if run.get('status') in ('in_progress', 'completed'):
                return run
    return None


def _find_protocol_in_progress(registry: dict, protocol_id: str) -> dict | None:
    for run in reversed(registry.get('runs', [])):
        if run.get('protocol_id') == protocol_id and run.get('status') == 'in_progress':
            return run
    return None


def record_run_start(
    registry_path: Path,
    *,
    protocol_id: str,
    fingerprint: str,
    code_hash: str,
    output_dir: Path,
    mode: str,
) -> dict:
    with registry_file_lock(registry_path):
        registry = bootstrap_registry(registry_path)
        inflight = _find_protocol_in_progress(registry, protocol_id)
        if inflight:
            raise RuntimeError(
                f'Protocol {protocol_id} already in_progress (output {inflight.get("output_dir")}). '
                'Wait for completion/failure or use override with reason after marking failed.'
            )
        run = {
            'protocol_id': protocol_id,
            'fingerprint': fingerprint,
            'code_hash': code_hash,
            'output_dir': str(output_dir.resolve()),
            'mode': mode,
            'status': 'in_progress',
            'started_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        registry.setdefault('runs', []).append(run)
        _atomic_write_json(registry_path, registry)
        return run


def record_run_finish(
    registry_path: Path,
    *,
    protocol_id: str,
    fingerprint: str,
    status: str,
    message: str | None = None,
) -> None:
    with registry_file_lock(registry_path):
        registry = bootstrap_registry(registry_path)
        for run in reversed(registry.get('runs', [])):
            if (
                run.get('protocol_id') == protocol_id
                and run.get('fingerprint') == fingerprint
                and run.get('status') == 'in_progress'
            ):
                run['status'] = status
                run['finished_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                if message:
                    run['message'] = message
                break
        _atomic_write_json(registry_path, registry)


def assert_final_allowed(
    registry_path: Path,
    receipt_path: Path,
    config: dict,
    *,
    output_dir: Path | None = None,
    override: bool = False,
    override_reason: str | None = None,
) -> None:
    fp = config_fingerprint(config)
    protocol_id = config['protocol_id']
    if config.get('protocol_role') != 'final':
        raise RuntimeError('assert_final_allowed called on non-final protocol')

    with registry_file_lock(registry_path):
        registry = bootstrap_registry(registry_path)
        inflight = _find_protocol_in_progress(registry, protocol_id)
        if inflight and not override:
            raise RuntimeError(
                f'Final protocol {protocol_id} run in_progress at {inflight.get("output_dir")}. '
                'Cannot start another final attempt (including from a different --output cwd).'
            )
        for run in reversed(registry.get('runs', [])):
            if run.get('protocol_id') != protocol_id or run.get('fingerprint') != fp:
                continue
            st = run.get('status')
            if st == 'failed' and not override:
                raise RuntimeError(
                    f'Prior final attempt failed for fingerprint {fp}. '
                    'Pass --override-final with --override-reason to retry.'
                )
            if st == 'completed' and not override:
                raise RuntimeError(
                    f'Final protocol {protocol_id} (fingerprint {fp}) already completed '
                    f"(output {run.get('output_dir')}). Registry blocks repeat final peek. "
                    'Use --override-final with --override-reason for intentional rerun.'
                )

        existing = load_receipt(receipt_path)
        if existing and existing.get('fingerprint') == fp and existing.get('status') == 'completed' and not override:
            raise RuntimeError(
                'Final evaluation receipt already exists for this protocol fingerprint in the output directory. '
                'Pass --override-final with --override-reason to record an intentional rerun.'
            )
        if output_dir is not None:
            for run in registry.get('runs', []):
                if (
                    run.get('protocol_id') == protocol_id
                    and run.get('fingerprint') == fp
                    and run.get('status') == 'completed'
                    and run.get('output_dir') != str(output_dir.resolve())
                    and not override
                ):
                    raise RuntimeError(
                        'Completed final run exists for this fingerprint under a different output directory. '
                        'Use a single output path or --override-final with reason.'
                    )
    if override and not override_reason:
        raise ValueError('override-final requires a non-empty override-reason')


def write_final_receipt(
    receipt_path: Path,
    config: dict,
    results_summary: dict,
    *,
    code_hash: str,
    registry_path: Path,
    override: bool = False,
    override_reason: str | None = None,
    prior_receipt: dict | None = None,
) -> dict:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    fp = config_fingerprint(config)
    history = []
    if prior_receipt:
        history.append(prior_receipt)
    prev = load_receipt(receipt_path)
    if prev:
        history.append(prev)

    receipt = {
        'mode': 'final',
        'status': 'completed',
        'protocol_id': config['protocol_id'],
        'fingerprint': fp,
        'code_hash': code_hash,
        'registry_path': str(registry_path.resolve()),
        'timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'seeds': config.get('seeds'),
        'protocol_role': config.get('protocol_role'),
        'override': override,
        'override_reason': override_reason,
        'prior_receipts': history,
        'results_summary': results_summary,
        'exposure_record_v1': FINAL_HOLDOUT_V1,
        'exposure_record_v2': V2_EXPOSURE_RECORD,
        'registry_scope_note': REGISTRY_SCOPE_NOTE,
    }
    _atomic_write_json(receipt_path, receipt)
    record_run_finish(registry_path, protocol_id=config['protocol_id'], fingerprint=fp, status='completed')
    return receipt


def assert_output_writable(output_dir: Path, *, force: bool, role: str) -> None:
    marker = output_dir / 'pilot_results.json'
    if marker.exists() and not force:
        if role == 'final':
            raise RuntimeError(
                f'Output {output_dir} already contains pilot_results.json. '
                'Final runs require a fresh directory or --override-final workflow.'
            )
        raise RuntimeError(
            f'Output {output_dir} already contains pilot_results.json. Pass --force to overwrite development artifacts.'
        )


def embargo_note() -> str:
    return (
        'Real TrafficFlowBench February-week adapters need an explicit embargo and freshness '
        'evidence before treating holdout windows as final; this pilot uses only synthetic truth.'
    )
