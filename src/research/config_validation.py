"""One-shot validation for research protocol JSON (strict finite values)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from .stress_variants import STRESS_VARIANTS
from .synthetic_domain import validate_generative_scenario_params

ProtocolRole = Literal['development', 'final']

_RESERVED_FINAL_V1_SEEDS = frozenset({101, 202, 303, 404, 505})
_DEV_REGISTRY_SEEDS = frozenset({11, 22, 33, 44, 55})
RETIRED_EXPOSED_V2_SEEDS = frozenset({701, 802, 903, 1004, 1105})
# v3 archive (failed final attempt 2026-11-11); active reserve retired — see final_holdout_v3_seed_manifest.json.
FINAL_HOLDOUT_V3_ARCHIVE_SEEDS: frozenset[int] = frozenset({
    1010559880036807,
    1101903413522210,
    1261258823614491,
    1297921224299534,
    1418734052312821,
    1521028319156996,
    1536129947884980,
    1549975720351840,
    1583044294401646,
    1628714462425246,
})
# v4 amended reserve (manifest configs/research/final_holdout_v4_seed_manifest.json); v3 not rerun.
FINAL_HOLDOUT_V4_ACTIVE_SEEDS: frozenset[int] = frozenset({
    7783114929623311505,
    6242675308882067067,
    7603013050274570249,
    5561420057320424089,
    6031017206454009675,
    4905457364178722424,
    5597043143875218530,
    7244826587779272624,
    5757910615864632253,
    7205798323138762989,
})
FINAL_HOLDOUT_ACTIVE_SEEDS: frozenset[int] = FINAL_HOLDOUT_V4_ACTIVE_SEEDS
# Deprecated alias: overlap guard for development configs (treat as retired, not active).
FINAL_HOLDOUT_V2_SEEDS = RETIRED_EXPOSED_V2_SEEDS

V2_EXPOSURE_RECORD = {
    'protocol_id': 'final_holdout_v2',
    'status': 'retired_exposed',
    'retired_date': '2026-11-11',
    'retired_seeds': sorted(RETIRED_EXPOSED_V2_SEEDS),
    'exposed_subset': [701, 802],
    'exposure_reason': (
        'Independent reviewer ran run_one_scenario on reserved v2 seeds with final_protocol params '
        'despite instructions not to execute v2.'
    ),
    'known_configuration_note': 'See configs/research/final_protocol.json exposure_record',
}

V3_EXPOSURE_RECORD = {
    'protocol_id': 'final_holdout_v3',
    'status': 'retired_failed',
    'retired_date': '2026-11-11',
    'retired_seeds': sorted(FINAL_HOLDOUT_V3_ARCHIVE_SEEDS),
    'frozen_failure': {
        'code_hash': '07a399f8e9550149',
        'fingerprint': '5da3b1d664faf594',
        'failure_receipt': 'results/research_final_v3/final_attempt_failure.json',
        'registry_run_status': 'failed',
        'message': 'generative flows violate capacity; reduce demand or widen capacity',
    },
    'reported_full_matrix_exposure': {
        'disclosed_utc': '2026-11-11T11:45:00Z',
        'actor': 'implementation_subagent',
        'admission': (
            'Despite instructions not to execute reserved v3 scenarios, the implementing agent reported '
            'checking all 10 stress variants × 10 archived v3 seeds after generative repair (full 10×10 matrix).'
        ),
        'evidence_class': 'agent_reported',
        'independently_observed_subset': [],
        'distinction': (
            'Reported exposure is self-disclosed agent admission, not independently observed execution '
            'receipts like v2 subset 701/802.'
        ),
    },
    'manifest': 'configs/research/final_holdout_v3_seed_manifest.json',
    'exposure_record_path': 'configs/research/final_holdout_v3_exposure_record.json',
    'no_pristine_preregistered_claim': True,
}


class ConfigValidationError(ValueError):
    pass


def _require_int_list(value: Any, field: str, *, min_len: int = 1) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f'{field}: non-empty list required')
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigValidationError(f'{field}[{i}]: must be integer seed')
        out.append(item)
    if len(set(out)) != len(out):
        raise ConfigValidationError(f'{field}: duplicate seeds')
    if len(out) < min_len:
        raise ConfigValidationError(f'{field}: need at least {min_len} seeds')
    return out


def _require_str_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f'{field}: non-empty list required')
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(f'{field}[{i}]: must be non-empty string')
    return list(value)


def validate_and_normalize_protocol(
    raw: dict[str, Any],
    *,
    cli_mode: ProtocolRole,
    development_seed_allowlist: frozenset[int] | None = None,
    final_reserved_seeds: frozenset[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigValidationError('protocol config must be a JSON object')

    cfg = deepcopy(raw)
    for key in list(cfg):
        if key.startswith('_'):
            del cfg[key]

    protocol_id = cfg.get('protocol_id')
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ConfigValidationError('protocol_id is required')
    protocol_id = protocol_id.strip()

    role = cfg.get('protocol_role')
    if role not in ('development', 'final'):
        raise ConfigValidationError('protocol_role must be development or final')
    if role != cli_mode:
        raise ConfigValidationError(
            f'CLI mode {cli_mode!r} does not match protocol_role {role!r} in {protocol_id}'
        )

    seeds = _require_int_list(cfg.get('seeds'), 'seeds', min_len=5 if role == 'final' else 5)
    variants = _require_str_list(cfg.get('stress_variants'), 'stress_variants')
    for v in variants:
        if v not in STRESS_VARIANTS:
            raise ConfigValidationError(f'unknown stress variant: {v}')

    params = cfg.get('scenario_params')
    if not isinstance(params, dict):
        raise ConfigValidationError('scenario_params must be an object')
    for name in ('n_links', 'n_steps', 'demand_vph', 'noise_std', 'missing_rate', 'capacity_vehicles'):
        if name not in params:
            raise ConfigValidationError(f'scenario_params missing {name}')
    layout = str(params.get('layout', 'chain'))
    try:
        validated = validate_generative_scenario_params(params, layout=layout)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc
    for key, val in validated.items():
        if key != 'cfl_policy':
            params[key] = val

    if role == 'development':
        allow = development_seed_allowlist or _DEV_REGISTRY_SEEDS
        reserved_retired = final_reserved_seeds or RETIRED_EXPOSED_V2_SEEDS
        overlap_final = set(seeds) & reserved_retired
        if overlap_final:
            raise ConfigValidationError(f'development seeds overlap retired final holdout: {sorted(overlap_final)}')
        overlap_active = set(seeds) & FINAL_HOLDOUT_ACTIVE_SEEDS
        if overlap_active:
            raise ConfigValidationError(f'development seeds overlap active final holdout: {sorted(overlap_active)}')
        overlap_v3 = set(seeds) & FINAL_HOLDOUT_V3_ARCHIVE_SEEDS
        if overlap_v3:
            raise ConfigValidationError(f'development seeds overlap archived final v3 holdout: {sorted(overlap_v3)}')
        overlap_v4 = set(seeds) & FINAL_HOLDOUT_V4_ACTIVE_SEEDS
        if overlap_v4:
            raise ConfigValidationError(f'development seeds overlap active final v4 holdout: {sorted(overlap_v4)}')
        if allow and not set(seeds).issubset(allow):
            bad = sorted(set(seeds) - set(allow))
            raise ConfigValidationError(f'development seeds not in dev registry: {bad}')
        overlap_exposed = set(seeds) & _RESERVED_FINAL_V1_SEEDS
        if overlap_exposed:
            raise ConfigValidationError(f'seeds collide with retired exposed final v1: {sorted(overlap_exposed)}')
        overlap_v2 = set(seeds) & RETIRED_EXPOSED_V2_SEEDS
        if overlap_v2:
            raise ConfigValidationError(f'seeds collide with retired exposed final v2: {sorted(overlap_v2)}')

    if role == 'final':
        active = FINAL_HOLDOUT_ACTIVE_SEEDS
        overlap_v3 = set(seeds) & FINAL_HOLDOUT_V3_ARCHIVE_SEEDS
        if overlap_v3:
            raise ConfigValidationError(
                'final protocol must not use archived v3 holdout seeds (retired_failed; no v3 execution)'
            )
        if set(seeds) != set(active):
            raise ConfigValidationError(
                f'final seeds must match active holdout exactly: expected {sorted(active)}, got {seeds}'
            )
        overlap_dev = set(seeds) & _DEV_REGISTRY_SEEDS
        if overlap_dev:
            raise ConfigValidationError(f'final seeds overlap development registry: {sorted(overlap_dev)}')
        overlap_v1 = set(seeds) & _RESERVED_FINAL_V1_SEEDS
        if overlap_v1:
            raise ConfigValidationError('final protocol uses retired exposed v1 seeds')
        overlap_v2_retired = set(seeds) & RETIRED_EXPOSED_V2_SEEDS
        if overlap_v2_retired:
            raise ConfigValidationError('final protocol uses retired exposed v2 seeds')

    description = cfg.get('description', '')
    if not isinstance(description, str):
        raise ConfigValidationError('description must be a string')

    normalized = {
        'protocol_id': protocol_id,
        'protocol_role': role,
        'description': description,
        'seeds': seeds,
        'scenario_params': params,
        'stress_variants': variants,
    }
    return normalized


def canonical_protocol_content(config: dict[str, Any]) -> dict[str, Any]:
    """Semantic content only (path/mode/timestamp excluded) for fingerprinting."""
    keys = ('protocol_id', 'protocol_role', 'description', 'seeds', 'scenario_params', 'stress_variants')
    return {k: deepcopy(config[k]) for k in keys if k in config}
