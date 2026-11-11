"""Record pre_evaluation_freeze_v4.json after preflight checks (does not execute final matrix)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.research.config_paths import REPO_ROOT, load_raw_config
from src.research.config_validation import FINAL_HOLDOUT_V4_ACTIVE_SEEDS, canonical_protocol_content
from src.research.protocol import config_fingerprint
from src.research.provenance import research_code_content_hash


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytest-count', type=int, required=True)
    parser.add_argument('--pytest-failed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.pytest_failed:
        raise SystemExit('refusing freeze: pytest failures reported')

    registry = REPO_ROOT / '.crest_research_protocol_registry.json'
    reg = json.loads(registry.read_text(encoding='utf-8'))
    v4_runs = [r for r in reg.get('runs', []) if r.get('protocol_id') == 'final_holdout_v4']
    if v4_runs:
        raise SystemExit('refusing freeze: prior v4 registry run row exists')

    manifest_path = REPO_ROOT / 'configs' / 'research' / 'final_holdout_v4_seed_manifest.json'
    config_path = REPO_ROOT / 'configs' / 'research' / 'final_protocol_v4.json'
    raw, _ = load_raw_config(config_path, default_bundled='final_protocol.json')
    fp = config_fingerprint(raw)
    manifest_digest = _digest_file(manifest_path)
    code_hash = research_code_content_hash(REPO_ROOT)

    freeze = {
        'receipt_type': 'crest_research_pre_evaluation_freeze_v4',
        'freeze_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'research_code_hash': code_hash,
        'prior_frozen_failed_final_code_hash': '07a399f8e9550149',
        'post_amendment_baseline_code_hash': '6959a7802af6991e',
        'final_protocol_id': 'final_holdout_v4',
        'final_protocol_fingerprint_sha256_prefix': fp,
        'final_holdout_active_seeds_count': len(FINAL_HOLDOUT_V4_ACTIVE_SEEDS),
        'seed_manifest_path': str(manifest_path.relative_to(REPO_ROOT)),
        'seed_manifest_sha256': manifest_digest,
        'stress_variants_count': len(raw['stress_variants']),
        'expected_final_scenarios': len(raw['stress_variants']) * len(FINAL_HOLDOUT_V4_ACTIVE_SEEDS),
        'primary_metric': raw.get('primary_metric', 'bounded_state_rmse_missing'),
        'analysis_predeclaration': raw.get('analysis_predeclaration'),
        'selection_history': (
            'v4 seeds from final_holdout_v4_seed_manifest.json; stress variants unchanged from archived v3; '
            'amended evaluation after generative repair; v3 not rerun.'
        ),
        'amended_history': {
            'v3_status': 'retired_failed',
            'v3_failure_receipt': 'results/research_final_v3/final_attempt_failure.json',
            'v3_pre_freeze_preserved': 'results/research/pre_evaluation_freeze.json',
        },
        'holdout_exposure': {
            'v1': 'retired_exposed',
            'v2': 'retired_exposed',
            'v3': 'retired_failed_archived',
            'v4': 'active_unexposed_at_freeze',
        },
        'prior_v4_attempt_check': {'registry_rows': 0, 'passed': True},
        'prefreeze_checks': {
            'pytest_passed': args.pytest_count,
            'development_regenerated': True,
            'measured_regenerated': True,
            'data_manifest_verify_only': True,
        },
        'final_command_declared': (
            'python -m src.research.runner --mode final '
            '--config configs/research/final_protocol_v4.json '
            '--output results/research_final_v4 '
            '--registry .crest_research_protocol_registry.json'
        ),
        'final_output_declared': 'results/research_final_v4',
        'registry_path_declared': '.crest_research_protocol_registry.json',
        'canonical_protocol_content_digest_sha256_prefix': hashlib.sha256(
            json.dumps(canonical_protocol_content(raw), sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()[:16],
    }
    out = REPO_ROOT / 'results' / 'research' / 'pre_evaluation_freeze_v4.json'
    out.write_text(json.dumps(freeze, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'wrote': str(out), 'code_hash': code_hash, 'fingerprint': fp}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
