"""CLI for identifiability-aware research pilot (development default; gated final mode)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config_paths import REPO_ROOT, bundled_config_path, load_raw_config
from .config_validation import validate_and_normalize_protocol, FINAL_HOLDOUT_V2_SEEDS
from .experiments import plot_diagnostics, run_experiment_matrix, write_artifacts
from .protocol import (
    assert_final_allowed,
    assert_output_writable,
    default_registry_path,
    load_receipt,
    record_run_finish,
    record_run_start,
    research_code_content_hash,
    write_final_receipt,
)

DEFAULT_OUTPUT = REPO_ROOT / 'results' / 'research'


def load_config(path: Path | None, *, cli_mode: str) -> dict:
    if cli_mode == 'final':
        raw, resolved = load_raw_config(path, default_bundled='final_protocol.json')
    else:
        raw, resolved = load_raw_config(path, default_bundled='dev_pilot.json')
    cfg = validate_and_normalize_protocol(
        raw,
        cli_mode=cli_mode,  # type: ignore[arg-type]
        final_reserved_seeds=FINAL_HOLDOUT_V2_SEEDS,
    )
    cfg['_config_resolved'] = str(resolved)
    cfg['scenario_set'] = cli_mode
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='CREST identifiability research pilot')
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Protocol JSON (default: bundled dev_pilot or final_protocol by mode)',
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--registry', type=Path, default=None, help='Protocol registry ledger (default: CWD or env)')
    parser.add_argument('--mode', choices=('development', 'final'), default='development')
    parser.add_argument('--force', action='store_true', help='Allow overwriting development artifacts')
    parser.add_argument('--override-final', action='store_true')
    parser.add_argument('--override-reason', type=str, default='')
    args = parser.parse_args(argv)

    if args.mode == 'final' and args.config is not None:
        try:
            raw, _ = load_raw_config(args.config, default_bundled='dev_pilot.json')
            if raw.get('protocol_role') == 'development':
                raise SystemExit('Refusing final mode with a development protocol file')
        except FileNotFoundError:
            pass

    registry_path = args.registry or default_registry_path()
    output_dir = args.output.expanduser().resolve()

    config = load_config(args.config, cli_mode=args.mode)
    code_hash = research_code_content_hash(REPO_ROOT)
    fingerprint = __import__('src.research.protocol', fromlist=['config_fingerprint']).config_fingerprint(config)

    assert_output_writable(output_dir, force=args.force or args.override_final, role=config['protocol_role'])

    receipt_path = output_dir / 'final_receipt.json'
    prior_receipt = load_receipt(receipt_path) if config['protocol_role'] == 'final' else None

    if args.mode == 'final':
        assert_final_allowed(
            registry_path,
            receipt_path,
            config,
            output_dir=output_dir,
            override=args.override_final,
            override_reason=args.override_reason or None,
        )
        record_run_start(
            registry_path,
            protocol_id=config['protocol_id'],
            fingerprint=fingerprint,
            code_hash=code_hash,
            output_dir=output_dir,
            mode=args.mode,
        )

    try:
        payload = run_experiment_matrix(config, REPO_ROOT, code_hash=code_hash)
        write_artifacts(payload, output_dir)
        figure = plot_diagnostics(payload, output_dir)

        if args.mode == 'final':
            write_final_receipt(
                receipt_path,
                config,
                {'scenario_summaries': payload['scenario_summaries']},
                code_hash=code_hash,
                registry_path=registry_path,
                override=args.override_final,
                override_reason=args.override_reason or None,
                prior_receipt=prior_receipt,
            )
    except Exception as exc:
        if args.mode == 'final':
            record_run_finish(
                registry_path,
                protocol_id=config['protocol_id'],
                fingerprint=fingerprint,
                status='failed',
                message=str(exc),
            )
        raise

    print(json.dumps({
        'mode': args.mode,
        'protocol_id': config['protocol_id'],
        'output': str(output_dir),
        'registry': str(registry_path),
        'code_hash': code_hash,
        'n_runs': len(payload['rows']),
        'figure': str(figure) if figure else None,
        'default_bundled_dev': str(bundled_config_path('dev_pilot.json')),
        'summaries': payload['scenario_summaries'][:3],
    }, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
