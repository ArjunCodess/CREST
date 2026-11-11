"""Resolve bundled or repository research protocol configs."""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def bundled_config_path(filename: str) -> Path:
    try:
        pkg = resources.files('src.research').joinpath('bundled_configs', filename)
        with resources.as_file(pkg) as path:
            return path
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return REPO_ROOT / 'configs' / 'research' / filename


def load_raw_config(path: Path | None, *, default_bundled: str) -> tuple[dict, Path]:
    if path is None:
        resolved = bundled_config_path(default_bundled)
    else:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f'config not found: {resolved}')
    with resolved.open(encoding='utf-8') as f:
        raw = json.load(f)
    return raw, resolved


def resolve_protocol_config_name(path: Path, default_bundled: str) -> str:
    """Identify bundled default by protocol_id after load, not Path object identity."""
    if path is None:
        return default_bundled
    resolved = path.expanduser().resolve()
    bundled = bundled_config_path(default_bundled).resolve()
    repo = (REPO_ROOT / 'configs' / 'research' / default_bundled).resolve()
    if resolved == bundled or resolved == repo:
        return default_bundled
    with resolved.open(encoding='utf-8') as f:
        raw = json.load(f)
    return raw.get('protocol_id', resolved.name)
