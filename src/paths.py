"""Project roots and overridable data or artifact locations."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolved(name: str, default: Path) -> Path:
    override = os.environ.get(name, "").strip()
    return Path(override).expanduser().resolve() if override else default.resolve()


def data_dir() -> Path:
    return _resolved("TFB_DATA_DIR", project_root() / "data")


def release_dir() -> Path:
    return _resolved("TFB_RELEASE_ROOT", data_dir() / "kaggle_public")


def artifacts_dir() -> Path:
    return _resolved("TFB_ARTIFACTS_DIR", project_root() / "artifacts")


def official_output_dir() -> Path:
    return _resolved("TFB_OFFICIAL_DIR", artifacts_dir() / "official")


def v8_replay_output_dir() -> Path:
    return _resolved("TFB_V8_REPLAY_DIR", artifacts_dir() / "v8_replay")
