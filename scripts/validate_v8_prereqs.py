"""Check v8 checkpoint-replay inputs without running inference."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import project_root  # noqa: E402

V8_IMPORTS = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "lightgbm",
    "torch",
    "polars",
    "catboost",
    "yaml",
    "requests",
    "threadpoolctl",
)
ANUBISYY_FILES = (
    "state_inference.py", "dataset-metadata.json", "queue_features.py",
    "queue_ongoing.py", "onset_lgb.txt", "ongoing_lgb.txt", "onset.cbm",
    "onset_settings.json",
)
BIZAI_SEEDS = (42, 7, 123)
OWN_PATHS = (
    "models/state",
    "models/queue/queue_proxy_cv.csv",
    "models/events",
    "models/onset",
    "models/enhanced/model_manifest.json",
    "models/ongoing",
    "models/fd_decisions",
    "models/wave_v7/model_manifest.json",
)


def _missing_imports() -> list[str]:
    missing: list[str] = []
    for name in V8_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def _release_issues(release: Path) -> list[str]:
    issues: list[str] = []
    if not release.is_dir():
        return [f"release root missing: {release}"]
    for marker in ("corridors", "submission_key.csv"):
        if not (release / marker).exists():
            issues.append(f"release missing {marker}")
    if (release / "FIXTURE_ONLY.json").exists():
        issues.append("release is a pytest fixture, not kaggle_public")
    return issues


def _package_issues(package: Path, variant: str = "unet") -> list[str]:
    issues: list[str] = []
    script = package / "scripts/reproduce_v8_submission.py"
    if not script.is_file():
        issues.append(f"missing {script.relative_to(package)}")
    lock = package / "requirements-v8.lock.txt"
    if not lock.is_file():
        issues.append("missing requirements-v8.lock.txt (v8 replay pin list)")
    anubis = package / "external/anubisyy"
    if not anubis.is_dir():
        issues.append("missing external/anubisyy (restore the matching v8 package assets)")
    else:
        for name in ANUBISYY_FILES:
            if not (anubis / name).is_file():
                issues.append(f"external/anubisyy missing {name}")
        if not any(anubis.glob("*.cbm")) and not any(anubis.glob("**/*.cbm")):
            issues.append("external/anubisyy has no CatBoost .cbm checkpoints")
    bizai = package / "external/bizai"
    if variant == "unet" and not bizai.is_dir():
        issues.append("missing external/bizai (MIT U-Net2 weights)")
    elif variant == "unet":
        for seed in BIZAI_SEEDS:
            weight = bizai / f"artifacts/weights/unet2/seed{seed}/weights.pt"
            if not weight.is_file():
                issues.append(f"missing BizAI weight seed{seed}")
        if not (bizai / "LICENSE").is_file():
            issues.append("external/bizai missing LICENSE")
    for rel in OWN_PATHS:
        if not (package / rel).exists():
            issues.append(f"missing own checkpoint path {rel}")
    starter = package / "starter/LICENSE"
    if not starter.is_file():
        issues.append("missing starter/LICENSE")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", type=Path, default=project_root())
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--variant", choices=("cat", "unet"), default="unet")
    ap.add_argument("--json", type=Path, help="Optional path to write the report JSON")
    args = ap.parse_args()
    package = args.package_root.resolve()
    release = args.release_root.resolve()
    missing = _missing_imports()
    issues = _release_issues(release) + _package_issues(package, args.variant)
    if missing:
        issues.append("python imports missing: " + ", ".join(missing))
    warnings: list[str] = []
    try:
        import torch

        if args.json is None and not torch.cuda.is_available():
            warnings.append("CUDA not available; use --device CPU or expect numeric drift vs GPU replay")
    except ImportError:
        pass
    report = {
        "ok": not issues,
        "package_root": str(package),
        "release_root": str(release),
        "issues": issues,
        "warnings": warnings,
    }
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text, flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
