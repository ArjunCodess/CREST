"""Check inputs for resume.ps1 / run.ps1 without starting training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import official_output_dir, project_root, release_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", type=Path, default=project_root())
    ap.add_argument("--release-root", type=Path, default=release_dir())
    args = ap.parse_args()
    package = args.package_root.resolve()
    release = args.release_root.resolve()
    issues: list[str] = []
    if not (release / "corridors").is_dir():
        issues.append(f"release missing corridors under {release}")
    if (release / "FIXTURE_ONLY.json").exists():
        issues.append("release is a test fixture, not competition data")
    report = {
        "ok": not issues,
        "release_root": str(release),
        "official_output_default": str(official_output_dir()),
        "issues": issues,
    }
    print(json.dumps(report, indent=2), flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
