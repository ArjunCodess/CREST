import json
import subprocess
import sys
from pathlib import Path


def test_validate_v8_prereqs_reports_missing_assets(tmp_path):
    release = tmp_path / "kaggle_public"
    (release / "corridors").mkdir(parents=True)
    (release / "submission_key.csv").write_text("id\n", encoding="utf-8")
    source = Path(__file__).resolve().parents[1]
    package = tmp_path / "missing_package"
    package.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(source / "scripts/validate_v8_prereqs.py"),
            "--package-root",
            str(package),
            "--release-root",
            str(release),
            "--json",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert any("external/anubisyy" in issue for issue in report["issues"])
