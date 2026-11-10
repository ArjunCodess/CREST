from pathlib import Path
import subprocess
import sys


def test_reproduction_preserves_existing_predictions(tmp_path):
    prediction = tmp_path / 'state.csv'
    prediction.write_text('retained prediction\n')
    package = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(package / 'scripts/reproduce_v4_submission.py'),
         '--release-root', str(tmp_path / 'unavailable-release'),
         '--package-root', str(package), '--output', str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'fresh output directory' in result.stderr
    assert prediction.read_text() == 'retained prediction\n'
    assert not (tmp_path / '.reproduction.lock').exists()


def test_reproduction_preserves_another_run_lock(tmp_path):
    lock = tmp_path / '.reproduction.lock'
    lock.write_text('another run')
    package = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(package / 'scripts/reproduce_v4_submission.py'),
         '--release-root', str(tmp_path / 'unavailable-release'),
         '--package-root', str(package), '--output', str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'FileExistsError' in result.stderr
    assert lock.read_text() == 'another run'
