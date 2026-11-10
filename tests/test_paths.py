from src.paths import (
    artifacts_dir,
    data_dir,
    official_output_dir,
    project_root,
    release_dir,
    v8_replay_output_dir,
)


def test_project_root_contains_src():
    root = project_root()
    assert (root / "src" / "paths.py").is_file()
    assert (root / "scripts" / "reproduce_v8_submission.py").is_file()


def test_env_overrides_release_and_artifacts(monkeypatch, tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    artifacts = tmp_path / "out"
    artifacts.mkdir()
    monkeypatch.setenv("TFB_RELEASE_ROOT", str(release))
    monkeypatch.setenv("TFB_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("TFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TFB_OFFICIAL_DIR", str(artifacts / "official"))
    monkeypatch.setenv("TFB_V8_REPLAY_DIR", str(artifacts / "v8"))
    assert release_dir() == release.resolve()
    assert artifacts_dir() == artifacts.resolve()
    assert data_dir() == (tmp_path / "data").resolve()
    assert official_output_dir() == (artifacts / "official").resolve()
    assert v8_replay_output_dir() == (artifacts / "v8").resolve()
