"""Targeted unit tests for measured data feasibility, provenance, and data contract.

Validates:
1. Research data manifest schema and policy compliance.
2. Downloaded raw measured data integrity (SHA-256, byte size, schema).
3. Physical validity of measured vehicle trajectories (velocities, headways).
4. Trajectory-to-macro adapter contract for CREST pipeline integration.
5. Evidence-based recording of blocked credential prerequisites.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import importlib.util

from src.research.trajectory_adapter import NormalizedTrajectory, aggregate_cell_edie

REPO_ROOT = Path(__file__).resolve().parents[1]

_frd_spec = importlib.util.spec_from_file_location(
    "fetch_research_data", REPO_ROOT / "scripts" / "fetch_research_data.py"
)
assert _frd_spec and _frd_spec.loader
frd = importlib.util.module_from_spec(_frd_spec)
_frd_spec.loader.exec_module(frd)
MANIFEST_PATH = REPO_ROOT / "configs" / "research_data_manifest.json"
DATA_DIR = REPO_ROOT / "data" / "research_external"
NGSIM_PATH = DATA_DIR / "ngsim_i80_slice.csv"

RAW_MEASURED_DATA = pytest.mark.skipif(
    not NGSIM_PATH.exists(),
    reason="NGSIM slice absent; offline tests use synthetic trajectories only",
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_structure_and_policy() -> None:
    """Verify manifest exists, is valid JSON, and adheres to strict safety policies."""
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["project"] == "CREST"
    assert "verified_fetched_sources" in manifest
    assert "blocked_candidate_sources" in manifest

    policy = manifest.get("policy", {})
    assert policy.get("no_synthetic_masquerading") is True
    assert policy.get("no_reidentification") is True
    assert policy.get("no_unauthorized_credentials") is True
    assert policy.get("max_download_budget_bytes", 0) <= 100 * 1024 * 1024


@RAW_MEASURED_DATA
def test_ngsim_i80_slice_integrity() -> None:
    """Verify downloaded NGSIM I-80 trajectory slice on disk matches manifest receipt."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    meta = manifest["verified_fetched_sources"]["ngsim_i80_slice"]

    file_path = REPO_ROOT / meta["local_relative_path"]
    assert file_path.exists(), f"File {file_path} does not exist"

    # Size and SHA256
    actual_size = file_path.stat().st_size
    assert actual_size == meta["size_bytes"], f"Size mismatch: {actual_size} vs {meta['size_bytes']}"
    assert compute_sha256(file_path) == meta["sha256"], "SHA256 mismatch"

    # Row count and columns
    df = pd.read_csv(file_path)
    assert len(df) == meta["record_count"] == 25000

    expected_cols = list(meta["fields"].keys())
    assert list(df.columns) == expected_cols


@RAW_MEASURED_DATA
def test_ngsim_i80_physics_and_identifiability() -> None:
    """Verify physical bounds and identifiability limits of NGSIM measured data."""
    df = pd.read_csv(NGSIM_PATH)

    # 1. Physical non-negativity
    assert (df["v_vel"] >= 0).all(), "Negative speed detected"
    assert (df["v_length"] > 0).all(), "Invalid vehicle length"
    assert (df["lane_id"] >= 1).all() and (df["lane_id"] <= 8).all(), "Unexpected lane ID"

    # 2. Time progression
    assert df["global_time"].is_monotonic_increasing, "Global time should be monotonically non-decreasing"

    # 3. Identifiability limitation check: freeway table has NO official OD zones
    assert df["o_zone"].isna().all(), "Expected o_zone to be unpopulated on freeway trajectory slice"
    assert df["d_zone"].isna().all(), "Expected d_zone to be unpopulated on freeway trajectory slice"


@pytest.mark.skipif(
    not (DATA_DIR / "i24_motion_sample_3.json").exists(),
    reason="I-24 sample absent",
)
def test_i24_motion_sample_integrity() -> None:
    """Verify downloaded I-24 MOTION sample JSON matches manifest receipt."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    meta = manifest["verified_fetched_sources"]["i24_motion_sample_3"]

    file_path = REPO_ROOT / meta["local_relative_path"]
    assert file_path.exists(), f"File {file_path} does not exist"

    actual_size = file_path.stat().st_size
    assert actual_size == meta["size_bytes"]
    assert compute_sha256(file_path) == meta["sha256"]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "data" in data
    assert data["metadata"]["camera_id"] == 3
    assert data["metadata"]["num_frames"] == 1801
    assert data["metadata"]["frame_rate"] == 30
    assert len(data["data"]) == 1801


def test_blocked_prerequisites_recorded() -> None:
    """Verify Caltrans PeMS and I-24 full portal blockers are documented with required prerequisites."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    blocked = manifest["blocked_candidate_sources"]

    assert "caltrans_pems_portal" in blocked
    assert blocked["caltrans_pems_portal"]["status"] == "blocked_prerequisite_required"
    assert len(blocked["caltrans_pems_portal"]["prerequisites_needed"]) > 0

    assert "i24_motion_inception_portal" in blocked
    assert blocked["i24_motion_inception_portal"]["status"] == "blocked_prerequisite_required"

    assert "metr_la_generic_loop" in blocked
    assert blocked["metr_la_generic_loop"]["status"] == "excluded_scientifically_unsuitable"


def test_trajectory_adapter_edie_fields_offline() -> None:
    """Offline adapter contract: finite Edie outputs on synthetic clip (q≈kv is definitional)."""
    t = np.arange(0, 6001, 100, dtype=np.int64)
    x = 5.0 + 9.0 * (t / 1000.0)
    v = np.full(t.size, 9.0)
    tr = NormalizedTrajectory("syn:edie", t, x, v)
    cell = aggregate_cell_edie(
        [tr],
        t_start_ms=0,
        t_end_ms=6000,
        x_start_m=0.0,
        x_end_m=50.0,
    )
    assert cell.edie_q_vph > 0
    assert cell.edie_k_vpkm > 0
    assert cell.edie_v_kmh > 0
    if np.isfinite(cell.edie_q_vph) and np.isfinite(cell.edie_k_vpkm) and np.isfinite(cell.edie_v_kmh):
        assert cell.edie_q_vph == pytest.approx(cell.edie_k_vpkm * cell.edie_v_kmh, rel=0.05)


@RAW_MEASURED_DATA
def test_ngsim_raw_timestamp_range_matches_manifest() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    ti = manifest["verified_fetched_sources"]["ngsim_i80_slice"]["time_interval"]
    df = pd.read_csv(NGSIM_PATH)
    assert int(df["global_time"].min()) == ti["raw_global_time_ms"][0]
    assert int(df["global_time"].max()) == ti["raw_global_time_ms"][1]
    assert df["vehicle_id"].nunique() == ti["unique_vehicle_ids"]
    assert df["global_time"].nunique() == ti["unique_global_timestamps"]


def test_verify_only_manifest_sha(tmp_path: Path) -> None:
    bad = tmp_path / "tiny.csv"
    bad.write_text("xyz", encoding="utf-8")
    manifest = {
        "verified_fetched_sources": {
            "ngsim_i80_slice": {
                "sha256": "abc",
                "size_bytes": 3,
                "local_relative_path": str(bad),
                "fields": {"a": {}},
                "record_count": 0,
            }
        }
    }
    with pytest.raises(frd.VerificationError, match="SHA256 mismatch"):
        frd.verify_source_manifest("ngsim_i80", manifest)


def test_verify_fails_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    manifest = {
        "verified_fetched_sources": {
            "ngsim_i80_slice": {
                "sha256": "00" * 32,
                "size_bytes": 0,
                "local_relative_path": str(missing),
                "fields": {"a": {}},
            }
        }
    }
    with pytest.raises(frd.VerificationError, match="File missing"):
        frd.verify_source_manifest("ngsim_i80", manifest)


def test_manifest_missing_trusted_sha_fails_closed() -> None:
    bad_manifest = {"verified_fetched_sources": {"ngsim_i80_slice": {"size_bytes": 1}}}
    with pytest.raises(frd.VerificationError, match="missing trusted sha256"):
        frd.verify_source_manifest("ngsim_i80", bad_manifest)


def test_corrupt_size_mismatch_fails(tmp_path: Path) -> None:
    manifest = {
        "verified_fetched_sources": {
            "ngsim_i80_slice": {
                "sha256": "38173aad33806922b4f22f516363f8a4fce53038554e6d85744e151f05870d1b",
                "size_bytes": 999999,
                "local_relative_path": "data/research_external/ngsim_i80_slice.csv",
                "fields": {"vehicle_id": {}},
            }
        }
    }
    real = REPO_ROOT / "data/research_external/ngsim_i80_slice.csv"
    if not real.exists():
        pytest.skip("NGSIM slice absent")
    with pytest.raises(frd.VerificationError, match="Size mismatch"):
        frd.verify_source_manifest("ngsim_i80", manifest)
