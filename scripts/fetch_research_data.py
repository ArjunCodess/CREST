#!/usr/bin/env python3
"""Fetch and verify measured traffic research slices (manifest-trusted receipts)."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_research_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "research_external"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "research_data_manifest.json"

CLI_SOURCE_TO_MANIFEST: dict[str, str] = {
    "ngsim_i80": "ngsim_i80_slice",
    "i24_sample": "i24_motion_sample_3",
}


class VerificationError(RuntimeError):
    pass


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise VerificationError(f"Manifest missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "verified_fetched_sources" not in data:
        raise VerificationError("Manifest missing verified_fetched_sources")
    return data


def _manifest_entry(manifest: dict[str, Any], cli_source: str) -> dict[str, Any]:
    key = CLI_SOURCE_TO_MANIFEST.get(cli_source)
    if not key:
        raise VerificationError(f"Unknown source '{cli_source}'")
    verified = manifest["verified_fetched_sources"]
    if key not in verified:
        raise VerificationError(f"Manifest has no entry for {key}")
    entry = verified[key]
    if not entry.get("sha256"):
        raise VerificationError(f"Manifest entry {key} missing trusted sha256 (fail closed)")
    if entry.get("size_bytes") is None:
        raise VerificationError(f"Manifest entry {key} missing size_bytes")
    return entry


def _resolve_local_path(manifest: dict[str, Any], cli_source: str, output_dir: Path) -> Path:
    entry = _manifest_entry(manifest, cli_source)
    rel = entry.get("local_relative_path")
    if not rel:
        raise VerificationError(f"Manifest entry missing local_relative_path for {cli_source}")
    return (REPO_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)


def _schema_check_csv(path: Path, entry: dict[str, Any]) -> None:
    import pandas as pd

    fields = entry.get("fields")
    if not isinstance(fields, dict):
        raise VerificationError("Manifest CSV fields schema missing")
    df = pd.read_csv(path, nrows=1)
    expected_cols = list(fields.keys())
    if list(df.columns) != expected_cols:
        raise VerificationError(
            f"CSV column schema mismatch: expected {len(expected_cols)} cols, got {list(df.columns)[:5]}..."
        )


def _schema_check_json(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "metadata" not in data or "data" not in data:
        raise VerificationError("I-24 sample JSON missing metadata/data envelope")


def verify_source_manifest(
    cli_source: str,
    manifest: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail closed: bytes, sha256, schema, and optional record_count must match manifest."""
    entry = _manifest_entry(manifest, cli_source)
    path = _resolve_local_path(manifest, cli_source, output_dir or DEFAULT_OUT_DIR)
    if not path.is_file():
        raise VerificationError(f"File missing: {path}")

    size = path.stat().st_size
    expected_size = int(entry["size_bytes"])
    if size != expected_size:
        raise VerificationError(f"Size mismatch for {cli_source}: {size} != {expected_size}")

    expected_sha = str(entry["sha256"]).lower()
    actual_sha = compute_sha256(path)
    if actual_sha != expected_sha:
        raise VerificationError(
            f"SHA256 mismatch for {cli_source}: expected {expected_sha}, got {actual_sha}"
        )

    if path.suffix.lower() == ".csv":
        _schema_check_csv(path, entry)
        if entry.get("record_count") is not None:
            with path.open(encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            if n_lines - 1 != int(entry["record_count"]):
                raise VerificationError(
                    f"Record count mismatch: {n_lines - 1} != {entry['record_count']}"
                )
    elif path.suffix.lower() == ".json":
        _schema_check_json(path)

    logger.info("Verification passed for %s (SHA256: %s)", cli_source, actual_sha)
    return {
        "source": cli_source,
        "path": str(path),
        "size_bytes": size,
        "sha256": actual_sha,
        "manifest_key": CLI_SOURCE_TO_MANIFEST[cli_source],
    }


def _build_download_url(entry: dict[str, Any]) -> str:
    if entry.get("query_url"):
        return str(entry["query_url"])
    if entry.get("download_url"):
        return str(entry["download_url"])
    if entry.get("soda_endpoint_url"):
        return str(entry["soda_endpoint_url"])
    raise VerificationError("Manifest entry has no download URL")


def fetch_source(
    cli_source: str,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    timeout_sec: int = 45,
) -> dict[str, Any]:
    entry = _manifest_entry(manifest, cli_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = _resolve_local_path(manifest, cli_source, output_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    full_url = _build_download_url(entry)
    max_bytes = int(manifest.get("policy", {}).get("max_download_budget_bytes", 100 * 1024 * 1024))
    src_cap = entry.get("max_bytes")
    if isinstance(src_cap, int):
        max_bytes = min(max_bytes, src_cap)

    logger.info("Fetching %s from: %s", cli_source, full_url)
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": "CREST-Research-Data-Client/1.0 (Traffic Science Research)"},
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        content = resp.read()

    size_bytes = len(content)
    if size_bytes > max_bytes:
        raise RuntimeError(
            f"Downloaded {size_bytes} bytes exceeds safety threshold {max_bytes} for {cli_source}"
        )

    fd, tmp_name = tempfile.mkstemp(prefix=f"crest_{cli_source}_", suffix=target_path.suffix, dir=target_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "wb") as f:
            f.write(content)
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    expected_sha = str(entry["sha256"]).lower()
    actual_sha = compute_sha256(target_path)
    if actual_sha != expected_sha:
        target_path.unlink(missing_ok=True)
        raise VerificationError(
            f"Downloaded hash {actual_sha} does not match manifest trusted hash {expected_sha}"
        )

    t_end = datetime.datetime.now(datetime.timezone.utc)
    return {
        "source_key": cli_source,
        "local_path": str(target_path),
        "size_bytes": size_bytes,
        "sha256": actual_sha,
        "retrieved_utc": t_end.isoformat(),
        "manifest_sha256_unchanged": expected_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify measured traffic research data slices.")
    parser.add_argument(
        "--source",
        choices=["ngsim_i80", "i24_sample", "all"],
        default="all",
        help="Source to fetch (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Trusted manifest with expected sha256/size/schema",
    )
    parser.add_argument("--verify-only", action="store_true", help="Verify on-disk files against manifest")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest.expanduser().resolve())
    except VerificationError as exc:
        logger.error("%s", exc)
        return 1

    sources = list(CLI_SOURCE_TO_MANIFEST.keys()) if args.source == "all" else [args.source]

    if args.verify_only:
        try:
            for src in sources:
                verify_source_manifest(src, manifest)
        except VerificationError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    for src in sources:
        try:
            res = fetch_source(src, manifest, args.output_dir.expanduser().resolve())
            print(json.dumps(res, indent=2))
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", src, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
