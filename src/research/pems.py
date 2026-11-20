"""Capture the rolling Caltrans feed without treating late downloads as live data."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE_URL = "https://pems.dot.ca.gov/feeds"
DISTRICTS = (3, 4, 5, 6, 7, 8, 10, 11, 12)
FEED_NAME = re.compile(r'5minagg_(\d{14})\.txt')
PEMS_TIMEZONE = "America/Los_Angeles"
CADENCE_MS = 300_000


class AmbiguousFeedTime(ValueError):
    """A wall time that the compact feed cannot map to one UTC interval."""


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def local_interval(value: str) -> datetime:
    """Reject ambiguous wall times since the compact feed has no UTC offset."""
    naive = datetime.strptime(value, "%m/%d/%Y %H:%M:%S")
    zone = ZoneInfo(PEMS_TIMEZONE)
    first = naive.replace(tzinfo=zone, fold=0)
    alternate = naive.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != alternate.utcoffset():
        raise AmbiguousFeedTime(f"Ambiguous or nonexistent PeMS timestamp {value}")
    if naive.minute % 5 or naive.second:
        raise ValueError(f"Expected an aligned five-minute interval, got {value}")
    return first.astimezone(UTC)


def download(session: requests.Session, url: str, budget: int) -> bytes:
    with session.get(url, timeout=(10, 45), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(64 * 1024):
            content.extend(block)
            if len(content) > budget:
                raise ValueError(f"Download exceeds byte budget for {url}")
    return bytes(content)


def capture(root: Path, districts: tuple[int, ...], session: requests.Session) -> dict:
    received = datetime.now(UTC)
    metadata_day = received.date().isoformat()
    pointer = root / "station-metadata" / f"{metadata_day}.json"
    station_url = f"{BASE_URL}/vds_config.xml"
    if not pointer.exists():
        data = download(session, station_url, 20_000_000)
        if b"<pems " not in data[:1024] or b"</pems>" not in data[-1024:]:
            raise ValueError("Unexpected PeMS station metadata")
        digest = hashlib.sha256(data).hexdigest()
        atomic_write(root / "station-metadata" / f"{digest}.xml.gz", gzip.compress(data, mtime=0))
        atomic_write(pointer, json.dumps({"sha256": digest, "source_url": station_url}).encode())
    topology = json.loads(pointer.read_text(encoding="utf-8"))["sha256"]
    new = 0
    errors = {}
    gaps = {}
    for district in districts:
        directory = root / f"D{district}"
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{BASE_URL}/D{district}/Data/5min"
        try:
            listing = download(session, f"{prefix}/", 128_000).decode("utf-8")
            names = sorted({match.group() for match in FEED_NAME.finditer(listing)})
            if not names:
                raise ValueError("No timestamped files in the PeMS directory")
            for name in names:
                stamp = name.removeprefix("5minagg_").removesuffix(".txt")
                receipt = directory / f"{stamp}.json"
                if receipt.exists():
                    continue
                raw = download(session, f"{prefix}/{name}", 3_000_000)
                arrival = datetime.now(UTC)
                header, _, body = raw.partition(b"\n")
                if not body or datetime.strptime(header.decode().strip(), "%m/%d/%Y %H:%M:%S").strftime("%Y%m%d%H%M%S") != stamp:
                    raise ValueError(f"Unexpected timestamp in {name}")
                # Store exact source bytes, including files whose wall time cannot be normalized.
                digest = hashlib.sha256(raw).hexdigest()
                blob = directory / f"{digest}.txt.gz"
                atomic_write(blob, gzip.compress(raw, mtime=0))
                atomic_write(receipt, json.dumps({
                    "source_url": f"{prefix}/{name}", "district": district, "raw_sha256": digest,
                    "raw_file": blob.name, "topology_sha256": topology, "arrival": arrival.isoformat(),
                    "feed_local_time": header.decode().strip(),
                }, indent=2).encode())
                new += 1
        except (requests.RequestException, ValueError) as exc:
            errors[str(district)] = str(exc)
        intervals = sorted(datetime.strptime(file.stem, "%Y%m%d%H%M%S") for file in directory.glob("*.json"))
        gaps[str(district)] = [
            {"after_local": a.isoformat(), "before_local": b.isoformat(),
             "missing_windows": (b - a) // timedelta(minutes=5) - 1}
            for a, b in zip(intervals, intervals[1:]) if b - a > timedelta(minutes=5)
        ]
    result = {"captured_at": received.isoformat(), "new_files": new, "gaps": gaps, "district_errors": errors}
    atomic_write(root / "capture_status.json", json.dumps(result, indent=2).encode())
    return result


def compact_rows(raw: bytes) -> tuple[datetime, list[list[str]]]:
    """Read station-first snapshots. Keep only the documented shared channels."""
    header, _, body = raw.decode("utf-8-sig").partition("\n")
    interval = local_interval(header.strip())
    rows = list(csv.reader(body.splitlines()))
    if not rows or any(len(row) != 11 or not row[0].isdigit() for row in rows):
        raise ValueError("PeMS compact schema changed or the response was incomplete")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("PeMS snapshot repeats a station ID")
    return interval, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/crest/pems"))
    parser.add_argument("--districts", type=int, nargs="+", choices=DISTRICTS, default=[4, 7])
    parser.add_argument("--poll-seconds", type=int, default=0)
    options = parser.parse_args()
    if options.poll_seconds and options.poll_seconds < 60:
        parser.error("Use at least 60 seconds between polls")
    with requests.Session() as session:
        while True:
            try:
                receipt = capture(options.output, tuple(dict.fromkeys(options.districts)), session)
                print(json.dumps(receipt), flush=True)
                if not options.poll_seconds:
                    return int(bool(receipt["district_errors"]))
            except (requests.RequestException, ValueError) as exc:
                if not options.poll_seconds:
                    raise
                print(str(exc), flush=True)
            time.sleep(options.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
