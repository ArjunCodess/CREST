"""Archive complete FHWA freeway recordings as time-partitioned source CSVs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import requests

from .pems import atomic_write

ENDPOINT = "https://data.transportation.gov/resource/8ect-6jqj"
SITES = ("i-80", "us-101")
FIELDS = ("vehicle_id", "frame_id", "global_time", "local_y", "v_vel", "lane_id", "location")
BIN_MS = 60_000


def recording_bins(site: str) -> list[dict]:
    response = requests.get(f"{ENDPOINT}.json", timeout=90, params={
        "$select": f"floor(global_time/{BIN_MS}) as bucket,count(*) as rows,min(global_time) as first,max(global_time) as last",
        "$where": f"location='{site}'", "$group": "bucket", "$order": "bucket", "$limit": 5000,
    })
    response.raise_for_status()
    bins = [{key: int(row[key]) for key in ("bucket", "rows", "first", "last")} for row in response.json()]
    if not bins or len({row["bucket"] for row in bins}) != len(bins):
        raise ValueError(f"Missing or duplicate NGSIM time bins for {site}")
    return bins


def fetch_bin(directory: Path, site: str, row: dict) -> dict:
    stamp = row["bucket"] * BIN_MS
    path = directory / f"{stamp}.csv.gz"
    manifest_path = directory / f"{stamp}.json"
    if manifest_path.exists() and path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["rows"] == row["rows"] and hashlib.sha256(path.read_bytes()).hexdigest() == saved["gzip_sha256"]:
            return saved
    fields = ",".join(FIELDS)
    response = requests.get(f"{ENDPOINT}.csv", timeout=(10, 150), params={
        "$select": fields, "$where": f"location='{site}' AND global_time >= {stamp} AND global_time < {stamp + BIN_MS}",
        "$order": "global_time,vehicle_id,frame_id", "$limit": row["rows"] + 1,
    })
    response.raise_for_status()
    data = response.content
    lines = data.splitlines()
    if tuple(next(csv.reader([lines[0].decode("utf-8-sig")]))) != FIELDS or len(lines) != row["rows"] + 1:
        raise ValueError(f"NGSIM source row count or schema changed for {site} at {stamp}")
    encoded = gzip.compress(data, compresslevel=3, mtime=0)
    receipt = {**row, "site": site, "filename": path.name, "source_url": response.url,
               "sha256": hashlib.sha256(data).hexdigest(), "gzip_sha256": hashlib.sha256(encoded).hexdigest()}
    atomic_write(path, encoded)
    atomic_write(manifest_path, json.dumps(receipt, indent=2).encode())
    return receipt


def archive(site: str, directory: Path, workers: int) -> dict:
    bins = recording_bins(site)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(partial(fetch_bin, directory, site), bins))
    periods: list[list[dict]] = []
    for entry in files:
        if not periods or entry["first"] - periods[-1][-1]["last"] > 1000:
            periods.append([])
        periods[-1].append(entry)
    result = {
        "site": site, "fields": FIELDS, "total_rows": sum(entry["rows"] for entry in files),
        "periods": [
            {"first": period[0]["first"], "last": period[-1]["last"], "files": period}
            for period in periods
        ],
    }
    atomic_write(directory / "recordings.json", json.dumps(result, indent=2).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/crest/ngsim"))
    parser.add_argument("--site", nargs="+", choices=SITES, default=SITES)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    for site in args.site:
        result = archive(site, args.output / site, args.workers)
        print(f"{site}: {result['total_rows']} rows in {len(result['periods'])} recordings", flush=True)


if __name__ == "__main__":
    main()
