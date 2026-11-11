"""Verify frozen v4 inference artifacts match analysis_receipt.json witnesses."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "results/research_final_v4/analysis_receipt.json"
FILES = {
    "pilot_results.json": REPO / "results/research_final_v4/pilot_results.json",
    "pilot_metrics.csv": REPO / "results/research_final_v4/pilot_metrics.csv",
    "final_receipt.json": REPO / "results/research_final_v4/final_receipt.json",
    "pre_evaluation_freeze_v4.json": REPO / "results/research/pre_evaluation_freeze_v4.json",
    "registry": REPO / ".crest_research_protocol_registry.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt["immutable_input_sha256"]
    report: dict[str, object] = {
        "receipt_type": "crest_research_manuscript_immutable_verification",
        "analysis_receipt_path": str(RECEIPT.relative_to(REPO)),
        "inference_source_hash_expected": receipt["inference_source_hash"],
        "files": {},
        "all_match": True,
    }
    for key, path in FILES.items():
        digest = sha256_file(path)
        ok = digest == expected[key]
        report["files"][key] = {
            "path": str(path.relative_to(REPO)),
            "sha256": digest,
            "matches_receipt": ok,
        }
        if not ok:
            report["all_match"] = False
    out = REPO / "research_manuscript/generated/immutable_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
