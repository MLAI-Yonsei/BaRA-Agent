"""Storage helpers for verification: SeenHashes + JSONL writer + summary."""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable

from web_crawler.pipeline.verification.schema import ArtifactRecord


# ---------------------------------------------------------------------------
# Per-site dedup set
# ---------------------------------------------------------------------------

class SeenHashes:
    """sha256-keyed dedup set; scoped per-site by the caller.

    A separate instance is created per site so the same stock photo on
    different sites is treated as a distinct artifact (intentional — see
    section 0 of the implementation plan).
    """

    def __init__(self):
        self._hashes: dict[str, str] = {}   # sha256 -> first URL that produced it

    def __contains__(self, h: str) -> bool:
        return h in self._hashes

    def first_url(self, h: str) -> str | None:
        return self._hashes.get(h)

    def add(self, h: str, url: str) -> None:
        if h and h not in self._hashes:
            self._hashes[h] = url

    def __len__(self) -> int:
        return len(self._hashes)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def append_record(record: ArtifactRecord, jsonl_path: str) -> None:
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_records(jsonl_path: str) -> list[ArtifactRecord]:
    if not os.path.exists(jsonl_path):
        return []
    out: list[ArtifactRecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ArtifactRecord(**json.loads(line)))
            except Exception:
                # unknown fields will be tolerated by callers reading dicts
                continue
    return out


# ---------------------------------------------------------------------------
# Summary statistics (Section 9.4 of the plan)
# ---------------------------------------------------------------------------

def summarize(records: Iterable[ArtifactRecord]) -> dict:
    records = list(records)
    n = len(records)
    by_type: dict[str, list[ArtifactRecord]] = {"image": [], "video": []}
    for r in records:
        by_type.setdefault(r.artifact_type, []).append(r)

    def block(rs: list[ArtifactRecord]) -> dict:
        total = len(rs)
        if total == 0:
            return {"total": 0}
        gate1 = sum(1 for r in rs if r.gate1_observed)
        gate2 = sum(1 for r in rs if r.gate2_download_ok)
        gate3 = sum(1 for r in rs if r.gate3_mime_ok)
        gate4 = sum(1 for r in rs if r.gate4_not_duplicate)
        gate5 = sum(1 for r in rs if r.gate5_not_hallucinated)
        included = sum(1 for r in rs if r.final_decision == "include")
        reasons: Counter = Counter()
        for r in rs:
            for reason in r.exclusion_reasons:
                reasons[reason] += 1
        return {
            "total": total,
            "gate1_observed_pass": gate1,
            "gate1_observed_rate": round(gate1 / total, 4),
            "gate2_download_pass": gate2,
            "gate2_download_rate": round(gate2 / total, 4),
            "gate3_mime_pass": gate3,
            "gate3_mime_rate": round(gate3 / total, 4),
            "gate4_dedup_pass": gate4,
            "gate4_dedup_rate": round(gate4 / total, 4),
            "gate5_not_halluc_pass": gate5,
            "gate5_not_halluc_rate": round(gate5 / total, 4),
            "final_included": included,
            "final_inclusion_rate": round(included / total, 4),
            "exclusion_reasons": dict(reasons.most_common()),
        }

    return {
        "total_candidates": n,
        "by_type": {t: block(rs) for t, rs in by_type.items()},
    }


def write_summary(records: Iterable[ArtifactRecord], summary_path: str) -> dict:
    summary = summarize(records)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
