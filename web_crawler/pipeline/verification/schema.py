"""Data classes for the verification module.

The record fields intentionally mirror Section 4.2 / Section 7 of
`bara_artifact_verification_plan.md` so that the JSONL files we emit can be
read back into the same shape without ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class ArtifactCandidate:
    """A candidate artifact extracted by Step 2.

    For ``image`` / ``video``: ``url`` is the actual media URL.
    For ``text``: ``url`` is a stable per-candidate identifier
    (e.g. ``text:<page_index>:<hash8>``) and ``text_content`` holds the raw
    string to be verified.

    ``source_page`` is the URL of the page the LLM/agent was reading when it
    produced this candidate; ``page_index`` is the same per-page index Step 2/3 use.
    """
    url: str
    type: str  # "image" | "video" | "text"
    source_page: str
    page_index: int
    text_content: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Per-gate verdict + optional evidence string."""
    passed: bool
    evidence: Optional[str] = None
    reason: Optional[str] = None  # only set when passed == False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Full per-artifact record
# ---------------------------------------------------------------------------

@dataclass
class ArtifactRecord:
    """Verification record for a single image/video URL."""

    # ---- identity ----
    artifact_url: str
    artifact_type: str           # "image" | "video"
    source_page: str
    page_index: int

    # ---- gate 1: observed in source page ----
    gate1_observed: bool = False
    observation_evidence: Optional[str] = None   # e.g. '<img src="...">'
    observation_channel: Optional[str] = None    # "dom" | "network" | None

    # ---- gate 2: download ----
    gate2_download_ok: bool = False
    http_status: Optional[int] = None
    download_bytes: Optional[int] = None

    # ---- gate 3: MIME ----
    gate3_mime_ok: bool = False
    mime_type: Optional[str] = None              # Content-Type or magic
    mime_source: Optional[str] = None            # "header" | "magic"

    # ---- gate 4: dedup ----
    file_hash: Optional[str] = None              # sha256 hex
    gate4_not_duplicate: bool = True             # True = not a duplicate
    duplicate_of: Optional[str] = None           # url of the first occurrence

    # ---- gate 5: hallucination (derived) ----
    gate5_not_hallucinated: bool = True
    hallucination_risk: str = "unknown"          # "low" | "high" | "unknown"

    # ---- text-specific (only set when artifact_type == "text") ----
    text_content: Optional[str] = None           # first ~200 chars of normalized text
    text_similarity: Optional[float] = None      # T1=1.0; T2 fuzzy Jaccard otherwise

    # ---- final ----
    final_decision: str = "exclude"              # "include" | "exclude"
    exclusion_reasons: list[str] = field(default_factory=list)
    saved_path: Optional[str] = None

    # ---- metadata ----
    verified_at: Optional[str] = None            # ISO timestamp
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    # convenience -----------------------------------------------------------

    def all_gates_passed(self) -> bool:
        # For text artifacts, gates 2/3/4 are no-ops (default True) and only
        # gate1_observed (T1 or T2) really matters; this expression still
        # evaluates correctly because the no-ops are True.
        return (
            self.gate1_observed
            and self.gate2_download_ok
            and self.gate3_mime_ok
            and self.gate4_not_duplicate
            and self.gate5_not_hallucinated
        )

    def short_repr(self) -> str:
        flags = "".join([
            "1" if self.gate1_observed else "·",
            "2" if self.gate2_download_ok else "·",
            "3" if self.gate3_mime_ok else "·",
            "4" if self.gate4_not_duplicate else "·",
            "5" if self.gate5_not_hallucinated else "·",
        ])
        verdict = "✓" if self.final_decision == "include" else "✗"
        return f"[{flags}] {verdict} {self.artifact_type} {self.artifact_url[:80]}"
