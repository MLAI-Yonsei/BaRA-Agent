"""Artifact-unit verification module for BaRA.

Implements the 5-gate verification described in
`bara_artifact_verification_plan.md`:

  Gate 1. Was the artifact actually observed in the source page?
  Gate 2. Is it downloadable?
  Gate 3. Does the MIME type match the modality?
  Gate 4. Is the file hash not a duplicate?
  Gate 5. Is the hallucination risk low?           (derived from Gate 1 + Gate 2)

Each artifact produces an `ArtifactRecord` that is either INCLUDED in the
final dataset (all gates passed) or EXCLUDED with reasons.

Image and video are supported; text is intentionally out of scope (the
gates above don't carry the same meaning for text — that would require a
separate substring/paraphrase gate set).
"""

from web_crawler.pipeline.verification.schema import (
    ArtifactCandidate,
    ArtifactRecord,
    GateResult,
)
from web_crawler.pipeline.verification.store import (
    SeenHashes,
    append_record,
    write_summary,
)

__all__ = [
    "ArtifactCandidate",
    "ArtifactRecord",
    "GateResult",
    "SeenHashes",
    "append_record",
    "write_summary",
]
