"""5-gate orchestrator.

Given a list of `ArtifactCandidate`s, runs:

   Gate 1  →  observation.observe(...)            (against captured DOM/network)
   Gate 2  →  download.verified_download(...)     (HTTP fetch)
   Gate 3  →  download.verified_download(...)     (MIME)
   Gate 4  →  SeenHashes                          (sha256 dedup, site-scoped)
   Gate 5  →  derived: high if (¬gate1 ∧ ¬gate2)

Decisions:
   strict (default) — all 5 gates must pass to include
   audit            — include even if gate 5 is high or relevance is low
                       (we still emit the record and exclusion_reasons)
   collection-only  — ignore gate 1 and gate 5 entirely (used for post-hoc
                       evaluation on already-collected data where we have no
                       page context)

The orchestrator is intentionally synchronous-friendly: each candidate is a
simple HTTP fetch + a few in-memory checks.  We expose a coroutine wrapper
only for the Playwright capture stage, which is the only async-heavy work.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from web_crawler.pipeline.verification.capture import (
    PageContext,
    capture_page_context,
    dump_page_context,
)
from web_crawler.pipeline.verification.download import verified_download
from web_crawler.pipeline.verification.observation import (
    DomMediaIndex,
    index_dom,
    observe,
)
from web_crawler.pipeline.verification.schema import (
    ArtifactCandidate,
    ArtifactRecord,
)
from web_crawler.pipeline.verification.store import (
    SeenHashes,
    append_record,
    write_summary,
)
from web_crawler.pipeline.verification.text import (
    dom_visible_text,
    evaluate_text_candidate,
)


# ---------------------------------------------------------------------------
# Runner config
# ---------------------------------------------------------------------------

@dataclass
class VerificationConfig:
    mode: str = "strict"                 # "strict" | "audit" | "collection_only"
    output_dir: str = "verification_out"
    keep_downloaded_files: bool = True   # if False, included artifacts are still recorded but bytes are dropped
    download_timeout_image_s: int = 30   # generous for images (1-5 MB typical)
    download_timeout_video_s: int = 600  # 10 min — covers HD/long-form on slow connections
    download_max_bytes_image: int = 50 * 1024 * 1024            # 50 MB ceiling per image
    download_max_bytes_video: int = 2 * 1024 * 1024 * 1024      # 2 GB ceiling per video
    capture_timeout_ms: int = 60_000     # Playwright page load timeout (per source page)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_site_dir(config: VerificationConfig, site_root: str) -> str:
    return os.path.join(config.output_dir, site_root)


def _safe_filename(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    name = os.path.basename(p.path) or "artifact"
    name = name.split("?", 1)[0].split("#", 1)[0]
    # strip dangerous characters
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return name[:128] or "artifact"


def _ext_for(mime: str | None, artifact_type: str) -> str:
    if mime:
        m = mime.lower()
        if m.startswith("image/jpeg"):    return ".jpg"
        if m.startswith("image/png"):     return ".png"
        if m.startswith("image/gif"):     return ".gif"
        if m.startswith("image/webp"):    return ".webp"
        if m.startswith("image/svg"):     return ".svg"
        if m.startswith("image/"):        return ".img"
        if m.startswith("video/mp4"):     return ".mp4"
        if m.startswith("video/webm"):    return ".webm"
        if m.startswith("application/vnd.apple.mpegurl"): return ".m3u8"
        if m.startswith("video/"):        return ".vid"
    return ".bin" if artifact_type == "image" else ".vid"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def verify_candidates_for_page(
    candidates: list[ArtifactCandidate],
    *,
    page_context: PageContext | None,
    seen_hashes: SeenHashes,
    config: VerificationConfig,
    site_root: str,
) -> list[ArtifactRecord]:
    """Verify every candidate associated with a single source page.

    If `page_context` is None we run in collection-only mode automatically
    for these candidates (gate 1 and gate 5 are skipped/derived).
    """
    records: list[ArtifactRecord] = []
    if not candidates:
        return records

    dom_index: Optional[DomMediaIndex] = None
    dom_text_norm: Optional[str] = None
    if page_context is not None:
        dom_index = index_dom(page_context.html, base_url=page_context.final_url)
        dom_text_norm = dom_visible_text(page_context.html)

    site_dir = _per_site_dir(config, site_root)
    artifacts_dir = os.path.join(site_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    jsonl_path = os.path.join(site_dir, "verification_records.jsonl")

    for cand in candidates:
        if cand.type == "text":
            rec = _verify_one_text(
                cand=cand,
                page_context=page_context,
                dom_text_norm=dom_text_norm,
                config=config,
                artifacts_dir=artifacts_dir,
            )
        else:
            rec = await _verify_one(
                cand=cand,
                page_context=page_context,
                dom_index=dom_index,
                seen_hashes=seen_hashes,
                config=config,
                artifacts_dir=artifacts_dir,
            )
        append_record(rec, jsonl_path)
        records.append(rec)
        print(f"  {rec.short_repr()}")

    return records


def _verify_one_text(
    *,
    cand: ArtifactCandidate,
    page_context: PageContext | None,
    dom_text_norm: Optional[str],
    config: VerificationConfig,
    artifacts_dir: str,
) -> ArtifactRecord:
    """Run text verification (T1 + T2 only) and pack into ArtifactRecord.

    T3 (boilerplate) and T4 (dedup) were dropped — they over-filtered
    legitimate content.  Only T1 (DOM substring) and T2 (paraphrase Jaccard
    ≥ 0.1) gate inclusion.  No boilerplate flag or text hash is recorded.
    """
    started = time.monotonic()
    text = cand.text_content or cand.url   # candidate text is carried in .text_content; .url is a stable id
    rec = ArtifactRecord(
        artifact_url=cand.url,
        artifact_type="text",
        source_page=cand.source_page,
        page_index=cand.page_index,
        verified_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        # text has no download / MIME / dedup step — those gates stay at their schema defaults
        gate2_download_ok=True,
        gate3_mime_ok=True,
        mime_type="text/plain",
    )
    reasons: list[str] = []

    outcome = evaluate_text_candidate(
        text=text,
        page_context=page_context,
        cached_dom_text_norm=dom_text_norm,
    )

    # Pack T1 + T2 results into shared ArtifactRecord schema
    rec.gate1_observed = outcome.observed
    rec.observation_channel = outcome.observation_channel
    rec.observation_evidence = outcome.observation_evidence
    rec.text_similarity = outcome.similarity
    rec.text_content = text[:200] if text else None
    rec.download_bytes = outcome.char_count
    rec.gate5_not_hallucinated = outcome.observed
    rec.hallucination_risk = "low" if outcome.observed else "high"

    # Text inclusion policy: T1 + T2 ONLY.
    if not outcome.observed:
        reasons.append("text not found in source DOM (hallucinated or paraphrased)")

    if config.mode == "collection_only":
        include = True   # no DOM context — accept everything
    else:                # "strict" / "audit" — both now mean "T1+T2 only"
        include = outcome.observed

    if include:
        rec.final_decision = "include"
        rec.exclusion_reasons = []
        # Persist passed text to artifacts/texts/included_texts.jsonl (one per line, full body)
        if config.keep_downloaded_files and text:
            text_dir = os.path.join(artifacts_dir, "texts")
            os.makedirs(text_dir, exist_ok=True)
            text_file = os.path.join(text_dir, "included_texts.jsonl")
            try:
                import json as _json
                with open(text_file, "a", encoding="utf-8") as ftx:
                    ftx.write(_json.dumps({
                        "candidate_id": cand.url,
                        "text": text,
                        "source_page": cand.source_page,
                        "page_index": cand.page_index,
                        "char_count": outcome.char_count,
                        "word_count": outcome.word_count,
                        "observation_channel": outcome.observation_channel,
                        "text_similarity": outcome.similarity,
                        "verified_at": rec.verified_at,
                    }, ensure_ascii=False) + "\n")
                rec.saved_path = text_file
            except Exception as e:
                rec.exclusion_reasons.append(f"text save failed: {e}")
    else:
        rec.final_decision = "exclude"
        rec.exclusion_reasons = reasons or ["unknown"]

    rec.duration_ms = int((time.monotonic() - started) * 1000)
    return rec


async def _verify_one(
    *,
    cand: ArtifactCandidate,
    page_context: PageContext | None,
    dom_index: Optional[DomMediaIndex],
    seen_hashes: SeenHashes,
    config: VerificationConfig,
    artifacts_dir: str,
) -> ArtifactRecord:
    started = time.monotonic()
    rec = ArtifactRecord(
        artifact_url=cand.url,
        artifact_type=cand.type,
        source_page=cand.source_page,
        page_index=cand.page_index,
        verified_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    )
    reasons: list[str] = []

    # ---- Gate 1 ----
    if config.mode == "collection_only" or page_context is None:
        rec.gate1_observed = True
        rec.observation_channel = "skipped"
        rec.observation_evidence = "collection_only mode — gate 1 skipped"
    else:
        obs = observe(cand.url, cand.type, page_context, dom_index)
        rec.gate1_observed = obs.observed
        rec.observation_channel = obs.channel
        rec.observation_evidence = obs.evidence
        if not obs.observed:
            reasons.append("not observed in source page")

    # ---- Gates 2 + 3 + sha256 ----
    if cand.type == "video":
        timeout_s = config.download_timeout_video_s
        max_bytes = config.download_max_bytes_video
    else:
        timeout_s = config.download_timeout_image_s
        max_bytes = config.download_max_bytes_image
    tmp_path = os.path.join(artifacts_dir, "_pending_" + _safe_filename(cand.url))
    dl = verified_download(
        cand.url, cand.type, tmp_path,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        referer=cand.source_page,
    )
    rec.gate2_download_ok = dl.gate2_download_ok
    rec.http_status = dl.http_status
    rec.download_bytes = dl.bytes_total
    rec.mime_type = dl.mime_used
    rec.mime_source = dl.mime_source
    rec.gate3_mime_ok = dl.gate3_mime_ok
    rec.file_hash = dl.file_hash

    if not dl.gate2_download_ok:
        reasons.append("download failed" + (f" ({dl.error})" if dl.error else ""))
    if dl.gate2_download_ok and not dl.gate3_mime_ok:
        reasons.append(f"MIME type does not match artifact type (got {dl.mime_used})")

    # ---- Gate 4 (dedup) ----
    if rec.file_hash and rec.file_hash in seen_hashes:
        rec.gate4_not_duplicate = False
        rec.duplicate_of = seen_hashes.first_url(rec.file_hash)
        reasons.append("duplicate artifact (hash match)")
    else:
        rec.gate4_not_duplicate = True

    # ---- Gate 5 (derived hallucination signal) ----
    if config.mode == "collection_only":
        rec.gate5_not_hallucinated = True
        rec.hallucination_risk = "skipped"
    else:
        if (not rec.gate1_observed) and (not rec.gate2_download_ok):
            rec.gate5_not_hallucinated = False
            rec.hallucination_risk = "high"
            reasons.append("high hallucination risk (not in DOM, not in network, download failed)")
        else:
            rec.gate5_not_hallucinated = True
            rec.hallucination_risk = "low"

    # ---- Final decision ----
    if config.mode == "strict":
        include = rec.all_gates_passed()
    elif config.mode == "audit":
        # ignore gate 5 only; everything else must pass
        include = (rec.gate1_observed and rec.gate2_download_ok and
                   rec.gate3_mime_ok and rec.gate4_not_duplicate)
    elif config.mode == "collection_only":
        include = (rec.gate2_download_ok and rec.gate3_mime_ok
                   and rec.gate4_not_duplicate)
    else:
        include = False

    if include:
        rec.final_decision = "include"
        rec.exclusion_reasons = []
        # bump dedup state only for actually included artifacts
        if rec.file_hash:
            seen_hashes.add(rec.file_hash, rec.artifact_url)
        # promote tmp file to a stable name, separating image vs video subdir
        if config.keep_downloaded_files and dl.content_path and os.path.exists(dl.content_path):
            ext = _ext_for(rec.mime_type, rec.artifact_type)
            final_name = _safe_filename(cand.url)
            if not final_name.lower().endswith(ext.lower()):
                final_name += ext
            type_subdir = "images" if rec.artifact_type == "image" else "videos"
            type_dir = os.path.join(artifacts_dir, type_subdir)
            os.makedirs(type_dir, exist_ok=True)
            final_path = os.path.join(type_dir, final_name)
            try:
                shutil.move(dl.content_path, final_path)
                rec.saved_path = final_path
            except Exception as e:
                rec.saved_path = None
                rec.exclusion_reasons.append(f"file save failed: {e}")
        else:
            # not keeping bytes; still record the result
            try:
                if dl.content_path and os.path.exists(dl.content_path):
                    os.remove(dl.content_path)
            except Exception:
                pass
    else:
        rec.final_decision = "exclude"
        rec.exclusion_reasons = reasons or ["unknown"]
        # discard the tmp body
        try:
            if dl.content_path and os.path.exists(dl.content_path):
                os.remove(dl.content_path)
        except Exception:
            pass

    rec.duration_ms = int((time.monotonic() - started) * 1000)
    return rec


# ---------------------------------------------------------------------------
# High-level driver: capture page → verify candidates
# ---------------------------------------------------------------------------

async def verify_page(
    source_page: str,
    page_index: int,
    candidates_for_page: list[ArtifactCandidate],
    *,
    seen_hashes: SeenHashes,
    config: VerificationConfig,
    site_root: str,
) -> tuple[PageContext | None, list[ArtifactRecord]]:
    """Capture the source page once, then verify all candidates from it."""
    site_dir = _per_site_dir(config, site_root)
    ctx: Optional[PageContext] = None

    if config.mode != "collection_only":
        ctx = await capture_page_context(
            source_page, timeout_ms=config.capture_timeout_ms
        )
        dom_path = os.path.join(site_dir, "dom_dump", f"page_{page_index}.html")
        net_path = os.path.join(site_dir, "network_dump", f"page_{page_index}.json")
        dump_page_context(ctx, dom_path, net_path)

    recs = await verify_candidates_for_page(
        candidates_for_page,
        page_context=ctx,
        seen_hashes=seen_hashes,
        config=config,
        site_root=site_root,
    )
    return ctx, recs


async def verify_site(
    candidates: Iterable[ArtifactCandidate],
    *,
    config: VerificationConfig,
    site_root: str,
) -> dict:
    """Verify every candidate for a single site; write summary; return it."""
    by_page: dict[tuple[str, int], list[ArtifactCandidate]] = {}
    for c in candidates:
        by_page.setdefault((c.source_page, c.page_index), []).append(c)

    seen = SeenHashes()
    all_records: list[ArtifactRecord] = []
    for (page, page_idx), cands in sorted(by_page.items(), key=lambda kv: kv[0][1]):
        print(f"\n=== page_{page_idx}  {page}  ({len(cands)} candidates) ===")
        _, recs = await verify_page(
            page, page_idx, cands,
            seen_hashes=seen, config=config, site_root=site_root,
        )
        all_records.extend(recs)

    summary_path = os.path.join(_per_site_dir(config, site_root), "verification_summary.json")
    summary = write_summary(all_records, summary_path)
    return summary
