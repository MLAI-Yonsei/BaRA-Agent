"""Text-artifact verification (T1 + T2 only, LLM-free).

Image and video share a single "is this URL real" theme: source observation +
downloadability + MIME + dedup + hallucination.  Text is structurally
different — it's not a URL, it's a substring of the page.  So the gates are
re-shaped:

    T1. observed     — the text appears verbatim in the captured DOM
                       (substring match after normalization).
    T2. not paraphrased — when T1 fails, the text is *still* close to some
                       region of the DOM (token-Jaccard ≥ threshold = 0.1).
                       If both T1 and T2 fail → hallucination.

T3 (boilerplate heuristic) and T4 (text dedup) were dropped because they
over-filtered legitimate content (short titles, repeated category labels,
news section headers).  Only T1 + T2 drive the include/exclude decision and
nothing else is recorded for text artifacts.

LLM is not used anywhere.  All checks are deterministic Python + BeautifulSoup.
"""
from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

from web_crawler.pipeline.verification.capture import PageContext


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase + decode entities + collapse whitespace + trim.

    The same function is applied to both the candidate text and the DOM text
    so observation checks compare apples to apples.
    """
    if not s:
        return ""
    s = html_mod.unescape(s)
    s = s.replace(" ", " ")          # NBSP
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


# ---------------------------------------------------------------------------
# DOM → plain text extraction (run once per page, cache it)
# ---------------------------------------------------------------------------

def dom_visible_text(html: str) -> str:
    """Strip script/style/noscript and return concatenated visible text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return normalize_text(soup.get_text(separator=" "))


# ---------------------------------------------------------------------------
# Gate T1 + T2 — observation / paraphrase fallback
# ---------------------------------------------------------------------------

@dataclass
class TextObservation:
    observed: bool
    channel: Optional[str]        # "dom_exact" | "dom_fuzzy" | None
    similarity: float             # 1.0 if exact; otherwise the best Jaccard
    evidence_snippet: Optional[str]


def _token_set(s: str) -> set[str]:
    return set(t for t in s.split() if t)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def observe_text(
    candidate_text: str,
    dom_text_norm: str,
    fuzzy_threshold: float = 0.1,
) -> TextObservation:
    """Try exact substring first; fall back to token Jaccard for paraphrase.

    Default fuzzy_threshold is 0.1 — lenient enough to keep LLM paraphrases
    (sim 0.4-0.7 cluster) while still requiring at least minimal token overlap
    (so pure hallucinations stay out)."""
    norm_cand = normalize_text(candidate_text)
    if not norm_cand:
        return TextObservation(False, None, 0.0, None)

    # T1 — exact substring (most expressive guarantee)
    if norm_cand in dom_text_norm:
        return TextObservation(True, "dom_exact", 1.0,
                               norm_cand[:160])

    # T2 — sliding-window Jaccard fallback.  We avoid O(N*M) by chunking the
    # DOM into sentence-ish spans of similar length and comparing token sets.
    cand_tokens = _token_set(norm_cand)
    if len(cand_tokens) < 3:
        # too short for fuzzy matching to be meaningful
        return TextObservation(False, None, 0.0, None)

    # Build windows over the DOM text whose token count brackets the candidate.
    dom_tokens = dom_text_norm.split()
    n_cand = len(cand_tokens)
    window = max(8, int(n_cand * 1.5))
    step = max(4, window // 2)

    best_sim = 0.0
    best_span: Optional[str] = None
    for start in range(0, max(1, len(dom_tokens) - window + 1), step):
        win = set(dom_tokens[start: start + window])
        sim = _jaccard(cand_tokens, win)
        if sim > best_sim:
            best_sim = sim
            if sim >= fuzzy_threshold:
                best_span = " ".join(dom_tokens[start: start + window])[:160]
                break

    if best_sim >= fuzzy_threshold and best_span is not None:
        return TextObservation(True, "dom_fuzzy", round(best_sim, 3), best_span)

    return TextObservation(False, None, round(best_sim, 3), None)


# ---------------------------------------------------------------------------
# Public — collapse the T1 + T2 gates into the shared ArtifactRecord fields
# ---------------------------------------------------------------------------

@dataclass
class TextGateOutcome:
    """Per-candidate outcome packaged for the runner.

    Mapped onto the shared ArtifactRecord like this:
      gate1_observed         ← T1 or T2 succeeded
      observation_channel    ← "dom_exact" / "dom_fuzzy" / None
      observation_evidence   ← matched span
      gate2_download_ok      ← True (text has no download step)
      gate3_mime_ok          ← True (text has no MIME)
      gate4_not_duplicate    ← True (no dedup for text)
      gate5_not_hallucinated ← True iff observed
      text_content           ← first 200 chars of normalized text
      download_bytes         ← character count (repurposed)
      mime_type              ← "text/plain"
    """
    observed: bool
    observation_channel: Optional[str]
    observation_evidence: Optional[str]
    similarity: float
    char_count: int
    word_count: int


def evaluate_text_candidate(
    text: str,
    page_context: PageContext | None,
    cached_dom_text_norm: Optional[str] = None,
    fuzzy_threshold: float = 0.1,
) -> TextGateOutcome:
    """Run T1 + T2 on a single candidate string."""
    if page_context is not None and cached_dom_text_norm is None:
        cached_dom_text_norm = dom_visible_text(page_context.html)

    norm_text = normalize_text(text)
    words = norm_text.split()

    if page_context is None:
        # collection_only mode: skip observation
        obs = TextObservation(True, "skipped", 0.0, None)
    else:
        obs = observe_text(text, cached_dom_text_norm or "", fuzzy_threshold)

    return TextGateOutcome(
        observed=obs.observed,
        observation_channel=obs.channel,
        observation_evidence=obs.evidence_snippet,
        similarity=obs.similarity,
        char_count=len(text or ""),
        word_count=len(words),
    )
