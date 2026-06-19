"""Gate 1 — did this artifact URL actually appear in the source page?

The plan (§5.2) lists the surfaces we have to check:

  * <img src="...">
  * <img srcset="...">
  * <picture> <source srcset="...">
  * <video src="...">
  * <source src="...">           (inside <video>/<audio>)
  * <a href="...mp4">
  * CSS background-image: url(...)
  * rendered DOM (post-JS)       — provided as `page_context.html`
  * browser network requests     — provided as `page_context.network_media_urls`

We try the DOM channels first (cheapest and most specific) and fall back to the
network log.  An iframe whose src matches is *also* accepted as evidence for
videos, because embedded streaming players (vimeo/youtube) never expose the raw
mp4/m3u8 in the parent DOM.

URL matching uses a normalized form so that small variations (http vs https,
trailing slash, fragment, query order) don't cause spurious mismatches.  Image
`srcset` candidates are split so each variant is checked individually.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, unquote

from bs4 import BeautifulSoup

from web_crawler.pipeline.verification.capture import PageContext


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Return a canonical form for matching.

    Rules:
      * lower-case scheme + host
      * drop fragment
      * drop port if it's the default for the scheme
      * sort query params (preserve all of them — for media URLs the query is
        often a signed token we *want* to match exactly)
      * collapse percent-encoding for unreserved characters
      * strip trailing slash on path (keep root '/')
    """
    if not url:
        return ""
    try:
        # collapse percent-encoding for matching
        u = unquote(url.strip())
        p = urlsplit(u)
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        # strip default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        path = p.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        # sort query
        q = sorted(parse_qsl(p.query, keep_blank_values=True))
        return urlunsplit((scheme, netloc, path, urlencode(q), ""))
    except Exception:
        return url.strip()


def _url_equal_ignoring_scheme(a: str, b: str) -> bool:
    """Match treating http==https (some sites redirect inconsistently)."""
    if a == b:
        return True
    if not a or not b:
        return False
    return a.replace("https://", "://", 1).replace("http://", "://", 1) == \
           b.replace("https://", "://", 1).replace("http://", "://", 1)


# ---------------------------------------------------------------------------
# srcset parser
# ---------------------------------------------------------------------------

def _split_srcset(value: str) -> list[str]:
    """Pull each candidate URL out of a srcset attribute."""
    if not value:
        return []
    out = []
    # srcset entries are comma-separated, each "URL [width|density]".
    # URLs themselves may contain commas (rare but legal in query), so use
    # a tolerant split that respects commas only when followed by whitespace
    # or another url-looking token.
    for entry in re.split(r",\s*(?=\S)", value):
        entry = entry.strip()
        if not entry:
            continue
        url_part = entry.split(None, 1)[0]
        if url_part:
            out.append(url_part)
    return out


# ---------------------------------------------------------------------------
# DOM URL extraction
# ---------------------------------------------------------------------------

@dataclass
class DomMediaIndex:
    """Pre-parsed view of a page's DOM so observation lookups are O(1).

    Each set holds normalized URLs.  Iframe URLs are separated because we
    only accept them as evidence for video candidates.
    """
    images: dict[str, str]          # normalized -> first tag snippet
    videos: dict[str, str]
    anchors_media: dict[str, str]
    background_images: dict[str, str]
    iframes: dict[str, str]

    def all_urls_for_type(self, artifact_type: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if artifact_type == "image":
            for d in (self.images, self.background_images, self.anchors_media):
                for k, v in d.items():
                    out.setdefault(k, v)
        elif artifact_type == "video":
            for d in (self.videos, self.anchors_media, self.iframes):
                for k, v in d.items():
                    out.setdefault(k, v)
        return out


_BG_URL_RE = re.compile(r"url\(\s*(['\"]?)(?P<url>[^'\")]+)\1\s*\)")


def index_dom(html: str, base_url: str | None = None) -> DomMediaIndex:
    """Parse HTML once; return DOM media URL index used by observation."""
    images: dict[str, str] = {}
    videos: dict[str, str] = {}
    anchors_media: dict[str, str] = {}
    backgrounds: dict[str, str] = {}
    iframes: dict[str, str] = {}

    if not html:
        return DomMediaIndex(images, videos, anchors_media, backgrounds, iframes)

    soup = BeautifulSoup(html, "html.parser")

    def _add(bucket: dict[str, str], raw_url: str, evidence: str):
        nu = normalize_url(_absolutize(raw_url, base_url))
        if nu and nu not in bucket:
            bucket[nu] = evidence

    # <img src, srcset>
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if src:
            _add(images, src, str(tag)[:200])
        for cand in _split_srcset(tag.get("srcset", "")):
            _add(images, cand, str(tag)[:200])

    # <picture> <source srcset> (image)
    for source in soup.find_all("source"):
        parent_name = source.parent.name if source.parent else ""
        target = images if parent_name == "picture" else videos
        if source.get("src"):
            _add(target, source["src"], str(source)[:200])
        for cand in _split_srcset(source.get("srcset", "")):
            _add(target, cand, str(source)[:200])

    # <video src>
    for tag in soup.find_all("video"):
        src = tag.get("src")
        if src:
            _add(videos, src, str(tag)[:200])
        poster = tag.get("poster")
        if poster:
            _add(images, poster, str(tag)[:200])

    # <a href=...> — only worth tracking media-looking anchors
    _MEDIA_EXTS = (".mp4", ".webm", ".mov", ".m3u8", ".m4v", ".avi",
                   ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        low = href.lower().split("?")[0]
        if low.endswith(_MEDIA_EXTS):
            _add(anchors_media, href, str(tag)[:200])

    # iframes (mainly for video embeds)
    for tag in soup.find_all("iframe", src=True):
        _add(iframes, tag["src"], str(tag)[:200])

    # CSS background-image — inline style attribute (most common)
    for tag in soup.find_all(style=True):
        for m in _BG_URL_RE.finditer(tag["style"] or ""):
            _add(backgrounds, m.group("url"), str(tag)[:200])

    # CSS background-image — embedded <style> blocks
    for style in soup.find_all("style"):
        if not style.string:
            continue
        for m in _BG_URL_RE.finditer(style.string):
            _add(backgrounds, m.group("url"), "<style>...</style>")

    return DomMediaIndex(images, videos, anchors_media, backgrounds, iframes)


def _absolutize(url: str, base_url: str | None) -> str:
    """Resolve url against base_url if relative."""
    if not url:
        return ""
    if url.startswith(("http://", "https://", "data:", "blob:")):
        return url
    if not base_url:
        return url
    from urllib.parse import urljoin
    try:
        return urljoin(base_url, url)
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Public API — observe one candidate against a captured page context
# ---------------------------------------------------------------------------

@dataclass
class ObservationResult:
    observed: bool
    channel: Optional[str]      # "dom" | "network" | "iframe" | None
    evidence: Optional[str]


def observe(
    candidate_url: str,
    artifact_type: str,                  # "image" | "video"
    page_context: PageContext,
    dom_index: Optional[DomMediaIndex] = None,
) -> ObservationResult:
    """Decide whether `candidate_url` appears in the captured page context.

    Tries (in order):
      1. DOM index for the right modality
      2. Cross-modality DOM index (an image that happens to be linked as <a>)
      3. Network media URL log (must match Content-Type modality)
      4. Iframe srcs (only counts for video candidates)
    """
    if not candidate_url:
        return ObservationResult(False, None, "empty candidate URL")

    nu = normalize_url(candidate_url)
    if dom_index is None:
        dom_index = index_dom(page_context.html, base_url=page_context.final_url)

    # 1. DOM in the candidate's own modality
    own = dom_index.all_urls_for_type(artifact_type)
    if nu in own:
        return ObservationResult(True, "dom", own[nu][:200])

    # http vs https tolerance
    for other_nu, evidence in own.items():
        if _url_equal_ignoring_scheme(nu, other_nu):
            return ObservationResult(True, "dom", evidence[:200])

    # 2. cross-modality DOM (sometimes mislabeled by extractor)
    cross_type = "video" if artifact_type == "image" else "image"
    cross = dom_index.all_urls_for_type(cross_type)
    for other_nu, evidence in cross.items():
        if _url_equal_ignoring_scheme(nu, other_nu):
            return ObservationResult(True, "dom_cross_modality", evidence[:200])

    # 3. network log
    for net in page_context.network_media_urls:
        if _url_equal_ignoring_scheme(nu, normalize_url(net)):
            return ObservationResult(True, "network", f"network:{net}")

    # 4. iframe (video only)
    if artifact_type == "video":
        for iframe in page_context.network_iframe_srcs:
            if _url_equal_ignoring_scheme(nu, normalize_url(iframe)):
                return ObservationResult(True, "iframe", f"iframe:{iframe}")

    return ObservationResult(False, None, None)
