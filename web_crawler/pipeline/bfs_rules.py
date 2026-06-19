"""Pure-Python BFS rule helpers for the Rule BFS Step 1 default.

The LLM worker is responsible only for fetching a rendered page and returning
every anchor it observes. Filtering, normalization, deduplication, and
queue management all live here so they are deterministic and inspectable.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, urljoin


# Keys preserved on the query string after normalization. Same set as the
# legacy inline BFS prompt's rule (3).
_KEEP_QUERY_KEYS = frozenset({
    "page", "p", "pg", "offset", "start", "sort", "category", "board",
})

# Path substrings that disqualify a URL outright.
_EXCLUDED_PATH_PARTS = ("/privacy", "/terms", "/login", "/signup")

# File extensions that disqualify a URL outright.
# Kept in sync with ablation/prompts/general_prompt.py::_GENERAL_BASE so the
# LLM worker and the code filter agree on what is "not a browsable page".
# Note: .html is intentionally NOT excluded here even though VIDEO_EXT in
# runtime.py lists it (that's a Step 2 streaming-embed convention, not a
# BFS rule — BFS must follow .html pages).
_EXCLUDED_EXTENSIONS = (
    # Archive / binary
    ".pdf", ".zip", ".rar", ".7z", ".apk", ".dmg", ".exe",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    # Video / streaming
    ".mp4", ".mov", ".avi", ".wmv", ".flv", ".mkv", ".webm", ".m4v",
    ".m3u8", ".ts",
)


def _registrable_suffix(host: str) -> str:
    """Return a coarse registrable-domain proxy for a hostname.

    This is intentionally simple (last two labels) — sufficient to keep
    same-site BFS within e.g. ``blog.example.com`` <-> ``www.example.com``
    while rejecting unrelated hosts. We do not depend on a PSL library so
    multi-label TLDs (``.co.uk``, ``.com.au``) are slightly conservative.
    """
    host = (host or "").lower().strip()
    if not host:
        return ""
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


def normalize(url: str, base: Optional[str] = None) -> Optional[str]:
    """Normalize an anchor href to a canonical, comparable form.

    - Resolves relative URLs against ``base`` when provided.
    - Strips fragments.
    - Drops query keys not in the whitelist.
    - Treats http/https and trailing slashes as equivalent (scheme lowered,
      single trailing slash preserved only when path is empty).
    - Lowercases scheme and host.

    Returns ``None`` for non-http(s) schemes (mailto, javascript, tel, ...).
    """
    if not url:
        return None
    raw = url.strip()
    if not raw:
        return None
    # Reject HTML template placeholders like {{PARENT_URL}} or {slug} that
    # survive in static pages — urljoin would otherwise accept them as
    # relative paths and emit URLs that 404 on visit.
    if "{" in raw or "}" in raw:
        return None
    if base:
        raw = urljoin(base, raw)

    parts = urlsplit(raw)
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept = [
        (k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=False)
        if k in _KEEP_QUERY_KEYS
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, netloc, path, query, ""))


def same_site(url: str, start_url: str) -> bool:
    """True when ``url`` shares the registrable domain (or a subdomain) with ``start_url``."""
    u_host = (urlsplit(url).hostname or "").lower()
    s_host = (urlsplit(start_url).hostname or "").lower()
    if not u_host or not s_host:
        return False
    return _registrable_suffix(u_host) == _registrable_suffix(s_host)


def excluded(url: str) -> bool:
    """True when ``url`` matches the exclude rules (boilerplate paths / binary files)."""
    parts = urlsplit(url)
    path = (parts.path or "").lower()
    if any(part in path for part in _EXCLUDED_PATH_PARTS):
        return True
    if any(path.endswith(ext) for ext in _EXCLUDED_EXTENSIONS):
        return True
    return False


def filter_children(
    raw_anchors: Iterable[str],
    parent_url: str,
    start_url: str,
    max_width: int,
    already_seen: Optional[Set[str]] = None,
    resolve=None,
) -> tuple[List[str], List[str]]:
    """Apply the full child-selection pipeline for one parent page.

    Order: normalize → same_site → not excluded → drop URLs already in
    ``already_seen`` (visited or queued elsewhere) → dedup within this
    parent → (optional) liveness check → take first ``max_width`` live.

    The ``already_seen`` exclusion is what lets BFS progress past a shared
    site navigation: if every page on the site starts with the same nav
    menu, those URLs are visited / queued after the first parent, so on
    subsequent parents they don't consume the width budget and the
    *next* unique anchors get picked up instead.

    ``resolve``, when given, is ``resolve(url) -> (alive: bool, final_url: str)``.
    Dead candidates are skipped so the width budget counts only live links,
    and a live candidate is stored as its ``final_url`` (a 3xx redirect is
    followed to its canonical target). With no ``resolve`` the liveness step
    is skipped and the returned dead list is empty.

    Returns ``(live_children, dead_links)``.
    """
    already_seen = already_seen or set()
    seen = set()
    out: List[str] = []
    dead: List[str] = []
    for raw in raw_anchors:
        norm = normalize(raw, base=parent_url)
        if norm is None:
            continue
        if not same_site(norm, start_url):
            continue
        if excluded(norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        if norm in already_seen:
            continue
        url = norm
        if resolve is not None:
            alive, final_url = resolve(norm)
            if not alive:
                dead.append(norm)
                continue
            url = final_url
        if url in out or url in already_seen:
            continue
        out.append(url)
        if len(out) >= max_width:
            break
    return out, dead
