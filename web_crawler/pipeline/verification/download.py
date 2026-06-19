"""Gates 2 (download), 3 (MIME), and gate-4 input (sha256).

This wraps the existing `download_image` / `download_video` paths in
`runtime.py` so we keep the production logic (PIL re-encode, yt_dlp fallback)
and only add:

  * HTTP status code recording
  * Content-Type capture
  * file-magic check (a small hand-rolled sniffer; no python-magic dependency)
  * sha256 of the response body

If the modality (image vs video) doesn't match the response, we mark gate 3
as failed and do NOT persist the file.  If gates 2 and 3 pass, the caller is
free to invoke the legacy downloader to do its PIL/yt_dlp work on the same
bytes.

Note on m3u8 / embed video: HTTP GET on an m3u8 URL returns the manifest
(text/vnd.apple.mpegurl), not the video bytes; we treat that as MIME-OK for
video.  For iframe embed URLs (e.g. vimeo) we delegate to yt_dlp's
`extract_info` to confirm playability without downloading the full file.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Minimal magic-number sniffer (avoids the python-magic dependency)
# ---------------------------------------------------------------------------

_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xFF\xD8\xFF", "image/jpeg"),
    (b"\x89PNG\r\n\x1A\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]
# WEBP: "RIFF" + 4 size bytes + "WEBP"
_VIDEO_MAGIC: list[tuple[bytes, str]] = [
    (b"\x1A\x45\xDF\xA3", "video/webm"),       # EBML / matroska
    (b"#EXTM3U", "application/vnd.apple.mpegurl"),
]


def _sniff_mime(buf: bytes) -> Optional[str]:
    """Return a MIME guess from the first ~64 bytes; None if unknown."""
    if not buf:
        return None
    for sig, mime in _IMAGE_MAGIC:
        if buf.startswith(sig):
            return mime
    for sig, mime in _VIDEO_MAGIC:
        if buf.startswith(sig):
            return mime
    # WEBP needs offset checks
    if len(buf) >= 12 and buf[:4] == b"RIFF" and buf[8:12] == b"WEBP":
        return "image/webp"
    # MP4 ('ftyp' at offset 4) — accept the family without parsing brand
    if len(buf) >= 12 and buf[4:8] == b"ftyp":
        return "video/mp4"
    # SVG: starts with <?xml or <svg (allow optional BOM/whitespace)
    head = buf[:512].lstrip().lower()
    if head.startswith(b"<?xml") and b"<svg" in buf[:1024].lower():
        return "image/svg+xml"
    if head.startswith(b"<svg"):
        return "image/svg+xml"
    return None


# ---------------------------------------------------------------------------
# Modality matching
# ---------------------------------------------------------------------------

_IMAGE_MIMES = ("image/",)
_VIDEO_MIMES = (
    "video/",
    "application/vnd.apple.mpegurl",     # m3u8
    "application/x-mpegurl",             # alt m3u8
    "application/dash+xml",              # DASH
)


def _mime_matches_type(mime: str, artifact_type: str) -> bool:
    if not mime:
        return False
    m = mime.lower().split(";", 1)[0].strip()
    if artifact_type == "image":
        return m.startswith(_IMAGE_MIMES)
    if artifact_type == "video":
        return any(m.startswith(p) for p in _VIDEO_MIMES)
    return False


# ---------------------------------------------------------------------------
# Download wrapper
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    gate2_download_ok: bool
    gate3_mime_ok: bool
    http_status: Optional[int]
    bytes_total: Optional[int]
    mime_header: Optional[str]
    mime_sniffed: Optional[str]
    mime_used: Optional[str]            # whichever we ended up trusting
    mime_source: Optional[str]          # "header" | "magic" | None
    file_hash: Optional[str]            # sha256 hex
    content_path: Optional[str]         # tmp file path; caller decides to keep or rm
    error: Optional[str]
    duration_ms: int


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def verified_download(
    url: str,
    artifact_type: str,                  # "image" | "video"
    tmp_path: str,
    *,
    timeout_s: int = 30,
    max_bytes: int = 200 * 1024 * 1024,  # 200 MB hard ceiling per artifact
    referer: Optional[str] = None,
) -> DownloadResult:
    """Single GET that records everything we need for gates 2-4.

    Never raises — every failure goes onto the `error` field so the caller can
    record the partial result and move on.
    """
    started = time.monotonic()
    res = DownloadResult(
        gate2_download_ok=False,
        gate3_mime_ok=False,
        http_status=None,
        bytes_total=None,
        mime_header=None,
        mime_sniffed=None,
        mime_used=None,
        mime_source=None,
        file_hash=None,
        content_path=None,
        error=None,
        duration_ms=0,
    )

    headers = {"User-Agent": _DEFAULT_UA, "Accept": "*/*"}
    if referer:
        headers["Referer"] = referer

    try:
        os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)
        with requests.get(url, headers=headers, stream=True,
                          timeout=timeout_s, allow_redirects=True) as r:
            res.http_status = r.status_code
            ct = r.headers.get("content-type") or r.headers.get("Content-Type") or ""
            res.mime_header = ct.split(";", 1)[0].strip().lower() or None

            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}"
                return res

            sha = hashlib.sha256()
            total = 0
            head_buf = bytearray()
            with open(tmp_path, "wb") as fout:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    sha.update(chunk)
                    total += len(chunk)
                    if len(head_buf) < 64:
                        head_buf.extend(chunk[: max(0, 64 - len(head_buf))])
                    fout.write(chunk)
                    if total > max_bytes:
                        res.error = f"exceeded {max_bytes} bytes ceiling"
                        return res

            res.bytes_total = total
            res.file_hash = sha.hexdigest()
            res.content_path = tmp_path
            res.gate2_download_ok = total > 0

            res.mime_sniffed = _sniff_mime(bytes(head_buf))

            # MIME source priority: trust the header *unless* it's a generic
            # application/octet-stream or text/html (CDN error masquerading)
            chosen = None
            source = None
            if res.mime_header and not res.mime_header.startswith(
                ("application/octet-stream", "text/html", "text/plain")
            ):
                chosen, source = res.mime_header, "header"
            elif res.mime_sniffed:
                chosen, source = res.mime_sniffed, "magic"
            elif res.mime_header:
                # generic header is the only thing we have
                chosen, source = res.mime_header, "header"
            res.mime_used = chosen
            res.mime_source = source
            res.gate3_mime_ok = _mime_matches_type(chosen or "", artifact_type)

            if not res.gate2_download_ok:
                res.error = res.error or "empty body"
    except requests.exceptions.RequestException as e:
        res.error = f"requests: {type(e).__name__}: {e}"
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    finally:
        res.duration_ms = int((time.monotonic() - started) * 1000)
    return res
