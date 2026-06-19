"""Playwright-based page context capture (gate 1 raw material).

We deliberately do this with a *separate* Playwright session — not by tapping
into the browser-use Agent — for two reasons:

  1. The Agent in Step 2 abstracts the Playwright `page` object; reaching into
     its internals would couple us to a third-party library version.
  2. Capturing fresh DOM + network log is fast and deterministic; doing it
     after Step 2 keeps verification orthogonal to the LLM extraction.

What we capture for each source page:
  * Rendered HTML after `networkidle`           → `PageContext.html`
  * URLs whose responses had image/* or video/* Content-Type
                                                  → `PageContext.network_media_urls`

The HTML alone is enough for the vast majority of `<img>`/`<video>`/`<source>`
references; the network log catches assets that are injected after JS execution
but whose URL never appears verbatim in the DOM (e.g. background fetches,
some hero carousels).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


@dataclass
class PageContext:
    """Everything we need to decide whether an artifact URL was 'observed'."""
    requested_url: str
    final_url: str                              # after redirects
    status: Optional[int]
    html: str                                   # rendered DOM (page.content())
    network_media_urls: list[str] = field(default_factory=list)
    network_iframe_srcs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "status": self.status,
            "network_media_urls": self.network_media_urls,
            "network_iframe_srcs": self.network_iframe_srcs,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


async def capture_page_context(
    url: str,
    *,
    timeout_ms: int = 30_000,
    wait_until: str = "networkidle",
    user_agent: str = _DEFAULT_UA,
    headless: bool = True,
) -> PageContext:
    """Open `url` with Playwright; return DOM + network media URL log.

    Never raises — failures are recorded on the `PageContext.error` field so
    a single bad page doesn't poison the whole verification run.
    """
    import time
    started = time.monotonic()
    ctx = PageContext(
        requested_url=url,
        final_url=url,
        status=None,
        html="",
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            try:
                context = await browser.new_context(user_agent=user_agent)
                page = await context.new_page()

                # ---- network listeners ----
                def _on_response(response):
                    try:
                        ct = (response.headers or {}).get("content-type", "").lower()
                        if ct.startswith("image/") or ct.startswith("video/"):
                            ctx.network_media_urls.append(response.url)
                    except Exception:
                        pass

                def _on_frame_attached(frame):
                    try:
                        if frame.url and frame.url.startswith(("http://", "https://")):
                            ctx.network_iframe_srcs.append(frame.url)
                    except Exception:
                        pass

                page.on("response", _on_response)
                page.on("frameattached", _on_frame_attached)

                try:
                    resp = await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                    ctx.status = resp.status if resp else None
                    ctx.final_url = page.url
                except PlaywrightTimeoutError:
                    # Even on networkidle timeout, we usually still got a useful DOM.
                    ctx.error = f"networkidle timeout after {timeout_ms}ms"
                    ctx.final_url = page.url

                try:
                    ctx.html = await page.content()
                except Exception as e:
                    ctx.error = (ctx.error + "; " if ctx.error else "") + f"page.content failed: {e}"

                # also pull every <iframe src> from rendered DOM
                try:
                    iframe_srcs = await page.eval_on_selector_all(
                        "iframe[src]", "els => els.map(e => e.src)"
                    )
                    for s in iframe_srcs:
                        if s and s.startswith(("http://", "https://")):
                            ctx.network_iframe_srcs.append(s)
                except Exception:
                    pass

            finally:
                await browser.close()
    except Exception as e:
        ctx.error = (ctx.error + "; " if ctx.error else "") + f"playwright launch failed: {e}"

    # de-dupe while preserving order
    ctx.network_media_urls = list(dict.fromkeys(ctx.network_media_urls))
    ctx.network_iframe_srcs = list(dict.fromkeys(ctx.network_iframe_srcs))

    ctx.duration_ms = int((time.monotonic() - started) * 1000)
    return ctx


# ---------------------------------------------------------------------------
# Disk dump (so a verification run is reproducible offline)
# ---------------------------------------------------------------------------

def dump_page_context(ctx: PageContext, dom_path: str, network_path: str) -> None:
    """Write the rendered HTML and network metadata to disk."""
    os.makedirs(os.path.dirname(dom_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(network_path) or ".", exist_ok=True)
    with open(dom_path, "w", encoding="utf-8") as f:
        f.write(ctx.html or "")
    with open(network_path, "w", encoding="utf-8") as f:
        json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI for quick eyeballing
# ---------------------------------------------------------------------------

async def _amain(url: str):
    ctx = await capture_page_context(url)
    print(f"url      : {ctx.requested_url}")
    print(f"final    : {ctx.final_url}")
    print(f"status   : {ctx.status}")
    print(f"error    : {ctx.error}")
    print(f"html_len : {len(ctx.html)}")
    print(f"media#   : {len(ctx.network_media_urls)}")
    for u in ctx.network_media_urls[:10]:
        print(f"  - {u}")
    print(f"iframe#  : {len(ctx.network_iframe_srcs)}")
    for u in ctx.network_iframe_srcs[:5]:
        print(f"  - {u}")
    print(f"ms       : {ctx.duration_ms}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "https://www.berkeley.edu/"
    asyncio.run(_amain(target))
