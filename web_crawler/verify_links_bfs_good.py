#!/usr/bin/env python3
"""Verify URLs from links_bfs.json files WITHOUT underestimating bot-protected sites.

Why this exists
---------------
`verify_links_bfs.py` checks every URL with a bare `requests.get()`. WAFs/bot
detection (Cloudflare et al.) block that client via TLS fingerprint (JA3),
missing browser headers, and JavaScript challenges — returning 403 even though
the link is perfectly alive in a real browser. BaRA itself collects links with a
real headless Chromium (browser_use / Playwright), so a `requests`-based grader
*underestimates* BaRA: it marks links dead that BaRA legitimately reached.

This script grades links the same way BaRA reaches them, so the annotation
ground truth is fair:

  Tier 1 (fast)  : requests.get() — cheap, handles the vast majority of links.
  Tier 2 (fair)  : if Tier 1 is NOT alive, retry with a real headless Chromium
                   (Playwright). A real browser has a genuine TLS fingerprint,
                   full ordered headers, and runs JS challenges — so it passes
                   the same bot checks BaRA passes. The browser verdict wins.

Output schema is identical to verify_links_bfs.py (files[].{path,total,alive,
tally,results[]}, grand_total, grand_alive, grand_tally) so downstream
micro/macro aggregation works unchanged. Each result additionally carries a
`method` field ("http" or "browser") so you can see which links only survived
because they were checked with a real browser.

Usage:
    python -m web_crawler.verify_links_bfs_good data/step1_runs/ --output report.json
    python -m web_crawler.verify_links_bfs_good a/links_bfs.json b/links_bfs.json \\
        --http-concurrency 16 --browser-concurrency 4 --output report.json

    # Skip the browser tier entirely (behaves like the old script):
    python -m web_crawler.verify_links_bfs_good path/ --no-browser-fallback

    # Send everything straight to the browser (most faithful, slowest):
    python -m web_crawler.verify_links_bfs_good path/ --browser-only
"""

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import requests
    from urllib3.exceptions import InsecureRequestWarning
except ImportError:
    print("❌ This script requires `requests` (pip install requests)", file=sys.stderr)
    sys.exit(2)

try:
    from playwright.async_api import async_playwright
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeout
except ImportError:
    async_playwright = None  # browser tier unavailable; --no-browser-fallback still works

warnings.simplefilter("ignore", InsecureRequestWarning)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Chromium launch args that drop the most obvious automation tells. The first
# one stops `navigator.webdriver` from being forced true and removes the
# "Chrome is being controlled by automated software" surface that many WAFs
# fingerprint on.
STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
]

# Injected before any page script runs — masks the residual headless tells that
# bot detectors probe (webdriver flag, empty plugins, missing chrome runtime,
# languages). Cheap and well-understood; not a full anti-detect suite.
STEALTH_INIT_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || {runtime: {}};
const _query = window.navigator.permissions && window.navigator.permissions.query;
if (_query) {
  window.navigator.permissions.query = (p) => (
    p && p.name === 'notifications'
      ? Promise.resolve({state: Notification.permission})
      : _query(p)
  );
}
"""


@dataclass
class CheckResult:
    url: str
    status: int = 0
    final_url: str = ""
    elapsed_ms: int = 0
    category: str = ""
    error: str = ""
    method: str = ""  # "http" | "browser" — how the verdict was reached


def classify(status: int) -> str:
    if 200 <= status < 400:
        return "alive"
    if 400 <= status < 500:
        return "client_error"
    if 500 <= status < 600:
        return "server_error"
    return "unknown"


def collect_urls(links_bfs_path: str) -> list[str]:
    """Union of visited_order and by_depth URLs, dedup, original order preserved."""
    with open(links_bfs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen: set[str] = set()
    urls: list[str] = []
    for u in data.get("visited_order") or []:
        if isinstance(u, str) and u and u not in seen:
            seen.add(u)
            urls.append(u)
    for _depth_key, depth_urls in (data.get("by_depth") or {}).items():
        for u in depth_urls or []:
            if isinstance(u, str) and u and u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def find_links_files(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if os.path.isfile(p):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f == "links_bfs.json":
                        ap = os.path.abspath(os.path.join(root, f))
                        if ap not in seen:
                            seen.add(ap)
                            out.append(os.path.join(root, f))
        else:
            print(f"⚠️  not found: {p}", file=sys.stderr)
    return out


# ---------------------------------------------------------------- Tier 1: HTTP
def http_check(url: str, timeout: int, verify_tls: bool) -> CheckResult:
    headers = {"User-Agent": UA, "Accept": "*/*"}
    result = CheckResult(url=url, method="http")
    t0 = time.time()
    try:
        r = requests.get(
            url, timeout=timeout, allow_redirects=True,
            headers=headers, stream=True, verify=verify_tls,
        )
        result.status = r.status_code
        result.final_url = r.url
        r.close()
    except requests.exceptions.Timeout as e:
        result.category = "timeout"
        result.error = type(e).__name__
    except requests.exceptions.SSLError as e:
        result.category = "conn_error"
        result.error = f"SSL: {type(e).__name__}"
    except requests.exceptions.ConnectionError as e:
        result.category = "conn_error"
        result.error = type(e).__name__
    except requests.exceptions.RequestException as e:
        result.category = "request_error"
        result.error = type(e).__name__
    finally:
        result.elapsed_ms = int((time.time() - t0) * 1000)
    if not result.category:
        result.category = classify(result.status)
    return result


# ------------------------------------------------------------ Tier 2: Browser
async def browser_check(context, url: str, timeout_ms: int) -> CheckResult:
    """Navigate with a real Chromium page; capture the final main-frame status.

    Captures the *last* main-frame navigation response so JS/Cloudflare
    challenges (which navigate again after solving) report their resolved
    status, not the interstitial.
    """
    result = CheckResult(url=url, method="browser")
    page = await context.new_page()
    last = {"status": 0, "url": url}

    def on_response(resp):
        try:
            req = resp.request
            if req.is_navigation_request() and resp.frame == page.main_frame:
                last["status"] = resp.status
                last["url"] = resp.url
        except Exception:
            pass

    page.on("response", on_response)
    t0 = time.time()
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Give JS challenges a chance to resolve; tolerate networkidle never firing.
        try:
            await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
        except PlaywrightTimeout:
            pass
        result.status = last["status"] or (resp.status if resp else 0)
        result.final_url = page.url or last["url"]
    except PlaywrightTimeout as e:
        result.category = "timeout"
        result.error = type(e).__name__
    except PlaywrightError as e:
        result.category = "conn_error"
        result.error = (type(e).__name__ + ": " + str(e).splitlines()[0])[:80]
    except Exception as e:  # noqa: BLE001
        result.category = "conn_error"
        result.error = type(e).__name__
    finally:
        result.elapsed_ms = int((time.time() - t0) * 1000)
        try:
            await page.close()
        except Exception:
            pass
    if not result.category:
        result.category = classify(result.status)
    return result


def _fmt_url(url: str, max_len: int = 80) -> str:
    return url if len(url) <= max_len else url[: max_len - 1] + "…"


async def verify_all_files(files: list[str], args) -> list[dict]:
    use_browser = (not args.no_browser_fallback) and async_playwright is not None
    if not args.no_browser_fallback and async_playwright is None:
        print("⚠️  playwright not installed → browser tier disabled; "
              "install with `pip install playwright && playwright install chromium`",
              file=sys.stderr)

    # Pre-load each file's URLs.
    file_urls = [(path, collect_urls(path)) for path in files]
    http_pool = ThreadPoolExecutor(max_workers=args.http_concurrency)
    loop = asyncio.get_event_loop()

    browser = context = pw = None
    browser_sem = asyncio.Semaphore(args.browser_concurrency)
    if use_browser:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(
            headless=not args.headful,
            args=STEALTH_LAUNCH_ARGS,
        )
        context = await browser.new_context(
            user_agent=None,  # keep Chromium's genuine UA + client hints
            ignore_https_errors=args.insecure,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        await context.add_init_script(STEALTH_INIT_JS)

    async def http_async(url: str) -> CheckResult:
        return await loop.run_in_executor(
            http_pool, http_check, url, args.timeout, not args.insecure
        )

    async def browser_async(url: str) -> CheckResult:
        async with browser_sem:
            return await browser_check(context, url, args.timeout * 1000)

    async def check_one(url: str) -> CheckResult:
        if args.browser_only:
            if not use_browser:
                return CheckResult(url=url, category="request_error",
                                   error="browser tier unavailable", method="browser")
            return await browser_async(url)
        r = await http_async(url)
        if r.category == "alive" or not use_browser:
            return r
        # Tier 2: bot-blocked / errored on plain HTTP → retry with real browser.
        b = await browser_async(url)
        # Browser verdict is authoritative (matches how BaRA reaches links).
        return b

    summaries: list[dict] = []
    try:
        for path, urls in file_urls:
            total = len(urls)
            print(f"\n\U0001f4c2 {path}  ({total} URLs)")
            if total == 0:
                summaries.append({"path": path, "total": 0, "alive": 0, "tally": {}, "results": []})
                continue
            tasks = [asyncio.ensure_future(check_one(u)) for u in urls]
            results: list[CheckResult] = await asyncio.gather(*tasks)
            tally: dict[str, int] = {}
            for r in results:
                tally[r.category] = tally.get(r.category, 0) + 1
                if r.category != "alive":
                    st = f"{r.status:>3}" if r.status else "ERR"
                    print(f"  ✗ {st} [{r.method}] {_fmt_url(r.url)} {r.category} {r.error}")
                elif r.method == "browser":
                    print(f"  ✓ {r.status} [browser-recovered] {_fmt_url(r.url)}")
            alive = tally.get("alive", 0)
            pct = (alive / total * 100) if total else 0
            recovered = sum(1 for r in results if r.category == "alive" and r.method == "browser")
            print(f"  → {alive}/{total} alive ({pct:.0f}%)  details: {tally}"
                  + (f"  [+{recovered} browser-recovered]" if recovered else ""))
            summaries.append({
                "path": path, "total": total, "alive": alive,
                "tally": tally, "results": [asdict(r) for r in results],
            })
    finally:
        http_pool.shutdown(wait=False)
        if use_browser:
            try:
                await context.close()
                await browser.close()
            finally:
                await pw.stop()
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(
        description="Verify links_bfs.json URLs with HTTP + real-browser fallback (no bot-block underestimation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("paths", nargs="+", help="links_bfs.json file(s) and/or directory(ies)")
    p.add_argument("--http-concurrency", type=int, default=16, help="Parallel Tier-1 HTTP checks (default 16)")
    p.add_argument("--browser-concurrency", type=int, default=4, help="Parallel Tier-2 browser checks (default 4)")
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds (default 30)")
    p.add_argument("--output", default=None, help="Write full per-URL report JSON here")
    p.add_argument("--insecure", action="store_true", help="Skip TLS verification (both tiers)")
    p.add_argument("--no-browser-fallback", action="store_true", help="HTTP only; behaves like the old script")
    p.add_argument("--browser-only", action="store_true", help="Skip HTTP; check every URL with the browser")
    p.add_argument("--headful", action="store_true", help="Run Chromium with a visible window (defeats some headless detection)")
    args = p.parse_args()

    files = find_links_files(args.paths)
    if not files:
        print("❌ No links_bfs.json files found.", file=sys.stderr)
        sys.exit(1)

    mode = "browser-only" if args.browser_only else ("http-only" if args.no_browser_fallback else "http+browser-fallback")
    print(f"▶ Verifying {len(files)} links_bfs.json file(s)  [mode={mode}]")
    print(f"▶ http_concurrency={args.http_concurrency}, browser_concurrency={args.browser_concurrency}, "
          f"timeout={args.timeout}s, verify_tls={not args.insecure}")

    summaries = asyncio.run(verify_all_files(files, args))

    grand_total = sum(s["total"] for s in summaries)
    grand_alive = sum(s.get("alive", 0) for s in summaries)
    grand_tally: dict[str, int] = {}
    grand_recovered = 0
    for s in summaries:
        for k, v in s.get("tally", {}).items():
            grand_tally[k] = grand_tally.get(k, 0) + v
        grand_recovered += sum(1 for r in s.get("results", [])
                               if r.get("category") == "alive" and r.get("method") == "browser")

    pct = (grand_alive / grand_total * 100) if grand_total else 0
    print(f"\n{'=' * 60}")
    print(f"▶ Grand total: {grand_alive}/{grand_total} alive ({pct:.0f}%)  across {len(files)} file(s)")
    print(f"▶ Categories : {grand_tally}")
    if grand_recovered:
        print(f"▶ Browser-recovered (would be false-negatives on plain HTTP): {grand_recovered}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "files": summaries,
                "grand_total": grand_total,
                "grand_alive": grand_alive,
                "grand_tally": grand_tally,
                "browser_recovered": grand_recovered,
            }, f, ensure_ascii=False, indent=2)
        print(f"\U0001f4c4 Wrote: {args.output}")


if __name__ == "__main__":
    main()
