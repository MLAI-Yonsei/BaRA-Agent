"""End-to-end BaRA runner (Step 1 Playwright BFS -> Step 2 LLM -> verification).

Iterates over a set of start URLs and launches `web_crawler.main` for each.
Idempotent: skips topics whose step2_results.jsonl already exists.

Two input modes for the URL list:
  1) --topics-from-dir <root>
        Each subfolder of <root> named 'web_<topic>' becomes one run.
        The start URL is built as <base-url>/web_<topic>.
  2) --urls-file <file>
        One start URL per line.

Output layout: <out>/<run_name>/{links_bfs.json, step2_results.jsonl,
                                  verification_out/, run.log}
where <run_name> is 'web_<topic>' for topics-from-dir and a safe slug for
--urls-file mode.

Required:
  --api-key      LLM provider API key (e.g. OpenRouter key).

Example:
  python scripts/run_bara.py \\
      --topics-from-dir ./data/synthetic_GT_link \\
      --base-url https://example.com/web-ai \\
      --out ./runs/bara_synthetic \\
      --model google/gemini-3-flash-preview \\
      --api-key $OPENROUTER_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path


def slugify(url: str) -> str:
    s = re.sub(r"^https?://", "", url)
    s = re.sub(r"[^\w.-]+", "_", s).strip("_")
    return s[:200]


async def run_one(*, start_url: str, run_name: str, out_root: Path,
                  repo_root: Path, python_bin: str,
                  api_key: str, model: str, llm_provider: str,
                  max_depth: int, max_width: int, max_pages: int,
                  concurrency: int, timeout_s: int,
                  sema: asyncio.Semaphore):
    async with sema:
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "step2_results.jsonl").is_file():
            print(f"[skip  ] {run_name} - step2_results.jsonl exists", flush=True)
            return run_name, 0, 0

        log_path = out_dir / "run.log"
        cmd = [
            python_bin, "-m", "web_crawler.main",
            "--first_url", start_url,
            "--max_depth", str(max_depth),
            "--max_width", str(max_width),
            "--max_pages", str(max_pages),
            "--llm_provider", llm_provider,
            "--model_name", model,
            "--step2_concurrency", str(concurrency),
            "--enable_verification",
            "--verification_output_dir", "./verification_out",
            "--api_keys", api_key,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        print(f"[start ] {run_name}", flush=True)
        t0 = time.monotonic()
        flog = open(log_path, "w", encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(out_dir), env=env,
            stdout=flog, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            outcome = f"rc={rc}"
        except asyncio.TimeoutError:
            proc.kill()
            try: await proc.wait()
            except Exception: pass
            rc = -1
            outcome = f"TIMEOUT after {timeout_s}s"
        finally:
            try: flog.close()
            except Exception: pass
        elapsed = int(time.monotonic() - t0)
        print(f"[done  ] {run_name:<24}  {outcome:<28}  {elapsed:>4}s",
              flush=True)
        return run_name, rc, elapsed


def collect_targets(args) -> list[tuple[str, str]]:
    """Return list of (run_name, start_url) tuples."""
    out: list[tuple[str, str]] = []
    if args.topics_from_dir:
        base = args.base_url.rstrip("/") if args.base_url else None
        if base is None:
            print("ERROR: --topics-from-dir requires --base-url", file=sys.stderr)
            sys.exit(2)
        for p in sorted(Path(args.topics_from_dir).iterdir()):
            if not p.is_dir() or not p.name.startswith("web_"): continue
            topic = p.name.replace("web_", "", 1)
            out.append((f"web_{topic}", f"{base}/web_{topic}"))
    if args.urls_file:
        for raw in open(args.urls_file):
            url = raw.strip()
            if not url or url.startswith("#"): continue
            out.append((slugify(url), url))
    return out


async def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--topics-from-dir", type=Path,
                     help="Folder with web_<topic>/ subdirs.")
    src.add_argument("--urls-file", type=Path,
                     help="Plain-text file, one start URL per line.")
    ap.add_argument("--base-url",
                    help="Base URL for --topics-from-dir mode "
                         "(e.g. https://example.com/web-ai)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root directory.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="Repo root (for PYTHONPATH).")
    ap.add_argument("--python-bin", default=sys.executable,
                    help="Python interpreter (default: the current one).")
    ap.add_argument("--llm-provider", default="openrouter",
                    choices=["google", "ollama", "openrouter"])
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--api-key", required=True,
                    help="LLM provider API key.")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-width", type=int, default=5)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of topics to run in parallel (default: 1).")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Step 2 LLM concurrency within each topic (default: 1).")
    ap.add_argument("--timeout-s", type=int, default=1800,
                    help="Per-topic timeout in seconds (default: 1800).")
    args = ap.parse_args()

    targets = collect_targets(args)
    if not targets:
        print("ERROR: no targets discovered.", file=sys.stderr)
        sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"=== {len(targets)} targets, parallel={args.parallel} "
          f"concurrency={args.concurrency} model={args.model} ===",
          flush=True)

    sema = asyncio.Semaphore(args.parallel)
    t0 = time.monotonic()
    tasks = [asyncio.create_task(run_one(
        start_url=url, run_name=name, out_root=args.out,
        repo_root=args.repo_root, python_bin=args.python_bin,
        api_key=args.api_key, model=args.model,
        llm_provider=args.llm_provider,
        max_depth=args.max_depth, max_width=args.max_width,
        max_pages=args.max_pages,
        concurrency=args.concurrency, timeout_s=args.timeout_s,
        sema=sema,
    )) for (name, url) in targets]
    results = await asyncio.gather(*tasks)
    elapsed = int(time.monotonic() - t0)

    print()
    print("=" * 70)
    print(f"ALL DONE in {elapsed}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    fail = len(results) - ok
    print(f"success: {ok}   fail: {fail}")
    for name, rc, t in sorted(results):
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  {name:<24}  {mark}  {t}s")


if __name__ == "__main__":
    asyncio.run(main())
