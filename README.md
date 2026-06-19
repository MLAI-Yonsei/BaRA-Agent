# BaRA: Budget-constrained and Reliable Web Data Collection Agent

BaRA is a deterministic Playwright-based crawler combined with an LLM-driven
content extraction and an LLM-free verification module that filters
hallucinated and unverifiable artifacts.

## Pipeline

```
  Step 1 (Playwright BFS, no LLM)
        |  links_bfs.json  (visited_order + by_depth)
        v
  Step 2 (LLM extraction with reflection + retry-merge)
        |  step2_results.jsonl
        v
  Verification (LLM-free, 5 gates for image/video, 2 gates for text)
        |  verification_records.jsonl
        v
  Final dataset (only artifacts whose final_decision == "include")
```

### Verification gates

Image / Video Gate (5 gates, all must pass):

| Gate | Check |
|------|-------|
| T1 | Artifact URL is present in the source page's DOM media index. |
| T2 | Artifact is downloadable (HTTP 2xx). |
| T3 | MIME type matches the declared modality (header -> libmagic -> ext). |
| T4 | Content hash is not a duplicate within the same site. |
| T5 | Hallucination flag (derived from T1: URL not in source DOM -> high). |

Text Gate (2 gates, both via `gate1_observed`):

| Gate | Check |
|------|-------|
| T1 | Normalized candidate text appears verbatim in the page DOM. |
| T2 | When T1 fails, token-set similarity against a sliding DOM window passes a fuzzy threshold. |

## Repository layout

```
.
|-- web_crawler/                    Main pipeline package (web_crawler.main)
|   |-- main.py                     CLI entry
|   |-- pipeline/
|   |   |-- step1.py / step2.py
|   |   |-- runtime.py
|   |   |-- bfs_rules.py
|   |   `-- verification/           5-Gate (image/video) + Text Gate (T1+T2) (text)
|   |-- _patch_browser_use_*.py     Local patches for browser-use 0.7.0
|   `-- verify_links_bfs_good.py    URL liveness check (HTTP + browser fallback)
|-- scripts/
|   |-- run_bara.py                 End-to-end BaRA orchestrator
|   `-- eval_unified.py             P/R/Acc evaluator (image/video/text)
|-- data/
|   `-- seed_urls.selected.jsonl    Real-world seed URL list (50 sites)
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

`requirements.txt` pins the verified versions of `browser-use==0.7.0`,
`playwright`, `yt-dlp`, `firecrawl-py`, and supporting libraries.

The two patch files under `web_crawler/_patch_browser_use_*.py` are loaded at
import time by `web_crawler.pipeline.runtime` and fix two issues in
`browser-use==0.7.0`:
* hardcoded per-event timeouts that fire before slow hosts become
  interactable (overridable via `BROWSER_USE_EVENT_TIMEOUT`);
* `top_p` / `seed` being forwarded into the OpenAI client constructor (which
  rejects them).

## Run modes

The pipeline has two stages. Step 2 (extraction) flows directly into
verification when `--enable_verification` is set, so verification is the
follow-up gating phase of Step 2 rather than a separate run mode.

| Stage | Flag(s) | What it does | Output |
|-------|---------|--------------|--------|
| **Step 1: link collection** | `--run_mode step1` | Playwright BFS (no LLM) | `links_bfs.json` |
| **Step 2: extraction -> verification** | `--run_mode step2 --start_url_path links_bfs.json --enable_verification` | LLM extraction per page, then LLM-free gate filtering (image/video 5-Gate, text Gate T1+T2) | `step2_results.jsonl`, `verification_records.jsonl`, `artifacts/` |
| **Full pipeline** | `--run_mode all --enable_verification` | Step 1 -> Step 2 (extraction -> verification) | all of the above |

### 1) Full pipeline

```bash
export OPENROUTER_API_KEY=""   # set your OpenRouter API key here
PYTHONPATH=. python -m web_crawler.main \
    --first_url https://example.com/some-site \
    --max_depth 3 --max_width 5 --max_pages 50 \
    --llm_provider openrouter \
    --model_name google/gemini-3-flash-preview \
    --enable_verification \
    --verification_output_dir ./verification_out \
    --api_keys "$OPENROUTER_API_KEY"
```

Step 2 runs single-threaded by default. Pass `--step2_concurrency N` to
launch N concurrent LLM extraction calls per topic.

Output (in the current working directory):
* `links_bfs.json` -- Step 1 BFS result
* `step2_results.jsonl` -- one line per visited page
* `verification_out/<host>/verification_records.jsonl` -- one line per artifact
* `verification_out/<host>/artifacts/` -- downloaded media and accepted text

### 2) Step 1 only -- collect links

```bash
PYTHONPATH=. python -m web_crawler.main \
    --first_url https://example.com/some-site \
    --max_depth 3 --max_width 5 --max_pages 50 \
    --run_mode step1
```

Writes `links_bfs.json` and exits.

### 3) Step 2 (extraction -> verification) on existing links

```bash
PYTHONPATH=. python -m web_crawler.main \
    --first_url https://example.com/some-site \
    --start_url_path links_bfs.json \
    --run_mode step2 \
    --llm_provider openrouter \
    --model_name google/gemini-3-flash-preview \
    --enable_verification \
    --verification_output_dir ./verification_out \
    --api_keys "$OPENROUTER_API_KEY"
```

Step 2 writes `step2_results.jsonl` (one line per visited page) and then
runs verification on every extracted artifact. Per-artifact gate records
land in `verification_out/<host>/verification_records.jsonl`; passing
media / text are stored under `verification_out/<host>/artifacts/`.

### 4) Bulk runner for many URLs

```bash
python scripts/run_bara.py \
    --urls-file ./urls.txt \
    --out ./runs/bara_demo \
    --model google/gemini-3-flash-preview \
    --api-key "$OPENROUTER_API_KEY"
```

By default each topic runs sequentially with Step 2 concurrency 1. Use
`--parallel N` to launch N topics in parallel and `--concurrency M` to
allow M concurrent LLM calls inside each topic. `--timeout-s` (default
1800) sets the per-topic timeout.

## Output layout

A full pipeline run on a single start URL produces the following tree in
the working directory (the `<host_slug>` folder is named after the start
URL with non-word characters replaced by `_`):

```
./
|-- links_bfs.json                          # Step 1: visited_order + by_depth
|-- step2_results.jsonl                     # Step 2: one JSON line per page
|-- annotations/                            # LLM-free per-page ground truth (default ON)
|   `-- page_<N>.json                       # image/video/link/text counts + lists + text_full
|-- run.log                                 # combined stdout/stderr
`-- verification_out/
    `-- <host_slug>/
        |-- verification_records.jsonl      # one JSON line per artifact (gates + decision)
        |-- verification_summary.json       # site-level totals (by_type counts)
        |-- dom_dump/                       # rendered HTML per page (used by T1 / text gates)
        |   `-- page_<N>.html
        |-- network_dump/                   # per-page HTTP metadata captured during extraction
        |   `-- page_<N>.json
        `-- artifacts/                      # ONLY artifacts that passed verification
            |-- images/<hash>.{jpg,png,webp,...}
            |-- videos/<hash>.{mp4,webm,...}
            `-- texts/
                `-- included_texts.jsonl    # full text bodies that passed T1/T2
```

`annotations/` is produced by an LLM-free Playwright pass that records what
the page actually contains (images, videos, links, full text). It is the
ground-truth side used by `scripts/eval_unified.py`. Disable it with
`--no_annotation` if you only want the predictive outputs.

The bulk runner (`scripts/run_bara.py`) creates one such tree per URL
under `--out`:

```
./runs/bara_demo/
|-- web_<slug_1>/
|   |-- links_bfs.json
|   |-- step2_results.jsonl
|   |-- run.log
|   `-- verification_out/<host_slug>/...
|-- web_<slug_2>/...
`-- ...
```

### File-by-file

| File | Shape |
|------|-------|
| `links_bfs.json` | `{start_url, max_depth, max_width, max_pages, visited_order: [url], by_depth: {0: [url], 1: [url], ...}, dead_links: [url]}` |
| `step2_results.jsonl` | One JSON object per line: `{sub_url, page_index, last_extracted_content: {content, ...}}` -- `content` carries `Text(s)`, `Image(s)`, `Video(s)` sections |
| `annotations/page_<N>.json` | `{image_count, images: [url], video_count, videos: [url], link_count, links: [{text, href}], text_length, text_full}` |
| `verification_records.jsonl` | One JSON object per artifact: `{artifact_url, artifact_type, source_page, page_index, gate1_observed, gate2_download_ok, gate3_mime_ok, gate4_not_duplicate, gate5_not_hallucinated, file_hash, mime_type, text_similarity (text only), text_content (text only), final_decision, exclusion_reasons, verified_at, duration_ms}` |
| `verification_summary.json` | `{total_candidates, by_type: {image: {included, excluded, ...}, video: {...}, text: {...}}}` |
| `artifacts/texts/included_texts.jsonl` | One JSON per accepted text: `{candidate_id, text, source_page, page_index, char_count, word_count, observation_channel, text_similarity, verified_at}` |

## Real-world seed list

`data/seed_urls.selected.jsonl` is the 50-site seed list used for the
real-world experiments (one JSON object per line). Each entry exposes the
URL plus its source provenance (`tranco`, `hn_algolia`, `wikipedia`,
`curlie`), language detection, body length, depth-2 media counts, and topic
cluster id.

Extract the URLs into a plain text file consumable by `scripts/run_bara.py`:

```bash
python -c "import json; \
[print(json.loads(l)['url']) for l in open('data/seed_urls.selected.jsonl')]" \
    > realworld_urls.txt

python scripts/run_bara.py \
    --urls-file realworld_urls.txt \
    --out ./runs/bara_realworld \
    --model google/gemini-3-flash-preview \
    --api-key "$OPENROUTER_API_KEY"
```

## Evaluation

`scripts/eval_unified.py` computes:

* Step 1 (sub-link discovery): MICRO and MACRO P/R/F1 over all topics.
* Step 2 (data collection): per-page Precision, Recall, and Accuracy for
  image, video, and text modalities, with policies:
    - URL normalization (scheme/host lowercase, fragments and queries stripped,
      `/index.html` and trailing `/` removed).
    - Text normalized to a word set (lowercase + punctuation/whitespace).
    - None=exclude: pages where both pred and GT are empty for a modality
      are skipped.
    - GT pages flagged as error/404 are excluded entirely.
    - **Pred-side dead-page text=0**: if the pred page's joined text triggers
      the error-page heuristic, that page's text P/R/Acc are forced to 0.
      Image/video are unaffected (404 pages typically have no media URLs).
    - Missing pred page (URL mismatch) -> all modalities P=R=Acc=0.

GT layout expected:

```
gt_annotation/<topic>/page_N.json    # one JSON per page with
                                     # image_count, images[], video_count,
                                     # videos[], link_count, links[],
                                     # text_length, text_full
gt_links/web_<topic>/links_bfs.json  # visited_order + by_depth
```

Example:

```bash
python scripts/eval_unified.py \
    --run ./runs/bara_demo \
    --type bara \
    --gt-annotation ./data/gt_annotation \
    --gt-links      ./data/gt_links \
    --label bara_demo \
    --out-json ./eval_bara_demo.json
```

Set `--type baseline` for a browser-use-style run (`json_page/page_N/data.json`
layout). Add `--no-pred-dead-text-zero` to disable the dead-page text=0
policy.

## URL liveness verification

`web_crawler.verify_links_bfs_good` checks URLs in `links_bfs.json` files with
a two-tier strategy: a cheap HTTP probe first, then a real headless Chromium
fallback for sites that WAF out a bare `requests` (genuine TLS fingerprint,
full headers, JS challenge resolution). The browser verdict wins, so
bot-protected sites that BaRA legitimately reached are not falsely marked
dead. The output file aggregates per-file `total` / `alive` counts plus a
`grand_total` / `grand_alive`, which is what the Step 1 micro/macro link
metrics consume.

```bash
# single file
python -m web_crawler.verify_links_bfs_good path/to/links_bfs.json --output report.json

# folder (recursively finds every links_bfs.json under it)
python -m web_crawler.verify_links_bfs_good ./runs/bara_demo --output report.json

# HTTP-only (skip the browser tier)
python -m web_crawler.verify_links_bfs_good ./runs/bara_demo --no-browser-fallback
```
