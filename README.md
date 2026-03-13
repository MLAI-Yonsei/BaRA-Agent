# Web Crawler

This repository provides:

- a browser-based crawling pipeline
- step-by-step execution through a single entry point
- an evaluation script for text, image, and video outputs

The code is organized so that you can:

- run the full pipeline at once
- run Step 1 only
- run Step 2 only
- run Step 3 only
- evaluate generated outputs separately

## Repository Structure

```text
.
├── requirements.txt
├── README.md
└── web_crawler/
    ├── __init__.py
    ├── main.py
    ├── eval.py
    └── pipeline/
        ├── __init__.py
        ├── runtime.py
        ├── step1.py
        ├── step2.py
        └── step3.py
```

## What Each File Does

- `web_crawler/main.py`
  - Main entry point for the crawler pipeline.
  - Supports full execution and step-by-step execution with `--run_mode`.
- `web_crawler/pipeline/step1.py`
  - Link collection.
- `web_crawler/pipeline/step2.py`
  - Page-level content extraction.
- `web_crawler/pipeline/step3.py`
  - Content classification and output generation.
- `web_crawler/eval.py`
  - Evaluation for text, image, and video results.

## Requirements

- Python 3.11 or newer
- A clean virtual environment
- Chromium installed through Playwright
- Valid API credentials when using external services

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install the Playwright browser:

```bash
playwright install chromium
```

## Main Entry Point

Show all available arguments:

```bash
python -m web_crawler.main --help
```

The pipeline is executed through `web_crawler.main`.

Use `--run_mode` to choose how to run it:

- `all`: run Step 1 -> Step 2 -> Step 3
- `step1`: run only Step 1
- `step2`: run only Step 2
- `step3`: run only Step 3

## Run the Full Pipeline

Use this when you want to start from the first URL and execute everything in order.

```bash
python -m web_crawler.main \
  --run_mode all \
  --llm_provider google \
  --api_keys YOUR_API_KEY \
  --first_url https://example.com
```

Typical flow:

1. Step 1 collects links and creates `links_bfs.json`
2. Step 2 reads collected links and extracts page content
3. Step 3 classifies extracted content and writes results

## Run Each Step Independently

### Step 1 Only

Use this when you only want link collection.

```bash
python -m web_crawler.main \
  --run_mode step1 \
  --llm_provider google \
  --api_keys YOUR_API_KEY \
  --first_url https://example.com \
  --max_depth 1 \
  --max_width 2 \
  --max_pages 5 \
  --max_attempts 2
```

Expected output:

- `links_bfs.json`

### Step 2 Only

Use this when you already have a `links_bfs.json` file from Step 1.

```bash
python -m web_crawler.main \
  --run_mode step2 \
  --llm_provider google \
  --api_keys YOUR_API_KEY \
  --start_url_path links_bfs.json \
  --step2_results_file step2_results.jsonl \
  --max_attempts 2 \
  --step2_union_retry_attempts 2
```

Expected input:

- `links_bfs.json`

Expected output:

- `step2_results.jsonl`

### Step 3 Only

Use this when you already have Step 2 output.

```bash
python -m web_crawler.main \
  --run_mode step3 \
  --llm_provider google \
  --api_keys YOUR_API_KEY \
  --step2_results_file step2_results.jsonl \
  --step3_batch_size 50 \
  --step3_batch_retries 2
```

Expected input:

- `step2_results.jsonl`

Expected output:

- `json_page/page_*/legal_content.json`
- `json_page/page_*/illegal_content.json`
- classified text, image, and video outputs under the generated root directory

## Common Arguments

### Provider and model

- `--llm_provider {google,ollama}`
- `--api_keys`
- `--model_name`
- `--ollama_host`
- `--ollama_api_key`

### Target and filtering

- `--first_url`
- `--wanted`
- `--wanted_file`

### Step 1 controls

- `--max_depth`
- `--max_width`
- `--max_pages`
- `--max_attempts`
- `--step1_only`
- `--skip_step1_prefix`
- `--step1_links_path`

### Step 2 controls

- `--start_url_path`
- `--step2_results_file`
- `--step2_union_retry_attempts`

### Step 3 controls

- `--step3_batch_size`
- `--step3_batch_retries`

## Evaluation

Show all available arguments:

```bash
python -m web_crawler.eval --help
```

The evaluation script expects a data root with this structure:

```text
<data_root>/
├── annotations/
├── json_page/
├── legal/
└── illegal/
```

Run evaluation:

```bash
python -m web_crawler.eval --data_root /path/to/data_root
```

Run evaluation with detailed per-page output:

```bash
python -m web_crawler.eval \
  --data_root /path/to/data_root \
  --show_details
```

Run evaluation with line-based text comparison:

```bash
python -m web_crawler.eval \
  --data_root /path/to/data_root \
  --text_metric_unit line
```

## Notes

- The crawler uses [browser-use](https://github.com/browser-use/browser-use).
- Browser-based execution requires `playwright install chromium`.
- External credentials should be passed through arguments or environment variables.
- Step 2 and Step 3 can be executed independently only when their required input files already exist.
