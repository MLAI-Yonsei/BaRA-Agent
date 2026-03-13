import asyncio
import json
import os
import sys

from web_crawler.pipeline import Step1Manager, Step2Manager, Step3Manager
from web_crawler.pipeline.runtime import (
    StepByStepPipelineRunner,
    build_config_from_args,
    build_parser,
)


async def main():
    parser = build_parser()
    parser.add_argument(
        "--run_mode",
        type=str,
        default="all",
        choices=["all", "step1", "step2", "step3"],
        help="Execution mode: all pipeline or single step mode.",
    )
    parser.add_argument(
        "--start_url_path",
        type=str,
        default=None,
        help="Standalone Step2 input path (links_bfs.json).",
    )
    parser.add_argument(
        "--step2_results_file",
        type=str,
        default="step2_results.jsonl",
        help="Step2 output file for standalone Step3 mode.",
    )
    args = parser.parse_args()
    config = build_config_from_args(args)

    validator = StepByStepPipelineRunner(config)
    validator._validate_provider(parser)
    validator._check_ollama()

    step1_manager = Step1Manager(config)
    step2_manager = Step2Manager(config)
    step3_manager = Step3Manager(config)

    current_api_key_index = 0
    api_keys = config.api_keys if config.api_keys else [None]

    async def run_step1_mode(api_key):
        step1_success, attachment_path = await step1_manager.execute(api_key)
        if not step1_success or not attachment_path:
            return False
        print(f"✅ Step 1 completed: {attachment_path}")
        return True

    async def run_step2_mode(api_key):
        start_url_path = args.start_url_path or config.step1_links_path
        if not os.path.exists(start_url_path):
            raise FileNotFoundError(f"Step2 input file not found: {start_url_path}")

        written = 0
        with open(args.step2_results_file, "w", encoding="utf-8") as f:
            async for sub_url, last_extracted_content, page_index in step2_manager.iterate(api_key, start_url_path):
                row = {
                    "sub_url": sub_url,
                    "last_extracted_content": last_extracted_content,
                    "page_index": page_index,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        print(f"✅ Step 2 completed: {args.step2_results_file} ({written} items)")
        return True

    async def run_step3_mode(api_key):
        if not os.path.exists(args.step2_results_file):
            raise FileNotFoundError(f"Step3 input file not found: {args.step2_results_file}")

        processed = 0
        with open(args.step2_results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                await step3_manager.process(
                    row["sub_url"],
                    row["last_extracted_content"],
                    row["page_index"],
                    api_key,
                    current_api_key_index,
                    api_keys,
                )
                processed += 1
        print(f"✅ Step 3 completed: {processed} items processed")
        return True

    while current_api_key_index < len(api_keys):
        api_key = api_keys[current_api_key_index]
        if config.llm_provider == "google":
            print(f"🔑 Using API key {current_api_key_index + 1}/{len(api_keys)}...")
        else:
            print("🔍 Using Ollama model...")

        try:
            if args.run_mode == "step1":
                await run_step1_mode(api_key)
                return

            if args.run_mode == "step2":
                await run_step2_mode(api_key)
                return

            if args.run_mode == "step3":
                await run_step3_mode(api_key)
                return

            step1_success, attachment_path = await step1_manager.execute(api_key)
            if not step1_success or not attachment_path:
                break

            if config.step1_only:
                print("✅ Step 1 completed (debug mode)")
                return

            async for sub_url, last_extracted_content, page_index in step2_manager.iterate(api_key, attachment_path):
                print(f"\n{'='*80}")
                print(f"🔗 Processing URL: {sub_url}")
                print(f"📄 Page index: {page_index}")
                print(f"{'='*80}\n")
                await step3_manager.process(
                    sub_url,
                    last_extracted_content,
                    page_index,
                    api_key,
                    current_api_key_index,
                    api_keys,
                )
            return
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'quota' in error_msg.lower() or 'resource_exhausted' in error_msg.lower():
                print(f"⚠️ API key {current_api_key_index + 1} quota exceeded: {error_msg}")
                current_api_key_index += 1
                if current_api_key_index >= len(api_keys):
                    print("\n❌ All API keys have exceeded their quota.")
                    sys.exit(1)
                continue
            raise


if __name__ == "__main__":
    asyncio.run(main())
