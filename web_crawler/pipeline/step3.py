import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime

from web_crawler.pipeline.runtime import (
    _content_without_attachments,
    build_json_page_dir,
    build_multimodal_classifier_prompt,
    extract_folder_name,
    process_txt_file,
    run_step3,
    split_content_by_lines,
)


class Step3Manager:
    def __init__(self, config):
        self.config = config

    def _to_items_list(self, last_extracted_content):
        if isinstance(last_extracted_content, dict):
            content_only = _content_without_attachments(last_extracted_content.get("content", ""))
            separated_entries = split_content_by_lines([{**last_extracted_content, "content": content_only}])
        elif isinstance(last_extracted_content, str):
            content_only = _content_without_attachments(last_extracted_content)
            separated_entries = []
            lines = [line.strip() for line in content_only.split("\n") if line.strip()]
            for line in lines:
                if line.startswith("Text(s):") or line.startswith("Image(s):") or line.startswith("Video(s):"):
                    continue
                if line.startswith("- "):
                    separated_entries.append({"line_content": line[2:]})
                elif line and not line.startswith("```"):
                    separated_entries.append({"line_content": line})
        else:
            content_only = _content_without_attachments(str(last_extracted_content))
            separated_entries = split_content_by_lines([{"content": content_only}])

        items_list = []
        for entry in separated_entries:
            content = entry.get("line_content", str(entry)) if isinstance(entry, dict) else str(entry)
            if content:
                items_list.append(content)
        return items_list

    async def process(self, sub_url, last_extracted_content, page_index, api_key, current_api_key_index, api_keys):
        items_list = self._to_items_list(last_extracted_content)
        items_text = "\n".join([f"- {item}" for item in items_list[:200]])
        if len(items_list) > 200:
            items_text += f"\n... (showing only 200 of {len(items_list)} items)"

        print(f"🔍 Debug: extracted item count = {len(items_list)}")
        print(f"🔍 Debug: items_text length = {len(items_text)}")

        BATCH_SIZE = self.config.step3_batch_size
        all_illegal_items = []
        all_legal_items = []
        root_folder = extract_folder_name(self.config.first_url)
        page_output_dir = build_json_page_dir(root_folder, page_index)
        os.makedirs(page_output_dir, exist_ok=True)
        output_file = "verified_content_present.txt"
        verified_content = None

        if len(items_list) > BATCH_SIZE:
            print(f"📦 Batch processing: splitting {len(items_list)} items into batches of {BATCH_SIZE}.")
            num_batches = (len(items_list) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(items_list))
                batch_items = items_list[start_idx:end_idx]
                batch_text = "\n".join([f"- {item}" for item in batch_items])
                verifier_prompt = build_multimodal_classifier_prompt(batch_text)

                max_retries = self.config.step3_batch_retries
                batch_success = False
                for retry_attempt in range(max_retries):
                    temp_batch_output_dir = None
                    try:
                        temp_batch_output_dir = tempfile.mkdtemp()
                        batch_verified_content, _ = await run_step3(
                            verifier_prompt,
                            api_key,
                            self.config.model_name,
                            self.config.llm_provider,
                            self.config.ollama_host,
                            self.config.ollama_api_key,
                            output_dir=temp_batch_output_dir,
                        )
                        if batch_verified_content and len(batch_verified_content.strip()) > 0:
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', batch_verified_content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                            else:
                                json_match = re.search(r'\{.*"illegal".*"legal".*\}', batch_verified_content, re.DOTALL)
                                json_str = json_match.group(0) if json_match else None
                            if json_str:
                                batch_result = json.loads(json_str)
                                all_illegal_items.extend(batch_result.get("illegal", []))
                                all_legal_items.extend(batch_result.get("legal", []))
                                with open(os.path.join(page_output_dir, f"legal_content_batch_{batch_idx}.json"), "w", encoding="utf-8") as f:
                                    json.dump(batch_result.get("legal", []), f, ensure_ascii=False, indent=2)
                                with open(os.path.join(page_output_dir, f"illegal_content_batch_{batch_idx}.json"), "w", encoding="utf-8") as f:
                                    json.dump(batch_result.get("illegal", []), f, ensure_ascii=False, indent=2)
                                batch_success = True
                                break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Batch {batch_idx + 1}/{num_batches} JSON parse failed: {e}")
                    except Exception as e:
                        print(f"❌ Error while processing batch {batch_idx + 1}/{num_batches}: {e}")
                    finally:
                        if temp_batch_output_dir:
                            shutil.rmtree(temp_batch_output_dir, ignore_errors=True)
                    if retry_attempt < max_retries - 1:
                        await asyncio.sleep(2)

                if not batch_success:
                    print(f"❌ Batch {batch_idx + 1}/{num_batches} exceeded the maximum retry count. Skipping.")

            final_result = {"illegal": all_illegal_items, "legal": all_legal_items}
            with open(os.path.join(page_output_dir, "legal_content.json"), "w", encoding="utf-8") as f:
                json.dump(final_result.get("legal", []), f, ensure_ascii=False, indent=2)
            with open(os.path.join(page_output_dir, "illegal_content.json"), "w", encoding="utf-8") as f:
                json.dump(final_result.get("illegal", []), f, ensure_ascii=False, indent=2)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Verification timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                f.write("🚫 Illegal Content:\n")
                f.write("-" * 60 + "\n")
                illegal_items = final_result.get("illegal", [])
                if illegal_items:
                    for i, item in enumerate(illegal_items, 1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("(none)\n")
                f.write("\n" + "=" * 60 + "\n\n")
                f.write("✅ Legal Content:\n")
                f.write("-" * 60 + "\n")
                legal_items = final_result.get("legal", [])
                if legal_items:
                    for i, item in enumerate(legal_items, 1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("(none)\n")
                f.write("\n" + "=" * 60 + "\n")

            verified_content = json.dumps(final_result, ensure_ascii=False, indent=2)
        else:
            verifier_prompt = build_multimodal_classifier_prompt(items_text)
            try:
                verified_content, output_file = await run_step3(
                    verifier_prompt,
                    api_key,
                    self.config.model_name,
                    self.config.llm_provider,
                    self.config.ollama_host,
                    self.config.ollama_api_key,
                    output_dir=page_output_dir,
                )
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower() or 'resource_exhausted' in error_msg.lower():
                    print(f"⚠️ API key {current_api_key_index + 1} quota exceeded: {error_msg}")
                    current_api_key_index += 1
                    if current_api_key_index >= len(api_keys):
                        print("\n❌ All API keys have exceeded their quota.")
                        sys.exit(1)
                else:
                    raise

        if verified_content or output_file:
            process_txt_file(output_file, self.config.first_url, sub_url=sub_url, page_index=page_index)
        else:
            print(f"❌ Step 3 failed for {sub_url}")
