import asyncio
import json
import os
import re
import sys

from web_crawler.pipeline import Step1Manager, Step2Manager, Step3Manager
from web_crawler.pipeline.runtime import (
    StepByStepPipelineRunner,
    audit_website,
    build_config_from_args,
    build_parser,
    extract_folder_name,
)
from web_crawler.pipeline.verification import (
    ArtifactCandidate,
)
from web_crawler.pipeline.verification.runner import (
    VerificationConfig,
    verify_page,
)
from web_crawler.pipeline.verification.store import (
    SeenHashes,
    write_summary,
)


# ---------------------------------------------------------------------------
# Helpers: extract image/video URLs from Step 2's textual content
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")
_IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|gif|bmp|webp|svg)(?:[?#]|$)", re.IGNORECASE)
_VID_EXT_RE = re.compile(r"\.(mp4|mov|avi|mkv|webm|flv|m4v|m3u8|ts)(?:[?#]|$)", re.IGNORECASE)


def _section_body(content: str, section: str) -> str:
    """Return text between `- Section(s):` and the next sibling section."""
    head_re = re.compile(rf"(?:^|\n)(?:- )?{section}\(s\):", re.IGNORECASE)
    m = head_re.search(content)
    if not m:
        return ""
    start = m.end()
    next_re = re.compile(r"(?:^|\n)(?:- )?(Text|Image|Video)\(s\):", re.IGNORECASE)
    n = next_re.search(content, pos=start)
    return content[start: n.start()] if n else content[start:]


def _extract_urls(body: str, ext_re: re.Pattern) -> list[str]:
    seen: dict[str, None] = {}
    for m in _URL_RE.finditer(body or ""):
        url = m.group(0).rstrip(".,;:)")
        if ext_re.search(url):
            seen.setdefault(url, None)
    # Some pages emit URLs that lack a typical extension (CDN-served);
    # we still allow them in the modality body if they appear there.
    if not seen:
        for m in _URL_RE.finditer(body or ""):
            url = m.group(0).rstrip(".,;:)")
            seen.setdefault(url, None)
    return list(seen.keys())


def _extract_text_lines(body: str) -> list[str]:
    """Return each non-empty text line from the `- Text(s):` section.

    Items are typically prefixed with `- ` bullets but we accept bare lines too.
    """
    if not body:
        return []
    out = []
    seen = set()
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip leading bullet markers
        if line.startswith("- "):
            line = line[2:].strip()
        # reject section header echoes & explicit "none" markers
        if re.match(r"^(text|image|video)\(s\)\s*:$", line, re.IGNORECASE):
            continue
        if re.match(r"^(none|no .*|none found)$", line, re.IGNORECASE):
            continue
        if len(line) < 2:
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _candidates_from_extracted(last_extracted_content, sub_url: str, page_index: int) -> list[ArtifactCandidate]:
    """Pull image + video URLs and text lines out of Step 2 output and wrap as ArtifactCandidates.

    Accepts the same shape Step 2 yields: a dict with a `content` field, a
    string, or anything stringifiable.
    """
    if isinstance(last_extracted_content, dict):
        content = last_extracted_content.get("content", "")
    elif isinstance(last_extracted_content, str):
        content = last_extracted_content
    else:
        content = str(last_extracted_content or "")

    image_urls = _extract_urls(_section_body(content, "Image"), _IMG_EXT_RE)
    video_urls = _extract_urls(_section_body(content, "Video"), _VID_EXT_RE)
    text_lines = _extract_text_lines(_section_body(content, "Text"))

    out: list[ArtifactCandidate] = []
    for u in image_urls:
        out.append(ArtifactCandidate(url=u, type="image",
                                     source_page=sub_url, page_index=page_index))
    for u in video_urls:
        out.append(ArtifactCandidate(url=u, type="video",
                                     source_page=sub_url, page_index=page_index))
    for i, t in enumerate(text_lines):
        import hashlib as _h
        h8 = _h.sha1(t.encode("utf-8")).hexdigest()[:8]
        ident = f"text:{page_index}:{i}:{h8}"
        out.append(ArtifactCandidate(url=ident, type="text",
                                     source_page=sub_url, page_index=page_index,
                                     text_content=t))
    return out


async def _save_annotation(sub_url: str, page_index: int, base_dir: str = "."):
    """Run audit_website on `sub_url` and dump the result to
    `<base_dir>/annotations/page_<page_index>.json`.

    This is the ground-truth side-channel that used to live inside
    `process_txt_file()` (called only from Step 3).  Lifted out so it works
    independently of the legal/illegal classification step.

    LLM-free — uses Playwright to extract <img>/<video>/<a>/innerText from the
    rendered DOM.
    """
    try:
        ann_dir = os.path.join(base_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        path = os.path.join(ann_dir, f"page_{page_index}.json")
        data = await audit_website(sub_url)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"📋 Saved annotation: {path}")
    except Exception as e:
        print(f"⚠️ Annotation creation failed ({sub_url}): {e}")


def _verification_config_from(config) -> VerificationConfig:
    site_root = extract_folder_name(config.first_url)
    return VerificationConfig(
        mode=config.verification_mode,
        output_dir=os.path.join(config.verification_output_dir, site_root)
                   if not config.verification_output_dir.endswith(site_root)
                   else config.verification_output_dir,
        keep_downloaded_files=config.verification_keep_files,
        download_timeout_image_s=config.verification_download_timeout_image_s,
        download_timeout_video_s=config.verification_download_timeout_video_s,
        download_max_bytes_image=config.verification_max_bytes_image,
        download_max_bytes_video=config.verification_max_bytes_video,
        capture_timeout_ms=config.verification_capture_timeout_ms,
    )


def _filter_content_to_included(last_extracted_content, included_urls: set[str], included_texts: set[str] | None = None):
    """Rewrite Step 2's content so only included image/video/text items survive.

    `included_urls`  — set of image/video URLs that passed verification
    `included_texts` — set of text strings that passed verification (None = keep all text)
    """
    if isinstance(last_extracted_content, dict):
        content = last_extracted_content.get("content", "")
    elif isinstance(last_extracted_content, str):
        content = last_extracted_content
    else:
        content = str(last_extracted_content or "")
    if not content:
        return last_extracted_content

    def _kept_urls(section: str) -> list[str]:
        body = _section_body(content, section)
        out = []
        seen = set()
        for m in _URL_RE.finditer(body):
            url = m.group(0).rstrip(".,;:)")
            if url in included_urls and url not in seen:
                seen.add(url)
                out.append(url)
        return out

    text_body = _section_body(content, "Text")
    if included_texts is None:
        kept_texts = _extract_text_lines(text_body)
    else:
        kept_texts = [t for t in _extract_text_lines(text_body) if t in included_texts]

    new_parts = ["- Text(s):"]
    new_parts.extend([f"  - {t}" for t in kept_texts] or ["  - (none)"])

    imgs = _kept_urls("Image")
    new_parts.append("- Image(s):")
    new_parts.extend([f"  - {u}" for u in imgs] or ["  - (none)"])
    vids = _kept_urls("Video")
    new_parts.append("- Video(s):")
    new_parts.extend([f"  - {u}" for u in vids] or ["  - (none)"])
    new_content = "\n".join(new_parts)

    if isinstance(last_extracted_content, dict):
        out = dict(last_extracted_content)
        out["content"] = new_content
        return out
    return new_content


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
    parser.add_argument(
        "--step2_results_file_secondary",
        type=str,
        default=None,
        help="When --ablation_retry_merge_mode=last_and_best, write the 'best' selections "
             "to this jsonl while --step2_results_file gets the 'last' selections.",
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

        combined_mode = (
            getattr(config, "ablation_retry_merge_mode", "union") == "last_and_best"
        )
        secondary_path = args.step2_results_file_secondary

        if combined_mode and not secondary_path:
            raise ValueError(
                "--ablation_retry_merge_mode=last_and_best requires "
                "--step2_results_file_secondary to be set."
            )

        # ---- verification setup (when --enable_verification) ----
        ver_seen_hashes = SeenHashes() if config.enable_verification else None
        ver_config = _verification_config_from(config) if config.enable_verification else None
        ver_all_records: list = []  # collected for site-level summary
        if config.enable_verification:
            print(f"🛡️  Verification enabled (mode={config.verification_mode}, "
                  f"output={ver_config.output_dir})")

        written = 0
        f_primary = open(args.step2_results_file, "w", encoding="utf-8")
        f_secondary = open(secondary_path, "w", encoding="utf-8") if combined_mode else None
        try:
            async for sub_url, last_extracted_content, page_index in step2_manager.iterate(api_key, start_url_path):
                # ---- verification: per-page 5-gate ----
                if config.enable_verification:
                    # combined_mode returns a {last,best} dict — verify the 'last' selection
                    if combined_mode and isinstance(last_extracted_content, dict) and last_extracted_content.get("_combined_marker"):
                        verify_target = last_extracted_content["last"]
                    else:
                        verify_target = last_extracted_content

                    candidates = _candidates_from_extracted(verify_target, sub_url, page_index)
                    print(f"\n🛡️  verify page_{page_index} ({sub_url})  candidates={len(candidates)}")
                    _, recs = await verify_page(
                        sub_url, page_index, candidates,
                        seen_hashes=ver_seen_hashes,
                        config=ver_config,
                        site_root=".",   # ver_config.output_dir is already site-scoped
                    )
                    ver_all_records.extend(recs)
                    included_urls = {
                        r.artifact_url for r in recs
                        if r.final_decision == "include" and r.artifact_type in ("image", "video")
                    }
                    # for text records, the candidate URL was a synthetic identifier;
                    # the actual content lives in cand.text_content (recorded as record.text_content[:200])
                    included_texts = {
                        c.text_content for c, r in zip(candidates, recs)
                        if c.type == "text" and r.final_decision == "include" and c.text_content
                    }
                    if combined_mode and isinstance(last_extracted_content, dict) and last_extracted_content.get("_combined_marker"):
                        last_extracted_content = {
                            **last_extracted_content,
                            "last": _filter_content_to_included(last_extracted_content["last"], included_urls, included_texts),
                            "best": _filter_content_to_included(last_extracted_content["best"], included_urls, included_texts),
                        }
                    else:
                        last_extracted_content = _filter_content_to_included(last_extracted_content, included_urls, included_texts)

                if config.enable_annotation:
                    await _save_annotation(sub_url, page_index)

                if combined_mode and isinstance(last_extracted_content, dict) and last_extracted_content.get("_combined_marker"):
                    last_row = {
                        "sub_url": sub_url,
                        "last_extracted_content": last_extracted_content["last"],
                        "page_index": page_index,
                    }
                    best_row = {
                        "sub_url": sub_url,
                        "last_extracted_content": last_extracted_content["best"],
                        "page_index": page_index,
                    }
                    f_primary.write(json.dumps(last_row, ensure_ascii=False) + "\n")
                    f_secondary.write(json.dumps(best_row, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    row = {
                        "sub_url": sub_url,
                        "last_extracted_content": last_extracted_content,
                        "page_index": page_index,
                    }
                    f_primary.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if combined_mode and f_secondary is not None:
                        # No retry-merge happened (image+video weren't both missing); replicate to secondary.
                        f_secondary.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
        finally:
            f_primary.close()
            if f_secondary is not None:
                f_secondary.close()

        if combined_mode:
            print(f"✅ Step 2 completed (last_and_best): "
                  f"{args.step2_results_file} + {secondary_path} ({written} items each)")
        else:
            print(f"✅ Step 2 completed: {args.step2_results_file} ({written} items)")

        # ---- verification: site-level summary ----
        if config.enable_verification and ver_all_records:
            summary_path = os.path.join(ver_config.output_dir, "verification_summary.json")
            summary = write_summary(ver_all_records, summary_path)
            print(f"\n🛡️  Verification summary → {summary_path}")
            for t, b in summary.get("by_type", {}).items():
                if b.get("total", 0) == 0:
                    continue
                print(f"   {t:6s}: included {b['final_included']}/{b['total']} "
                      f"(g1={b['gate1_observed_rate']:.2f} g2={b['gate2_download_rate']:.2f} "
                      f"g3={b['gate3_mime_rate']:.2f} g4={b['gate4_dedup_rate']:.2f} "
                      f"g5={b['gate5_not_halluc_rate']:.2f})")
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
        if config.llm_provider in ("google", "openrouter"):
            print(f"🔑 Using API key {current_api_key_index + 1}/{len(api_keys)} ({config.llm_provider})...")
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

            # ---- verification setup for end-to-end mode ----
            ver_seen_hashes_e2e = SeenHashes() if config.enable_verification else None
            ver_config_e2e = _verification_config_from(config) if config.enable_verification else None
            ver_all_records_e2e: list = []
            if config.enable_verification:
                print(f"🛡️  Verification enabled (mode={config.verification_mode}, "
                      f"output={ver_config_e2e.output_dir})")

            # The pipeline now stops after verification — legal/illegal classification
            # (Step 3) is no longer part of the default flow.  Step 2 results are
            # persisted to step2_results.jsonl so they can be re-used (e.g. running
            # `--run_mode step3` later, or running the eval notebooks against them).
            step2_results_path = args.step2_results_file
            step2_written = 0
            f_step2 = open(step2_results_path, "w", encoding="utf-8")
            try:
                async for sub_url, last_extracted_content, page_index in step2_manager.iterate(api_key, attachment_path):
                    print(f"\n{'='*80}")
                    print(f"🔗 Step 2 yield: {sub_url}")
                    print(f"📄 Page index: {page_index}")
                    print(f"{'='*80}\n")

                    if config.enable_verification:
                        candidates = _candidates_from_extracted(last_extracted_content, sub_url, page_index)
                        print(f"🛡️  verify page_{page_index}  candidates={len(candidates)}")
                        _, recs = await verify_page(
                            sub_url, page_index, candidates,
                            seen_hashes=ver_seen_hashes_e2e,
                            config=ver_config_e2e,
                            site_root=".",
                        )
                        ver_all_records_e2e.extend(recs)
                        included_urls = {
                            r.artifact_url for r in recs
                            if r.final_decision == "include" and r.artifact_type in ("image", "video")
                        }
                        included_texts = {
                            c.text_content for c, r in zip(candidates, recs)
                            if c.type == "text" and r.final_decision == "include" and c.text_content
                        }
                        last_extracted_content = _filter_content_to_included(
                            last_extracted_content, included_urls, included_texts
                        )

                    if config.enable_annotation:
                        await _save_annotation(sub_url, page_index)

                    # Persist the (post-verification) extracted content so step3 can
                    # still be run manually later if the user wants the legal/illegal
                    # classification.  Default pipeline no longer runs step3 itself.
                    row = {
                        "sub_url": sub_url,
                        "last_extracted_content": last_extracted_content,
                        "page_index": page_index,
                    }
                    f_step2.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_step2.flush()
                    step2_written += 1
            finally:
                f_step2.close()
            print(f"\n✅ Step 2 results saved → {step2_results_path} ({step2_written} pages)")
            print("ℹ️  Step 3 (legal/illegal classification) is no longer part of the default flow.")
            print("    Run it explicitly with `--run_mode step3` if needed.")

            if config.enable_verification and ver_all_records_e2e:
                summary_path = os.path.join(ver_config_e2e.output_dir, "verification_summary.json")
                summary = write_summary(ver_all_records_e2e, summary_path)
                print(f"\n🛡️  Verification summary → {summary_path}")
                for t, b in summary.get("by_type", {}).items():
                    if b.get("total", 0) == 0:
                        continue
                    print(f"   {t:6s}: included {b['final_included']}/{b['total']} "
                          f"(g1={b['gate1_observed_rate']:.2f} g2={b['gate2_download_rate']:.2f} "
                          f"g3={b['gate3_mime_rate']:.2f} g4={b['gate4_dedup_rate']:.2f} "
                          f"g5={b['gate5_not_halluc_rate']:.2f})")
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
