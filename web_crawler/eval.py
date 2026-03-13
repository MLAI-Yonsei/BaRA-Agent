import argparse
import json
import re
from pathlib import Path


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m3u8", ".m4v")


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if "#" in url:
        url = url.split("#")[0]
    return url.rstrip("/") or url


def _extract_page_num(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not extract page number from: {path}")
    return int(match.group(1))


def load_annotation_page(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    images = raw.get("images") or []
    videos = raw.get("videos") or []
    return {
        "image_count": raw.get("image_count", 0),
        "images": {_normalize_url(url) for url in images},
        "video_count": raw.get("video_count", 0),
        "videos": {_normalize_url(url) for url in videos},
        "text_length": raw.get("text_length", 0),
        "text_full": (raw.get("text_full") or "").strip(),
    }


def load_all_annotations(annotation_root: Path) -> list[dict]:
    if not annotation_root.is_dir():
        return []
    pages = []
    for page_path in sorted(annotation_root.glob("page_*.json"), key=_extract_page_num):
        data = load_annotation_page(page_path)
        if data is not None:
            data["page_num"] = _extract_page_num(page_path)
            pages.append(data)
    return pages


def _is_image_url(value: str) -> bool:
    if not (isinstance(value, str) and value.strip().lower().startswith("http")):
        return False
    path = value.strip().lower().split("?")[0]
    return any(path.endswith(ext) for ext in IMAGE_EXT)


def _is_video_url(value: str) -> bool:
    if not (isinstance(value, str) and value.strip().lower().startswith("http")):
        return False
    path = value.strip().lower().split("?")[0]
    return any(path.endswith(ext) for ext in VIDEO_EXT)


def _load_content_json(page_root: Path, prefix: str) -> list:
    single = page_root / f"{prefix}_content.json"
    if single.exists():
        with open(single, "r", encoding="utf-8") as file:
            return json.load(file)
    items = []
    for batch_path in sorted(page_root.glob(f"{prefix}_content_batch_*.json")):
        with open(batch_path, "r", encoding="utf-8") as file:
            items.extend(json.load(file))
    return items


def load_merged_content_json(page_root: Path) -> list:
    return _load_content_json(page_root, "legal") + _load_content_json(page_root, "illegal")


def collect_images_videos_from_content(items: list) -> tuple[set, set]:
    images = set()
    videos = set()
    for item in items:
        if not isinstance(item, str):
            continue
        normalized = _normalize_url(item)
        if _is_image_url(item):
            images.add(normalized)
        elif _is_video_url(item):
            videos.add(normalized)
    return images, videos


def load_collected_text_for_page(legal_root: Path, illegal_root: Path, page_index: int) -> tuple[str, int]:
    parts = []
    for root in (legal_root, illegal_root):
        text_dir = root / "text" / f"page_{page_index}"
        if not text_dir.is_dir():
            continue
        text_files = sorted(
            text_dir.glob("text_*.txt"),
            key=lambda path: int(re.search(r"text_(\d+)", path.name).group(1)),
        )
        for text_file in text_files:
            parts.append(text_file.read_text(encoding="utf-8"))
    full_text = "\n".join(parts).strip()
    return full_text, len(full_text)


def load_all_collected_keyed_by_page(page_collect_root: Path, legal_root: Path, illegal_root: Path) -> dict[int, dict]:
    if not page_collect_root.is_dir():
        return {}
    result = {}
    for page_dir in sorted(page_collect_root.glob("page_*"), key=_extract_page_num):
        if not page_dir.is_dir():
            continue
        page_num = _extract_page_num(page_dir)
        items = load_merged_content_json(page_dir)
        images, videos = collect_images_videos_from_content(items)
        text_full, text_length = load_collected_text_for_page(legal_root, illegal_root, page_num)
        result[page_num] = {
            "page_num": page_num,
            "images": images,
            "videos": videos,
            "text_full": text_full,
            "text_length": text_length,
        }
    return result


def _text_to_line_set(text: str) -> set:
    return {line.strip() for line in (text or "").splitlines() if line.strip()}


def _text_to_word_set(text: str) -> set:
    return set(re.split(r"\s+", (text or "").strip())) - {""}


def _text_to_token_set(text: str, unit: str) -> set:
    return _text_to_word_set(text) if unit == "word" else _text_to_line_set(text)


def compute_set_metrics(gt_set: set, pred_set: set) -> dict:
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Recall": recall, "Precision": precision, "ACC": acc}


def aggregate_metrics(page_pairs: list[tuple[dict, dict]], text_metric_unit: str) -> dict:
    image_totals = {"TP": 0, "FP": 0, "FN": 0}
    video_totals = {"TP": 0, "FP": 0, "FN": 0}
    text_totals = {"TP": 0, "FP": 0, "FN": 0}

    for gt, pred in page_pairs:
        image_metrics = compute_set_metrics(gt["images"], pred["images"])
        video_metrics = compute_set_metrics(gt["videos"], pred["videos"])
        gt_tokens = _text_to_token_set(gt["text_full"], text_metric_unit)
        pred_tokens = _text_to_token_set(pred["text_full"], text_metric_unit)
        text_metrics = compute_set_metrics(gt_tokens, pred_tokens)

        for key in ("TP", "FP", "FN"):
            image_totals[key] += image_metrics[key]
            video_totals[key] += video_metrics[key]
            text_totals[key] += text_metrics[key]

    def _finalize(totals: dict) -> dict:
        tp = totals["TP"]
        fp = totals["FP"]
        fn = totals["FN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {"Recall": recall, "Precision": precision, "ACC": acc, "TP": tp, "FP": fp, "FN": fn}

    return {
        "image": _finalize(image_totals),
        "video": _finalize(video_totals),
        "text": _finalize(text_totals),
    }


def build_page_pairs(annotation_root: Path, legal_root: Path, illegal_root: Path, page_collect_root: Path) -> tuple[list[tuple[dict, dict]], set[int], set[int]]:
    gt_pages = load_all_annotations(annotation_root)
    pred_by_page = load_all_collected_keyed_by_page(page_collect_root, legal_root, illegal_root)

    gt_by_page = {page["page_num"]: page for page in gt_pages}
    common_page_nums = sorted(set(gt_by_page.keys()) & set(pred_by_page.keys()))
    page_pairs = [(gt_by_page[num], pred_by_page[num]) for num in common_page_nums]
    only_gt = set(gt_by_page.keys()) - set(pred_by_page.keys())
    only_pred = set(pred_by_page.keys()) - set(gt_by_page.keys())
    return page_pairs, only_gt, only_pred


def print_summary(page_pairs: list[tuple[dict, dict]], only_gt: set[int], only_pred: set[int], metrics: dict, text_metric_unit: str) -> None:
    print(f"Prepared {len(page_pairs)} matched page pairs.")
    if only_gt:
        print(f"Pages only in annotations: {sorted(only_gt)}")
    if only_pred:
        print(f"Pages only in collected results: {sorted(only_pred)}")

    print(f"[Text metric unit: {text_metric_unit}]")
    for data_type in ("image", "video", "text"):
        metric = metrics[data_type]
        print(
            f"{data_type.upper()}: "
            f"ACC={metric['ACC']:.4f}, "
            f"Recall={metric['Recall']:.4f}, "
            f"Precision={metric['Precision']:.4f} "
            f"(TP={metric['TP']}, FP={metric['FP']}, FN={metric['FN']})"
        )


def print_page_details(page_pairs: list[tuple[dict, dict]], text_metric_unit: str, sample_size: int) -> None:
    print("=" * 60)
    print("Per-page summary")
    print("=" * 60)
    for gt, pred in page_pairs:
        page_num = gt["page_num"]
        print(
            f"page_{page_num}: "
            f"gt images={len(gt['images'])}, videos={len(gt['videos'])}, text_len={gt['text_length']} | "
            f"pred images={len(pred['images'])}, videos={len(pred['videos'])}, text_len={pred['text_length']}"
        )

    print("\n" + "=" * 60)
    print("Video details")
    print("=" * 60)
    for gt, pred in page_pairs:
        page_num = gt["page_num"]
        gt_videos = gt["videos"]
        pred_videos = pred["videos"]
        tp_set = gt_videos & pred_videos
        fn_set = gt_videos - pred_videos
        fp_set = pred_videos - gt_videos
        if not gt_videos and not pred_videos:
            continue
        print(
            f"\n[page_{page_num}] GT={len(gt_videos)} | Pred={len(pred_videos)} | "
            f"TP={len(tp_set)}, FN={len(fn_set)}, FP={len(fp_set)}"
        )
        for label, values in (("FN", fn_set), ("FP", fp_set), ("TP", tp_set)):
            if values:
                print(f"  {label}:")
                for value in sorted(values):
                    print(f"    {value}")

    print("\n" + "=" * 60)
    print("Image details")
    print("=" * 60)
    for gt, pred in page_pairs:
        page_num = gt["page_num"]
        gt_images = gt["images"]
        pred_images = pred["images"]
        tp_set = gt_images & pred_images
        fn_set = gt_images - pred_images
        fp_set = pred_images - gt_images
        if not gt_images and not pred_images:
            continue
        print(
            f"\n[page_{page_num}] GT={len(gt_images)} | Pred={len(pred_images)} | "
            f"TP={len(tp_set)}, FN={len(fn_set)}, FP={len(fp_set)}"
        )
        for label, values in (("FN", fn_set), ("FP", fp_set)):
            if values:
                print(f"  {label}:")
                preview = sorted(values)[:sample_size]
                for value in preview:
                    print(f"    {value}")
                if len(values) > sample_size:
                    print(f"    ... {len(values) - sample_size} more")
        if tp_set:
            print(f"  TP: {len(tp_set)}")

    print("\n" + "=" * 60)
    print("Text details")
    print("=" * 60)
    for gt, pred in page_pairs:
        page_num = gt["page_num"]
        gt_tokens = _text_to_token_set(gt["text_full"], text_metric_unit)
        pred_tokens = _text_to_token_set(pred["text_full"], text_metric_unit)
        tp_set = gt_tokens & pred_tokens
        fn_set = gt_tokens - pred_tokens
        fp_set = pred_tokens - gt_tokens
        if not gt_tokens and not pred_tokens:
            continue
        print(
            f"\n[page_{page_num}] GT={len(gt_tokens)} | Pred={len(pred_tokens)} | "
            f"TP={len(tp_set)}, FN={len(fn_set)}, FP={len(fp_set)}"
        )
        if fn_set:
            print("  FN sample:")
            for value in sorted(fn_set)[:sample_size]:
                print(f"    {value}")
        if fp_set:
            print("  FP sample:")
            for value in sorted(fp_set)[:sample_size]:
                print(f"    {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate crawler outputs.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing annotations, legal, illegal, and json_page.")
    parser.add_argument("--annotation_root", type=str, default=None, help="Override annotation root path.")
    parser.add_argument("--legal_root", type=str, default=None, help="Override legal root path.")
    parser.add_argument("--illegal_root", type=str, default=None, help="Override illegal root path.")
    parser.add_argument("--page_collect_root", type=str, default=None, help="Override json_page root path.")
    parser.add_argument("--text_metric_unit", choices=["word", "line"], default="word", help="Token unit for text comparison.")
    parser.add_argument("--show_details", action="store_true", help="Print per-page details for text, image, and video.")
    parser.add_argument("--sample_size", type=int, default=15, help="Preview sample size for detail output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    annotation_root = Path(args.annotation_root) if args.annotation_root else data_root / "annotations"
    legal_root = Path(args.legal_root) if args.legal_root else data_root / "legal"
    illegal_root = Path(args.illegal_root) if args.illegal_root else data_root / "illegal"
    page_collect_root = Path(args.page_collect_root) if args.page_collect_root else data_root / "json_page"

    page_pairs, only_gt, only_pred = build_page_pairs(annotation_root, legal_root, illegal_root, page_collect_root)
    metrics = aggregate_metrics(page_pairs, args.text_metric_unit)

    print_summary(page_pairs, only_gt, only_pred, metrics, args.text_metric_unit)
    if args.show_details:
        print_page_details(page_pairs, args.text_metric_unit, args.sample_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
