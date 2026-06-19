"""Unified evaluator for BaRA / browser-use runs.

Applies the following policies consistently:
  * Sub-link (URL) discovery: collect_urls-style URL collection
        - GT  = collect_urls(links_bfs.json)
        - Pred (browser-use) = general_link.json["urls"]
        - Pred (BaRA)        = collect_urls(links_bfs.json) from pred side
    -> per-topic TP/FP/FN, then MICRO + MACRO P/R/F1.
  * Data collection (image/video/text): per-page set comparison.
        - URL normalization (scheme/host lowercase, /index.html, trailing /,
          fragment, query stripped -- path only).
        - Text normalized to word set (lowercase + punct + whitespace).
        - None=exclude (page-modality where both empty -> skipped).
        - GT page filtered out if is_error_page(text_full) is True.
        - Pred page: if is_error_page(joined_pred_text) -> TEXT forced to 0
          (image/video unchanged, since 404 pages typically have no media URLs).
        - Missing pred page (URL miss): all modalities P=R=Acc=0 (weighted).

BaRA verification policy applied at pred-load time:
  text:        include iff gate1_observed (T1 or T2 with fuzzy_threshold = 0.1)
  image/video: include iff final_decision == "include" (5-gate full)

Usage:
    python eval_unified.py --run <pred_root> --gt-annotation <ann_root> \\
        --gt-links <links_root> --type bara|baseline \\
        --label my_run --out-json report.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE    = re.compile(r"\s+")
_404_PAT  = re.compile(
    r'\b(404|not found|page not found|forbidden|access denied|error\s*\d{3})\b',
    re.IGNORECASE)
_IMG_EXT  = re.compile(r"\.(jpg|jpeg|png|gif|bmp|webp|svg)(?:[?#]|$)", re.IGNORECASE)
_VID_EXT  = re.compile(r"\.(mp4|mov|avi|mkv|webm|flv|m4v|m3u8|ts)(?:[?#]|$)", re.IGNORECASE)
_URL_RE   = re.compile(r"https?://[^\s\)\]\"'<>]+")


def norm_url(u: str) -> str:
    if not u: return ""
    try: p = urlparse(u.strip())
    except Exception: return u.strip().lower()
    path = re.sub(r"/index\.html?$", "", p.path or "")
    if path.endswith("/") and len(path) > 1: path = path[:-1]
    return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"


def words(s: str) -> set[str]:
    s = _PUNCT_RE.sub(" ", (s or "").lower())
    return {w for w in _WS_RE.sub(" ", s).strip().split() if w}


def is_error_page(text: str) -> bool:
    if not text: return True
    ln = len(text)
    if ln < 200: return True
    if ln < 1200 and _404_PAT.search(text[:600]): return True
    return False


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r


def jac(a, b):
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def collect_urls_from_bfs(p: Path) -> set[str]:
    """visited_order union flatten(by_depth) -- mirrors verify_links_bfs.collect_urls."""
    if not p.is_file(): return set()
    try: d = json.load(open(p))
    except Exception: return set()
    out = set()
    for u in d.get("visited_order") or []:
        if isinstance(u, str) and u: out.add(norm_url(u))
    for _, urls in (d.get("by_depth") or {}).items():
        for u in urls or []:
            if isinstance(u, str) and u: out.add(norm_url(u))
    return out


@dataclass
class PredPage:
    page_url: str
    texts_joined: str
    text_words: set[str]
    images: set[str]
    videos: set[str]


def _parse_section(content: str, sec: str) -> str:
    h = re.search(rf"(?:^|\n)(?:- )?{sec}\(s\):", content or "", re.IGNORECASE)
    if not h: return ""
    s = h.end()
    nxt = re.search(r"(?:^|\n)(?:- )?(Text|Image|Video)\(s\):",
                    content[s:], re.IGNORECASE)
    return content[s: s + nxt.start()] if nxt else content[s:]


def _extract_urls(body: str, ext_re) -> set[str]:
    seen = set()
    for m in _URL_RE.finditer(body or ""):
        u = m.group(0).rstrip(".,;:)")
        if ext_re.search(u): seen.add(norm_url(u))
    if not seen:
        for m in _URL_RE.finditer(body or ""):
            u = m.group(0).rstrip(".,;:)")
            seen.add(norm_url(u))
    return seen


def _extract_text_lines(body: str) -> list[str]:
    lines = []
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith("- "): line = line[2:].strip()
        if re.match(r"^(text|image|video)\(s\)\s*:$", line, re.IGNORECASE): continue
        if re.match(r"^(none|no .*|none found)$", line, re.IGNORECASE): continue
        if len(line) < 2: continue
        lines.append(line)
    return lines


def load_browser_use_pages(topic_dir: Path) -> dict[str, PredPage]:
    """Pred from browser-use's json_page/page_*/data.json."""
    out: dict[str, PredPage] = {}
    page_root = topic_dir / "json_page"
    if not page_root.is_dir(): return out
    for pd in sorted(page_root.glob("page_*"),
                     key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1))):
        dp = pd / "data.json"
        if not dp.is_file(): continue
        try: d = json.load(open(dp))
        except Exception: continue
        url = norm_url(d.get("page_url", ""))
        if not url: continue
        texts = d.get("texts") or []
        joined = " ".join(texts)
        tw = set()
        for t in texts: tw |= words(t)
        imgs = {norm_url(u) for u in (d.get("image_urls") or []) if u}
        vids = {norm_url(u) for u in (d.get("video_urls") or []) if u}
        out[url] = PredPage(url, joined, tw, imgs, vids)
    return out


def load_bara_pages(topic_dir: Path) -> dict[str, PredPage]:
    """Pred from BaRA's step2_results.jsonl PLUS verification_records.jsonl filter.

    text:        include iff record.gate1_observed (T1 or T2 with thr 0.1)
    image/video: include iff record.final_decision == "include"

    If no verification_out: uses raw step2 extraction (no filter).
    """
    out: dict[str, PredPage] = {}
    step2 = topic_dir / "step2_results.jsonl"
    if not step2.is_file(): return out

    by_url: dict[str, dict] = {}
    for line in open(step2):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        sub_url = r.get("sub_url") or r.get("page_url") or ""
        if not sub_url: continue
        last = r.get("last_extracted_content") or {}
        content = last.get("content", "") if isinstance(last, dict) else ""
        url_n = norm_url(sub_url)
        text_body = _parse_section(content, "Text")
        img_body  = _parse_section(content, "Image")
        vid_body  = _parse_section(content, "Video")
        text_lines = _extract_text_lines(text_body)
        by_url.setdefault(url_n, {"texts": [], "images": set(), "videos": set()})
        by_url[url_n]["texts"].extend(text_lines)
        by_url[url_n]["images"] |= _extract_urls(img_body, _IMG_EXT)
        by_url[url_n]["videos"] |= _extract_urls(vid_body, _VID_EXT)

    veri_dir = topic_dir / "verification_out"
    if veri_dir.is_dir():
        text_inc: dict[str, set[str]] = {}
        img_inc:  dict[str, set[str]] = {}
        vid_inc:  dict[str, set[str]] = {}
        for host_dir in veri_dir.iterdir():
            if not host_dir.is_dir(): continue
            rec_path = host_dir / "verification_records.jsonl"
            if not rec_path.is_file(): continue
            for line in open(rec_path):
                try: rec = json.loads(line)
                except Exception: continue
                src = norm_url(rec.get("source_page") or "")
                if not src: continue
                atype = rec.get("artifact_type")
                if atype == "text":
                    if rec.get("gate1_observed"):
                        txt = rec.get("text_content") or rec.get("artifact_url") or ""
                        text_inc.setdefault(src, set()).add(txt)
                else:
                    if rec.get("final_decision") == "include":
                        url = norm_url(rec.get("artifact_url") or "")
                        if not url: continue
                        if atype == "image":
                            img_inc.setdefault(src, set()).add(url)
                        elif atype == "video":
                            vid_inc.setdefault(src, set()).add(url)
            tx_jsonl = host_dir / "artifacts" / "texts" / "included_texts.jsonl"
            if tx_jsonl.is_file():
                for line in open(tx_jsonl):
                    try: r = json.loads(line)
                    except Exception: continue
                    src = norm_url(r.get("source_page") or "")
                    txt = r.get("text") or ""
                    if src and txt:
                        text_inc.setdefault(src, set()).add(txt)
        all_urls = set(by_url) | set(text_inc) | set(img_inc) | set(vid_inc)
        for url in all_urls:
            texts_pool = list(text_inc.get(url, set()))
            joined = " ".join(texts_pool) if texts_pool \
                     else " ".join(by_url.get(url, {}).get("texts", []))
            tw = set()
            for t in texts_pool: tw |= words(t)
            imgs = img_inc.get(url, set())
            vids = vid_inc.get(url, set())
            out[url] = PredPage(url, joined, tw, imgs, vids)
    else:
        for url, d in by_url.items():
            joined = " ".join(d["texts"])
            tw = set()
            for t in d["texts"]: tw |= words(t)
            out[url] = PredPage(url, joined, tw, d["images"], d["videos"])
    return out


@dataclass
class GTPage:
    page_url: str
    text_full: str
    text_words: set[str]
    images: set[str]
    videos: set[str]


def load_gt(topic: str, ann_root: Path, links_root: Path) -> list[GTPage]:
    """annotation/<topic>/page_*.json + links_bfs.json by_depth (page order)."""
    short = topic.replace("web_", "", 1)
    d = ann_root / short
    if not d.is_dir(): return []
    bfs_path = links_root / f"web_{short}" / "links_bfs.json"
    flat: list[str] = []
    if bfs_path.is_file():
        bfs = json.load(open(bfs_path))
        for k in sorted((bfs.get("by_depth") or {}).keys(), key=int):
            flat.extend(bfs["by_depth"][k])
    out = []
    for fp in sorted(d.glob("page_*.json"),
                     key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1))):
        idx = int(re.search(r"page_(\d+)", fp.name).group(1))
        try: data = json.load(open(fp))
        except Exception: continue
        if is_error_page(data.get("text_full", "")): continue
        url = norm_url(flat[idx]) if idx < len(flat) else ""
        out.append(GTPage(
            page_url=url,
            text_full=data.get("text_full", ""),
            text_words=words(data.get("text_full", "")),
            images={norm_url(u) for u in (data.get("images") or []) if u},
            videos={norm_url(u) for u in (data.get("videos") or []) if u},
        ))
    return out


def find_topic_dir(pred_root: Path, topic: str, pred_type: str) -> Path | None:
    """Try common naming conventions; return whichever folder exists.

    BaRA convention:        <pred_root>/web_<topic>/
    browser-use convention: <pred_root>/https___<host>_<path>_index.html/
    """
    short = topic.replace("web_", "", 1)
    candidates: list[Path] = []
    if pred_type == "bara":
        candidates += [pred_root / f"web_{short}", pred_root / short]
    else:
        # Try any subfolder ending with the topic short name + _index.html
        # plus a few common patterns.
        candidates += [pred_root / f"web_{short}"]
        try:
            for p in pred_root.iterdir():
                if p.is_dir() and p.name.endswith(f"web_{short}_index.html"):
                    candidates.append(p)
                    break
                if p.is_dir() and p.name.endswith(f"web_{short}_index_html"):
                    candidates.append(p)
                    break
        except FileNotFoundError:
            pass
    for c in candidates:
        if c.is_dir(): return c
    return None


def eval_run(*, label: str, pred_root: Path, ann_root: Path, links_root: Path,
             pred_type: str, topics: list[str],
             pred_dead_text_zero: bool = True) -> dict:
    """Run Step 1 + Step 2 eval over `topics`, return summary dict."""
    s1_rows = []
    s1_tp = s1_fp = s1_fn = 0
    n_topic_ok = 0
    for t in topics:
        gt_urls = collect_urls_from_bfs(
            links_root / f"web_{t.replace('web_', '', 1)}" / "links_bfs.json")
        td = find_topic_dir(pred_root, t, pred_type)
        pred_urls: set[str] = set()
        if td is not None:
            if pred_type == "bara":
                pred_urls = collect_urls_from_bfs(td / "links_bfs.json")
            else:
                gl = td / "general_link.json"
                if gl.is_file():
                    try:
                        data = json.load(open(gl))
                        pred_urls = {norm_url(u)
                                     for u in (data.get("urls") or []) if u}
                    except Exception:
                        pass
        if pred_urls: n_topic_ok += 1
        tp = len(gt_urls & pred_urls)
        fp = len(pred_urls - gt_urls)
        fn = len(gt_urls - pred_urls)
        p, r = prf(tp, fp, fn)
        s1_rows.append({"topic": t, "gt": len(gt_urls), "pred": len(pred_urls),
                        "tp": tp, "fp": fp, "fn": fn, "p": p, "r": r})
        s1_tp += tp; s1_fp += fp; s1_fn += fn

    n = len(s1_rows)
    s1_micro_p, s1_micro_r = prf(s1_tp, s1_fp, s1_fn)
    s1_macro_p = sum(r["p"] for r in s1_rows) / n if n else 0.0
    s1_macro_r = sum(r["r"] for r in s1_rows) / n if n else 0.0

    mod_rows: dict[str, list] = {"image": [], "video": [], "text": []}
    n_pages_total = 0
    n_pages_missing_pred = 0
    n_pages_pred_dead_text = 0
    n_topics_with_gt = 0
    for t in topics:
        gt_pages = load_gt(t, ann_root, links_root)
        if not gt_pages: continue
        n_topics_with_gt += 1
        td = find_topic_dir(pred_root, t, pred_type)
        if td is None:
            pred_by_url: dict[str, PredPage] = {}
        else:
            pred_by_url = (load_bara_pages(td) if pred_type == "bara"
                           else load_browser_use_pages(td))
        for gt in gt_pages:
            n_pages_total += 1
            pr = pred_by_url.get(gt.page_url)
            if pr is None:
                n_pages_missing_pred += 1
                for mod in ("image", "video", "text"):
                    mod_rows[mod].append((t, gt.page_url, 0.0, 0.0, 0.0, "missing"))
                continue
            pred_dead = pred_dead_text_zero and is_error_page(pr.texts_joined)
            if pred_dead: n_pages_pred_dead_text += 1
            for mod, gt_set, pr_set in [("image", gt.images, pr.images),
                                         ("video", gt.videos, pr.videos),
                                         ("text",  gt.text_words, pr.text_words)]:
                if mod == "text" and pred_dead:
                    mod_rows[mod].append((t, gt.page_url, 0.0, 0.0, 0.0, "pred_dead"))
                    continue
                if not gt_set and not pr_set: continue
                tp = len(gt_set & pr_set); fp = len(pr_set - gt_set); fn = len(gt_set - pr_set)
                p, r = prf(tp, fp, fn); a = jac(gt_set, pr_set)
                mod_rows[mod].append((t, gt.page_url, p, r, a, "ok"))

    s2 = {}
    for mod in ("image", "video", "text"):
        rs = mod_rows[mod]; nr = len(rs)
        if nr == 0:
            s2[mod] = dict(n_pages=0, P=None, R=None, Acc=None); continue
        s2[mod] = dict(
            n_pages=nr,
            P=sum(r[2] for r in rs) / nr,
            R=sum(r[3] for r in rs) / nr,
            Acc=sum(r[4] for r in rs) / nr,
        )

    return {
        "label": label,
        "n_topics": n,
        "step1": {
            "n_topic_with_pred_data": n_topic_ok,
            "n_topic_total": n,
            "gt_urls": sum(r["gt"] for r in s1_rows),
            "pred_urls": sum(r["pred"] for r in s1_rows),
            "tp": s1_tp, "fp": s1_fp, "fn": s1_fn,
            "micro_p": s1_micro_p,
            "micro_r": s1_micro_r,
            "micro_f1": (2 * s1_micro_p * s1_micro_r /
                         (s1_micro_p + s1_micro_r)
                         if (s1_micro_p + s1_micro_r) else 0),
            "macro_p": s1_macro_p,
            "macro_r": s1_macro_r,
            "macro_f1": (2 * s1_macro_p * s1_macro_r /
                         (s1_macro_p + s1_macro_r)
                         if (s1_macro_p + s1_macro_r) else 0),
        },
        "step2": {
            "total_gt_pages": n_pages_total,
            "missing_pred_pages": n_pages_missing_pred,
            "pred_dead_text_pages": n_pages_pred_dead_text,
            "topics_with_gt": n_topics_with_gt,
            **s2,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="Path to the run output root (per-topic subfolders).")
    ap.add_argument("--type", choices=["bara", "baseline"], required=True,
                    help="bara = BaRA output format; baseline = browser-use.")
    ap.add_argument("--gt-annotation", type=Path, required=True,
                    help="GT annotation root: <root>/<topic>/page_*.json")
    ap.add_argument("--gt-links", type=Path, required=True,
                    help="GT links root: <root>/web_<topic>/links_bfs.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-pred-dead-text-zero", action="store_true",
                    help="Disable the pred-side dead-page text=0 policy.")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    topics = sorted([p.name.replace("web_", "", 1)
                     for p in args.gt_links.iterdir()
                     if p.is_dir() and p.name.startswith("web_")])

    res = eval_run(label=args.label, pred_root=args.run,
                   ann_root=args.gt_annotation, links_root=args.gt_links,
                   pred_type=args.type, topics=topics,
                   pred_dead_text_zero=not args.no_pred_dead_text_zero)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out_json:
        json.dump(res, open(args.out_json, "w"),
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
