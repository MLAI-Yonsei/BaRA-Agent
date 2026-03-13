import asyncio
from dotenv import load_dotenv
from browser_use import Agent, BrowserProfile
from browser_use.llm import ChatGoogle, ChatOllama
from browser_use.llm.messages import UserMessage
import os
import json
from datetime import datetime
import yt_dlp
from firecrawl import Firecrawl
import signal
import sys
import argparse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
import requests
import tempfile
import shutil
from playwright.async_api import async_playwright

load_dotenv()
os.environ["BUBUS_MAX_MEMORY"] = "2GB"
os.environ["BUBUS_MAX_HISTORY_SIZE"] = "1000"
os.environ["BUBUS_AUTO_CLEAR"] = "true"
interrupted = False

def create_llm(provider: str, model_name: str, api_key: Optional[str] = None, 
               temperature: float = 0.3, top_p: float = 0.8, seed: Optional[int] = 42,
               ollama_host: Optional[str] = None, ollama_api_key: Optional[str] = None):
    """Create an LLM instance for the selected provider.
    
    Args:
        provider: ``google`` or ``ollama``
        model_name: model identifier to use
        api_key: Google API key when required
        temperature: generation temperature
        top_p: top-p sampling value
        seed: seed value for providers that support it
        ollama_host: Ollama server host URL
        ollama_api_key: optional API key for remote Ollama-backed models
        
    Returns:
        ChatGoogle or ChatOllama instance
    """
    if provider.lower() == "google":
        return ChatGoogle(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            api_key=api_key
        )
    elif provider.lower() == "ollama":
        host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if ollama_api_key:
            os.environ["OLLAMA_API_KEY"] = ollama_api_key
            print(f"🔑 OLLAMA_API_KEY environment variable set (for Ollama Cloud authentication)")
        # gemini_api_key = ollama_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # if gemini_api_key and not os.getenv("GEMINI_API_KEY"):
        #     os.environ["GEMINI_API_KEY"] = gemini_api_key
        
        return ChatOllama(
            model=model_name,
            host=host,
            timeout=240.0
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'google' or 'ollama'.")


def signal_handler(signum, frame):
    global interrupted
    print("\n🛑 Interrupt signal received. The current task will finish before shutdown...")
    interrupted = True
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
FEEDBACK_HISTORY_FILE = "feedback_history.json"


def is_url_downloadable(url, timeout=5):
    """Return whether the URL looks downloadable."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code in [200, 301, 302]
    except:
        return False


def filter_downloadable_urls(urls):
    """Filter a list down to downloadable URLs."""
    downloadable = []
    for url in urls:
        if is_url_downloadable(url):
            downloadable.append(url)
    return downloadable


async def extract_video_url_from_llm(formatted_prompt, api_key, model_name, llm_provider="google", ollama_host=None, ollama_api_key=None):
    llm = create_llm(
        provider=llm_provider,
        model_name=model_name,
        api_key=api_key,
        temperature=0.3,
        top_p=0.8,
        seed=42,
        ollama_host=ollama_host,
        ollama_api_key=ollama_api_key
    )
    messages = [UserMessage(content=formatted_prompt)]
    response = await llm.ainvoke(messages)
    if hasattr(response, 'completion'):
        verified_content = response.completion
    else:
        verified_content = str(response)
    return verified_content


def parse_video_urls_from_response(response_text, output_path, sub_url=None):
    """Extract JSON from an LLM response and append it to a file.
    
    Args:
        response_text: raw LLM response text
        output_path: destination file path such as ``video_urls.json``
        sub_url: current page URL used as optional metadata
    """
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text,
                               re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(
                r"\{[^{}]*\"video_urls\"[^{}]*\[[^\]]*\][^{}]*\}",
                response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                else:
                    raise ValueError("JSON could not be found.")

        data = json.loads(json_str)
        new_video_urls = data.get("video_urls", [])
        existing_video_urls = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    existing_video_urls = existing_data.get("video_urls", [])
            except Exception as e:
                print(f"⚠️ Failed to read existing file: {e}")
        seen = set(existing_video_urls)
        all_video_urls = existing_video_urls.copy()
        for url in new_video_urls:
            if url not in seen:
                all_video_urls.append(url)
                seen.add(url)
        
        accumulated_data = {"video_urls": all_video_urls}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(accumulated_data, f, ensure_ascii=False, indent=2)

        print(
            f"✅ {output_path} append save completed "
            f"(existing {len(existing_video_urls)} + new {len(new_video_urls)} = total {len(all_video_urls)})"
        )
        if sub_url:
            print(f"   Source: {sub_url}")
        return {"video_urls": new_video_urls}
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"Response content (first 500 chars): {response_text[:500]}")
        return None
def check_sections(content: str):
    text_exists = False
    text_matches = list(re.finditer(r"(?:^|\n)(?:- )?Text(?:\(s\))?:", content))
    for text_match in text_matches:
        start_pos = text_match.end()
        next_section = re.search(r"\n(?:- )?(Image|Video)\(s\):",
                                 content[start_pos:])
        if next_section:
            text_section = content[start_pos:start_pos + next_section.start()]
        else:
            next_url = re.search(r"\n\[https?://", content[start_pos:])
            if next_url:
                text_section = content[start_pos:start_pos + next_url.start()]
            else:
                text_section = content[start_pos:]
        items = re.findall(r"^\s+-\s+(.+)$", text_section, re.MULTILINE)
        items = [
            i.strip() for i in items if i.strip()
            and not re.match(r"^(none|No .*)$", i.strip(), re.IGNORECASE)
        ]
        if len(items) == 0:
            all_lines = [line.strip() for line in text_section.split('\n') if line.strip()]
            for line in all_lines:
                if (not re.match(r"^(none|No .*|None found)$", line, re.IGNORECASE) and
                    len(line) > 3):
                    items.append(line)
                    break
        
        if len(items) > 0:
            text_exists = True
            break
    image_extensions = r"\.(jpg|jpeg|png|gif|bmp|webp|svg)"
    url_pattern = r"https?://[^\s\)]+"

    image_exists = False
    image_matches = list(re.finditer(r"(?:^|\n)(?:- )?Image\(s\):", content))

    for img_match in image_matches:

        start_pos = img_match.end()

        next_section = re.search(r"\n(?:- )?(Text|Video)\(s\):",
                                 content[start_pos:])
        if next_section:
            img_section = content[start_pos:start_pos + next_section.start()]
        else:
            next_url = re.search(r"\n\[https?://", content[start_pos:])
            if next_url:
                img_section = content[start_pos:start_pos + next_url.start()]
            else:
                img_section = content[start_pos:]
        if (re.search(image_extensions, img_section, re.IGNORECASE)
                or re.search(url_pattern, img_section)):
            image_exists = True
            break
    video_extensions = r"\.(mp4|mov|avi|mkv|webm|flv|m4v|m3u8|ts|html)"
    video_exists = False
    video_matches = list(re.finditer(r"(?:^|\n)(?:- )?Video\(s\):", content))
    for vid_match in video_matches:

        start_pos = vid_match.end()

        next_section = re.search(r"\n(?:- )?(Text|Image)\(s\):",
                                 content[start_pos:])
        if next_section:
            vid_section = content[start_pos:start_pos + next_section.start()]
        else:
            next_url = re.search(r"\n\[https?://", content[start_pos:])
            if next_url:
                vid_section = content[start_pos:start_pos + next_url.start()]
            else:
                vid_section = content[start_pos:]
        if (re.search(video_extensions, vid_section, re.IGNORECASE)
                or re.search(url_pattern, vid_section)):
            video_exists = True
            break
    text_start = re.search(r"(?:^|\n)(?:- )?Text(?:\(s\))?:", content)
    img_start = re.search(r"(?:^|\n)(?:- )?Image\(s\):", content)
    vid_start = re.search(r"(?:^|\n)(?:- )?Video\(s\):", content)

    text_block = ""
    img_block = ""
    vid_block = ""

    if text_start and img_start:
        text_block = content[text_start.start():img_start.start()]
    elif text_start and not img_start:
        if vid_start:
            text_block = content[text_start.start():vid_start.start()]
        else:
            text_block = content[text_start.start():]
    if img_start and vid_start:
        img_block = content[img_start.start():vid_start.start()]
    if vid_start:
        vid_block = content[vid_start.start():]

    return {
        "text_exists": text_exists,
        "image_exists": image_exists,
        "video_exists": video_exists,
        "text_block": text_block,
        "image_block": img_block,
        "video_block": vid_block
    }


def build_multimodal_classifier_prompt(items_text: str) -> str:
    return f"""
                    You are a **multimodal content-safety classifier**.

                    You will receive a list of extracted items from various web pages.
                    Each item may contain text, an image, or a video URL.
                    Your task is to carefully analyze **both textual meaning and visual content** (if present)
                    and determine whether each item is **illegal/inappropriate** or **safe/legal**.

                    ---

                    ### Evaluation Rules

                    Classify an item as **illegal** if it **contains, depicts, or promotes** any of the following:
                    1. **Adult / Sexual content**
                    – Explicit nudity, sexual acts, erotic imagery, pornography, or terms such as
                        "19+", "sex", "porn", "nude", "adult video", "AV", "erotic", etc.
                    – Images or videos clearly depicting nudity or sexual behavior.

                    2. **Gambling / Betting**
                    – Mentions or visuals of casino, slot machines, sports betting, baccarat, roulette, or phrases such as
                        "casino", "bet", "slot", "sports betting", "baccarat".

                    3. **Piracy / Unauthorized Streaming**
                    – Offers or links to free downloads, torrents, or illegal streaming of copyrighted content.
                        Phrases such as "watch movies for free", "watch free", "torrent", "streaming link".

                    4. **Violence / Hate / Crime**
                    – Depicts or encourages physical violence, weapons, hate symbols, illegal drug use, or criminal activity.

                    5. **Other Illegal Activities**
                    – Promotion of scams, fraud, human trafficking, or any content violating law or public morals.

                    Classify an item as **legal** if it is:
                    - Neutral, educational, informational, artistic, or otherwise safe for general audiences.
                    - A technical element (e.g., "HD", "mp4", "gif", "search", "login") with no illicit meaning.
                    - An ordinary website section name (e.g., "movies", "drama", "news", "home", "support").

                    If an item is meaningless, repetitive, or purely decorative, you may ignore it.

                    ---

                    ### Input
                    Here are the extracted items (text lines and/or media URLs) to evaluate:

                    {items_text if items_text else "(no items found)"}

                    If any item is an image or video URL, examine its **visual content directly** using your multimodal capability.
                    If it is plain text, analyze its **semantic meaning**.

                    ---

                    ### Output Format
                    Return **only** a valid JSON object in this exact shape — no extra text, no comments:

                    ```json
                    {{
                        "illegal": [
                            "example of illegal or inappropriate content (text or URL)...",
                            ...
                        ],
                        "legal": [
                            "example of legal and appropriate content (text or URL)...",
                            ...
                        ]
                    }}
                    ```
                    """


def build_video_classifier_prompt(video_urls: List[str]) -> str:
    return f"""
        You are a **multimodal content-safety classifier**.

        You will receive a list of video URLs extracted from web pages.
        Your task is to carefully analyze the URLs and determine whether each video URL is **illegal/inappropriate** or **safe/legal**.

        ---

        ### Evaluation Rules

        Classify a video URL as **illegal** if it **contains, depicts, or promotes** any of the following:
        1. **Adult / Sexual content**
        – Explicit nudity, sexual acts, erotic imagery, pornography, or terms such as
            "19+", "sex", "porn", "nude", "adult video", "AV", "erotic", etc.
        – URLs clearly indicating adult or sexual content.

        2. **Gambling / Betting**
        – Mentions or visuals of casino, slot machines, sports betting, baccarat, roulette, or phrases such as
            "casino", "bet", "slot", "sports betting", "baccarat".

        3. **Piracy / Unauthorized Streaming**
        – Offers or links to free downloads, torrents, or illegal streaming of copyrighted content.
            Phrases such as "watch movies for free", "watch free", "torrent", "streaming link".

        4. **Violence / Hate / Crime**
        – Depicts or encourages physical violence, weapons, hate symbols, illegal drug use, or criminal activity.

        5. **Other Illegal Activities**
        – Promotion of scams, fraud, human trafficking, or any content violating law or public morals.

        Classify a video URL as **legal** if it is:
        - Neutral, educational, informational, artistic, or otherwise safe for general audiences.
        - A technical element (e.g., "HD", "mp4", "gif", "search", "login") with no illicit meaning.
        - An ordinary website section name (e.g., "movies", "drama", "news", "home", "support").

        If a URL is meaningless, repetitive, or purely decorative, you may ignore it.

        ---

        ### Input
        Here are the video URLs to evaluate:

        {json.dumps(video_urls, ensure_ascii=False, indent=2)}

        ---

        ### Output Format
        Return **only** a valid JSON object in this exact shape — no extra text, no comments:

        ```json
        {{
            "illegal": [
                "https://example.com/illegal-video-url...",
                ...
            ],
            "legal": [
                "https://example.com/legal-video-url...",
                ...
            ]
        }}
        """

# MAX_SCROLL_ACTIONS = 5
# MAX_SCROLL_SECONDS = 300
# NO_GROWTH_CYCLES = 2
MAX_SCROLL_ACTIONS = 5
MAX_SCROLL_SECONDS = 300
NO_GROWTH_CYCLES = 2
def build_page_extraction_task(sub_url: str) -> str:
    return f"""

                    Step 1: Go to {sub_url} and STAY on this specific page.
                    Step 2: Restricted Extraction (Single Page Only)
                   - DO NOT click any links, DO NOT navigate to other URLs.
                   - Mandatory: Scroll to the bottom repeatedly until no new content loads.
                   - Safety cap: stop scrolling if you have performed {MAX_SCROLL_ACTIONS} scroll actions, or spent {MAX_SCROLL_SECONDS} seconds on scrolling, or if page height does not increase for {NO_GROWTH_CYCLES} consecutive scroll cycles. After hitting any cap, proceed to extraction immediately.
                    Step 3: Comprehensive Data Extraction
                    - Extract EVERY piece of raw text, image URLs, and video URLs.
                    - All available image URLs (ending with .jpg, .png, .gif, .jpeg, .webp, .svg) that appear on that page.
                    - All available video URLs (ending with .mp4, .mov, .avi, .wmv, .flv, .mkv, .webm, .m4v, .m3u8, .ts) that appear on that page.
                    Step 4: Strict Output Format for Post-Processing
                    - Provide the data using the following specific delimiters.
                    - Do NOT summarize or use '...' to truncate text. List everything found.
                    - If a category is empty, write 'None' between the tags.

                    [Page URL]

                    - Text(s): (Extract all text blocks here. Each paragraph or block must be on a new line. No bullet points or decorative symbols)

                    - Image(s): (List all full Image URLs here, one per line)

                    - Video(s): (List all full Video URLs here, one per line)

                    """
def extract_json_from_text(raw_text):
    if not isinstance(raw_text, str):
        if isinstance(raw_text, (tuple, list)):
            raw_text = ' '.join(str(item) for item in raw_text) if raw_text else str(raw_text)
        else:
            raw_text = str(raw_text)
    
    if not raw_text:
        raise ValueError(f"❌ raw_text is empty: {type(raw_text)}")
    match = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    raise ValueError("❌ Could not find a JSON block (```json ... ```) or JSON object ({...}).")
def extract_folder_name(url: str):
    if not url or not isinstance(url, str):
        raise ValueError(f"extract_folder_name: url is not valid: {url} (type: {type(url)})")
    return re.sub(r'[\\/:*?"<>|]', '_', url)


def build_json_page_dir(root_folder: str, page_index: int) -> str:
    return os.path.join(root_folder, "json_page", f"page_{page_index}")
async def audit_website(url: str):
    """Audit a page and return image, video, link, and text data as JSON."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, wait_until="networkidle")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)

        site_data = await page.evaluate("""
            () => {
                const images = Array.from(document.querySelectorAll('img')).map(img => img.src);

                const videos = Array.from(document.querySelectorAll('video, iframe[src*="youtube"], iframe[src*="vimeo"]')).map(v => {
                    if (v.tagName.toLowerCase() === 'video') {
                        return v.src || (v.querySelector('source') ? v.querySelector('source').src : null);
                    }
                    return v.src || v.dataset.src;
                }).filter(src => src !== null);

                const links = Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                })).filter(l => l.href.startsWith('http'));

                const textContent = document.body.innerText;

                return {
                    image_count: images.length,
                    images: images,
                    video_count: videos.length,
                    videos: videos,
                    link_count: links.length,
                    links: links,
                    text_length: textContent.length,
                    text_full: textContent
                };
            }
        """)

        await browser.close()
        return site_data
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".wmv", ".flv", ".mkv", ".webm", ".m4v",
             ".m3u8", ".ts", ".html")
def is_image(url: str):
    if not isinstance(url, str):
        return False
    url = clean_url(url)
    if not url.startswith(('http://', 'https://')):
        return False

    lower = url.lower()
    path_part = lower.split("?")[0].split("#")[0]
    if path_part.endswith(IMAGE_EXT):
        return True
    if re.search(r"(format|fm)=(jpg|jpeg|png|gif|webp)", lower):
        return True

    return False
def clean_url(url: str) -> str:
    """Trim surrounding whitespace and a leading markdown list marker."""
    if not isinstance(url, str):
        return url
    url = url.strip()
    if url.startswith('- '):
        url = url[2:].strip()
    return url


def normalize_image_url(url: str) -> str:
    """Normalize image URLs so duplicates can be removed.
    
    This removes query parameters and fragments so the same asset can be
    compared consistently even when size or format parameters differ.
    
    Examples:
    - https://example.com/image.jpg?w=400&h=300 -> https://example.com/image.jpg
    - https://example.com/image.jpg#section -> https://example.com/image.jpg
    - https://example.com/image.jpg?w=400#section -> https://example.com/image.jpg
    """
    if not isinstance(url, str):
        return url
    
    url = clean_url(url)
    if '?' in url:
        url = url.split('?')[0]
    if '#' in url:
        url = url.split('#')[0]
    
    return url
def is_html_video(url: str):
    """Use yt_dlp to check whether an HTML URL resolves to actual video content."""
    if not url.lower().endswith('.html'):
        return False

    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and 'formats' in info:
                video_formats = [
                    f for f in info.get('formats', [])
                    if f.get('vcodec') != 'none'
                ]
                return len(video_formats) > 0
            if info and info.get('url'):
                content_type = info.get('http_headers',
                                        {}).get('Content-Type', '')
                if 'video' in content_type.lower():
                    return True
    except Exception:
        pass

    return False
def is_video(url: str):
    """
    Check whether a URL likely points to video content.

    HTML URLs are resolved with yt_dlp to verify that they expose actual
    streaming formats.
    """
    if not isinstance(url, str):
        return False
    url = clean_url(url)
    if not url.startswith(('http://', 'https://')):
        return False

    lower = url.lower()
    url_without_query = lower.split('?')[0] if '?' in lower else lower
    if url_without_query.endswith(VIDEO_EXT):
        if url_without_query.endswith('.html'):
            return is_html_video(url)
        return True
    if re.search(r'(/mp4/|/video/|/videos/|\.mp4\?|\.webm\?|\.m3u8\?|\.ts\?|\.mov\?|\.mkv\?|\.avi\?|\.flv\?|\.m4v\?)', lower):
        return True
    if re.search(r'(vid-|video-|mp4-|cdn.*video)', lower):
        return True
    if re.search(r"(format|fm)=(mp4|mov|mkv|webm|ts|m3u8)", lower):
        return True

    return False
def download_image(url, save_path):
    """Download and save an image.
    
    Returns:
        bool: ``True`` on success, otherwise ``False``
    """
    from PIL import Image
    import io

    try:
        url = clean_url(str(url))
        if not url.startswith(('http://', 'https://')):
            print(f"❌ Invalid URL format: {url}")
            return False
        
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            path_part = url.lower().split("?")[0].split("#")[0]
            if path_part.endswith(".svg"):
                save_path_svg = os.path.splitext(save_path)[0] + ".svg"
                with open(save_path_svg, "wb") as f:
                    f.write(r.content)
                print(f"📷 Saved image (SVG): {save_path_svg}")
                return True
            img = Image.open(io.BytesIO(r.content))
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(
                    img,
                    mask=img.split()[-1] if img.mode in ('RGBA',
                                                         'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            save_path_jpg = os.path.splitext(save_path)[0] + '.jpg'
            img.save(save_path_jpg, 'JPEG', quality=95)
            print(f"📷 Saved image: {save_path_jpg}")
            return True
        else:
            print(f"❌ Image download failed: {url} (status code: {r.status_code})")
            return False
    except Exception as e:
        print(f"❌ Image error: {e}")
        return False


def download_video(url, save_path):
    """Download and save a video.
    
    Returns:
        bool: ``True`` on success, otherwise ``False``
    """
    try:
        url = clean_url(str(url))
        if not url.startswith(('http://', 'https://')):
            print(f"❌ Invalid video URL format: {url}")
            return False
        url_lower = url.lower()
        is_direct_mp4 = url_lower.endswith('.mp4') or '.mp4?' in url_lower or '/mp4/' in url_lower
        
        if is_direct_mp4:
            print(f"📥 Direct MP4 link detected, downloading with requests: {url[:100]}...")
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://example.com/'
                }
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                if total_size > 0:
                    print(f"📊 File size: {total_size / (1024*1024):.2f} MB")
                downloaded = 0
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if downloaded % (1024 * 1024) == 0:
                                    print(f"📥 Download progress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
                
                print(f"✅ Video saved: {save_path} ({downloaded / (1024*1024):.2f} MB)")
                return True
            except requests.exceptions.RequestException as e:
                print(f"❌ Video download via requests failed: {url} | {e}")
                print(f"🔄 Retrying with yt_dlp...")
            except Exception as e:
                print(f"❌ Error during direct download: {url} | {e}")
                print(f"🔄 Retrying with yt_dlp...")
        ydl_opts = {
            'format': 'best',
            'outtmpl': save_path,
            'quiet': False,
            'no_warnings': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"🎥 Saved video: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Video download failed: {url} | {e}")
        return False
def process_txt_file(txt_path, first_url, sub_url=None, page_index=None):
    """Parse and persist text, image, and video items.
    
    Args:
        txt_path: text file to process
        first_url: root URL used to derive the output folder name
        sub_url: current page URL for metadata, optional
        page_index: current page index for folder naming, optional
    """
    processed_video_urls = set()
    video_urls_json_path = "video_urls.json"
    if os.path.exists(video_urls_json_path):
        try:
            with open(video_urls_json_path, "r", encoding="utf-8") as f:
                video_urls_data = json.load(f)
                video_urls = video_urls_data.get("video_urls", [])
                processed_video_urls = set(video_urls)
                print(
                    f"📹 Found {len(processed_video_urls)} video URLs in video_urls.json. Skipping duplicate downloads."
                )
        except Exception as e:
            print(f"⚠️ Failed to read video_urls.json: {e}")
    root_folder = extract_folder_name(first_url)
    if txt_path == "verified_content_present.txt" or os.path.basename(txt_path) == "verified_content_present.txt":
        print("📄 Detected verified_content_present.txt - using legal_content.json and illegal_content.json")
        if page_index is not None:
            page_output_dir = build_json_page_dir(root_folder, page_index)
            legal_content_path = os.path.join(page_output_dir, "legal_content.json")
            illegal_content_path = os.path.join(page_output_dir, "illegal_content.json")
        else:
            legal_content_path = "legal_content.json"
            illegal_content_path = "illegal_content.json"

        data_json = {"legal": [], "illegal": []}

        try:
            with open(legal_content_path, "r", encoding="utf-8") as f:
                legal_data = json.load(f)
                data_json["legal"] = legal_data if isinstance(legal_data, list) else [legal_data]
            print(f"✅ {len(data_json['legal'])} legal items loaded")
        except FileNotFoundError:
            try:
                with open("legal_content.json", "r", encoding="utf-8") as f:
                    legal_data = json.load(f)
                    data_json["legal"] = legal_data if isinstance(legal_data, list) else [legal_data]
                print(f"✅ {len(data_json['legal'])} legal items loaded (cwd)")
            except FileNotFoundError:
                print("⚠️ legal_content.json does not exist yet. Starting with an empty list.")
            except Exception as e:
                print(f"⚠️ Failed to read legal_content.json: {e}")
        except Exception as e:
            print(f"⚠️ Failed to read legal_content.json: {e}")

        try:
            with open(illegal_content_path, "r", encoding="utf-8") as f:
                illegal_data = json.load(f)
                data_json["illegal"] = illegal_data if isinstance(illegal_data, list) else [illegal_data]
            print(f"✅ {len(data_json['illegal'])} illegal items loaded")
        except FileNotFoundError:
            try:
                with open("illegal_content.json", "r", encoding="utf-8") as f:
                    illegal_data = json.load(f)
                    data_json["illegal"] = illegal_data if isinstance(illegal_data, list) else [illegal_data]
                print(f"✅ {len(data_json['illegal'])} illegal items loaded (cwd)")
            except FileNotFoundError:
                print("⚠️ illegal_content.json does not exist yet. Starting with an empty list.")
            except Exception as e:
                print(f"⚠️ Failed to read illegal_content.json: {e}")
        except Exception as e:
            print(f"⚠️ Failed to read illegal_content.json: {e}")
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        try:
            json_str = extract_json_from_text(raw_text)
            data_json = json.loads(json_str)
        except ValueError as e:
            print(f"⚠️ JSON extraction failed: {e}")
            print("Starting with an empty list.")
            data_json = {"legal": [], "illegal": []}
    print("📁 Folder name:", root_folder)
    if page_index is not None:
        for section in ["legal", "illegal"]:
            for sub in ["text", "image", "video"]:
                page_folder = f"{root_folder}/{section}/{sub}/page_{page_index}"
                os.makedirs(page_folder, exist_ok=True)
        print(f"📁 Output folder: {root_folder}/{{legal|illegal}}/{{text|image|video}}/page_{page_index}")
        if sub_url:
            print(f"📄 Source page: {sub_url}")
    else:
        for section in ["legal", "illegal"]:
            for sub in ["text", "image", "video"]:
                os.makedirs(f"{root_folder}/{section}/{sub}", exist_ok=True)
    if sub_url and page_index is not None:
        annotations_dir = f"{root_folder}/annotations"
        os.makedirs(annotations_dir, exist_ok=True)
        annotation_path = f"{annotations_dir}/page_{page_index}.json"
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                audit_result = asyncio.run(audit_website(sub_url))
            else:
                with ThreadPoolExecutor() as pool:
                    audit_result = pool.submit(asyncio.run, audit_website(sub_url)).result()
            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(audit_result, f, ensure_ascii=False, indent=4)
            print(f"📋 Saved annotation: {annotation_path}")
        except Exception as e:
            print(f"⚠️ Annotation creation failed ({sub_url}): {e}")
    download_counts = {
        "legal": {"text": 0, "image": 0, "video": 0},
        "illegal": {"text": 0, "image": 0, "video": 0}
    }
    
    for section in ["legal", "illegal"]:
        items = data_json[section]
        text_i = image_i = video_i = 0

        for item in items:
            if is_image(item):
                cleaned_item = clean_url(item)
                if page_index is not None:
                    save_path = f"{root_folder}/{section}/image/page_{page_index}/img_{image_i}.jpg"
                else:
                    save_path = f"{root_folder}/{section}/image/img_{image_i}.jpg"
                if download_image(cleaned_item, save_path):
                    image_i += 1
                    download_counts[section]["image"] = image_i
                else:
                    print(f"⚠️ Image download failed (skipping): {cleaned_item[:80]}...")
                continue
            else:
                if isinstance(item, str) and item.startswith(('http://', 'https://')):
                    if 'unsplash.com' in item.lower() or 'image' in item.lower() or 'photo' in item.lower():
                        print(f"🔍 Debug: URL detected, but not recognized as an image: {item[:100]}...")
                        print(f"   is_image() result: {is_image(item)}")
            is_video_result = is_video(item)
            if is_video_result:
                cleaned_item = clean_url(item)
                if cleaned_item in processed_video_urls:
                    print(f"⏭️ Skipping video URL (already handled by Firecrawl): {cleaned_item[:100]}...")
                    continue
                print(f"🎬 Trying to download video URL: {cleaned_item[:150]}...")
                if not isinstance(cleaned_item, str) or not cleaned_item.startswith(
                    ('http://', 'https://')):
                    print(f"⚠️ Invalid video URL format: {cleaned_item} (type: {type(cleaned_item)})")
                    continue

                if page_index is not None:
                    save_path = f"{root_folder}/{section}/video/page_{page_index}/video_{video_i}.mp4"
                else:
                    save_path = f"{root_folder}/{section}/video/video_{video_i}.mp4"
                if download_video(cleaned_item, save_path):
                    video_i += 1
                    download_counts[section]["video"] = video_i
                continue
            else:
                if isinstance(item, str) and item.startswith(('http://', 'https://')):
                    if 'mp4' in item.lower() or 'video' in item.lower() or 'vid-' in item.lower():
                        print(f"🔍 Debug: URL detected, but not recognized as a video: {item[:100]}...")
            if page_index is not None:
                text_path = f"{root_folder}/{section}/text/page_{page_index}/text_{text_i}.txt"
            else:
                text_path = f"{root_folder}/{section}/text/text_{text_i}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(str(item))
            print(f"✏️ Saved text: {text_path}")
            text_i += 1
            download_counts[section]["text"] = text_i
        if page_index is not None and sub_url:
            for section in ["legal", "illegal"]:
                items = data_json.get(section, [])
                if items:
                    metadata = {
                        "source_url": sub_url,
                        "page_index": page_index,
                        "category": section,
                        "text_count": download_counts[section]["text"],
                        "image_count": download_counts[section]["image"],
                        "video_count": download_counts[section]["video"],
                        "timestamp": datetime.now().isoformat()
                    }
                    for sub in ["text", "image", "video"]:
                        page_folder = f"{root_folder}/{section}/{sub}/page_{page_index}"
                        metadata_path = f"{page_folder}/metadata.json"
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                    print(f"💾 Saved metadata: {root_folder}/{section}/{{text|image|video}}/page_{page_index}/metadata.json")
                    print(f"   📊 Actual downloaded counts - text: {download_counts[section]['text']}, image: {download_counts[section]['image']}, video: {download_counts[section]['video']}")


async def process_video_urls_json(video_urls, first_url, sub_url, page_index, api_key,
                                  model_name, llm_provider="google", ollama_host=None, ollama_api_key=None):
    """Classify video URLs into legal or illegal groups and save them.
    
    Args:
        video_urls: video URLs extracted from the current page
        first_url: root URL used for output folder naming
        sub_url: current page URL
        page_index: current page index
        api_key: Google API key
        model_name: model identifier
    """
    try:
        if not first_url or not isinstance(first_url, str):
            print(f"❌ first_url is not valid: {first_url} (type: {type(first_url)})")
            print("⚠️ Skipping video_urls processing.")
            return
        if not video_urls or not isinstance(video_urls, list):
            print(f"⚠️ video_urls is not valid: {type(video_urls)}")
            return

        if len(video_urls) == 0:
            print("⚠️ No video URLs to process.")
            return

        print(f"📹 {len(video_urls)} video URLs will be classified as legal/illegal...")
        video_classifier_prompt = build_video_classifier_prompt(video_urls)
        llm = create_llm(
            provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=0.3,
            top_p=0.8,
            seed=42,
            ollama_host=ollama_host,
            ollama_api_key=ollama_api_key
        )

        from browser_use.llm.messages import UserMessage
        messages = [UserMessage(content=video_classifier_prompt)]
        response = await llm.ainvoke(messages)

        if hasattr(response, 'completion'):
            classified_text = response.completion
        else:
            classified_text = str(response)
        if not classified_text:
            print("❌ AI response is empty.")
            return
        try:
            with open("video_classification_response.txt", "w", encoding="utf-8") as f:
                f.write(classified_text)
            print("📝 AI response saved to video_classification_response.txt.")
        except Exception as e:
            print(f"⚠️ Failed to save AI response: {e}")
        try:
            json_str = extract_json_from_text(classified_text)
            classified_data = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"❌ JSON extraction/parsing failed: {e}")
            print(f"Response content (first 500 chars): {classified_text[:500]}")
            try:
                start_idx = classified_text.find('{')
                end_idx = classified_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = classified_text[start_idx:end_idx]
                    classified_data = json.loads(json_str)
                    print("✅ JSON parsing succeeded with the fallback method")
                else:
                    raise ValueError("JSON could not be found.")
            except Exception as e2:
                print(f"❌ Fallback JSON parsing also failed: {e2}")
                return
        if not isinstance(classified_data, dict):
            print(f"❌ classified_data is not a dictionary: {type(classified_data)}")
            return
        
        if "legal" not in classified_data or "illegal" not in classified_data:
            print(f"❌ classified_data does not contain 'legal' or 'illegal' keys.")
            print(f"Actual keys: {list(classified_data.keys())}")
            return
        root_folder = extract_folder_name(first_url)
        for section in ["legal", "illegal"]:
            page_folder = f"{root_folder}/{section}/video/page_{page_index}"
            os.makedirs(page_folder, exist_ok=True)

        print(f"📁 Video output folder: {root_folder}/{{legal|illegal}}/video/page_{page_index}")
        print(f"📄 Source page: {sub_url}")
        metadata = {
            "source_url": sub_url,
            "page_index": page_index,
            "total_videos": len(video_urls),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        for section in ["legal", "illegal"]:
            video_urls_list = classified_data.get(section, [])
            if not isinstance(video_urls_list, list):
                print(f"⚠️ {section} data is not a list: {type(video_urls_list)}")
                continue
            valid_urls = []
            for url in video_urls_list:
                if url and isinstance(url, str) and url.strip():
                    valid_urls.append(url.strip())
                else:
                    print(f"⚠️ Filtered out invalid URL ({section}): {url} (type: {type(url)})")
            
            print(f"📊 {section}: original {len(video_urls_list)} -> valid {len(valid_urls)}")
            
            video_i = 0
            page_folder = f"{root_folder}/{section}/video/page_{page_index}"
            for video_url in valid_urls:
                if is_video(video_url) or True:
                    save_path = f"{page_folder}/firecrawl_video_{video_i}.mp4"
                    if download_video(video_url, save_path):
                        video_i += 1
            if valid_urls:
                section_metadata = metadata.copy()
                section_metadata["category"] = section
                section_metadata["video_count"] = video_i
                section_metadata["video_urls"] = valid_urls
                
                metadata_path = f"{page_folder}/metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(section_metadata, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved metadata: {metadata_path}")
                print(f"   📊 Successfully downloaded videos: {video_i} (attempted: {len(valid_urls)})")

        print(
            f"✅ video_urls.json processing completed: legal {len(classified_data.get('legal', []))}, illegal {len(classified_data.get('illegal', []))}"
        )

    except Exception as e:
        print(f"❌ Error while processing video_urls.json: {e}")


def collect_long_term_memory(history):
    """Collect all long-term memory entries from history."""
    all_memories = []

    for i, hist_item in enumerate(history.history):
        if hasattr(hist_item, 'result') and hist_item.result:
            for j, result in enumerate(hist_item.result):
                if hasattr(result,
                           'long_term_memory') and result.long_term_memory:
                    all_memories.append({
                        'history_index': i,
                        'result_index': j,
                        'memory': result.long_term_memory
                    })

    return all_memories


def _content_without_attachments(raw_content: str) -> str:
    """Remove duplicated attachment blocks from extracted content.

    Only the main body is kept so Step 3 does not process the same item twice.
    """
    if not raw_content or "Attachments:" not in raw_content:
        return raw_content or ""
    return raw_content.split("Attachments:")[0].rstrip()


def split_content_by_lines(
        last_extracted_content_list: List[Dict]) -> List[Dict]:
    separated_entries = []
    for item in last_extracted_content_list:
        history_idx = item.get("history_index")
        result_idx = item.get("result_index")
        raw_content = item.get("content", "")
        lines = [
            line.strip() for line in raw_content.split("\n") if line.strip()
        ]
        for line in lines:
            separated_entries.append({
                "history_index": history_idx,
                "result_index": result_idx,
                "line_content": line
            })
    return separated_entries


def collect_extracted_content(history):
    """Collect all extracted_content entries from history as a fallback."""
    all_content = []

    for i, hist_item in enumerate(history.history):
        if hasattr(hist_item, 'result') and hist_item.result:
            for j, result in enumerate(hist_item.result):
                if hasattr(result,
                           'extracted_content') and result.extracted_content:
                    all_content.append({
                        'history_index': i,
                        'result_index': j,
                        'content': result.extracted_content
                    })

    return all_content


def get_failure_analysis_prompt(original_task, long_term_memories,
                                attempt_number):
    """Build a Gemini prompt for failure analysis using long-term memory."""
    prompt = f"""
You are an expert in analyzing the causes of web scraping task failures and suggesting improvements.

[Original Task]
{original_task}

[Long-term Memory so far (Attempt {attempt_number})]
{json.dumps(long_term_memories, ensure_ascii=False, indent=2)}

[Analysis Request]
Please analyze why the above task failed and provide the following:

1. Failure Analysis:
   - What type of failure it is (e.g., insufficient data, access permissions, technical issues, etc.)
   - The specific point of failure
   - Why this failure occurred

2. Improvement Suggestions:
   - Specific instructions that should be added to the prompt
   - Parts that should be clarified
   - Missing considerations that should be included

3. Improved Prompt:
   - A new version of the original task prompt with improvements
   - Concrete steps that address the identified failure causes

Respond in JSON format:
{{
    "failure_analysis": "Detailed analysis of the failure",
    "failure_type": "Type of failure",
    "improvements": ["Improvement 1", "Improvement 2"],
    "improved_prompt": "Improved task prompt"
}}

"""
    return prompt


async def get_feedback_from_gemini(llm, prompt):
    """Request feedback from Gemini."""
    try:
        from browser_use.llm.messages import UserMessage
        messages = [UserMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        if hasattr(response, 'completion'):
            return response.completion
        else:
            return str(response)
    except Exception as e:
        print(f"Gemini feedback request failed: {e}")
        return None


def save_feedback_history(attempt_number, original_task, long_term_memories,
                          feedback, improved_prompt):
    """Save feedback history along with long-term memory context."""
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'attempt_number': attempt_number,
        'original_task': original_task,
        'long_term_memories': long_term_memories,
        'feedback': feedback,
        'improved_prompt': improved_prompt
    }

    try:
        if os.path.exists(FEEDBACK_HISTORY_FILE):
            with open(FEEDBACK_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []

        history.append(feedback_data)
        with open(FEEDBACK_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"Feedback history saved: {FEEDBACK_HISTORY_FILE}")
    except Exception as e:
        print(f"Failed to save feedback history: {e}")


def save_extracted_content_list(extracted_content_list,
                                filename="extracted_content_list.json"):
    """Save the extracted content list to a JSON file."""
    try:
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'total_count': len(extracted_content_list),
            'extracted_contents': extracted_content_list
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"📁 Extracted content list saved: {filename}")
        print(f"   - Total {len(extracted_content_list)}items saved")
        return True
    except Exception as e:
        print(f"❌ Failed to save extracted content list: {e}")
        return False


def load_extracted_content_list(filename="extracted_content_list.json"):
    """Load a previously saved extracted content list."""
    try:
        if not os.path.exists(filename):
            print(f"⚠️ File does not exist: {filename}")
            return []

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📂 Loaded extracted content list: {filename}")
        print(f"   - Total {data.get('total_count', 0)}items loaded")
        return data.get('extracted_contents', [])
    except Exception as e:
        print(f"❌ Failed to load extracted content list: {e}")
        return []


def parse_gemini_feedback(feedback_text):
    """Parse a JSON payload from Gemini feedback text."""
    try:
        start_idx = feedback_text.find('{')
        end_idx = feedback_text.rfind('}') + 1

        if start_idx != -1 and end_idx != -1:
            json_str = feedback_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print("JSON response could not be found")
            return None
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return None


def get_success_evaluation_prompt(original_task, evaluation_data):
    """Build the Gemini success-evaluation prompt using the last extracted content."""
    prompt = f"""
You are an expert in evaluating the success of web scraping tasks.

[Original Task]
{original_task}

[Last Extracted Content (Final Result)]
{json.dumps(evaluation_data, ensure_ascii=False, indent=2)}

[Evaluation Request]
Please determine whether the above task was successful or not.

Success Criteria (Task is SUCCESSFUL if ANY of the following is true):
1. Links were successfully extracted from the page (even if only a few)
2. Text content was extracted and contains relevant information
3. Images or videos were found and extracted
4. The scraping process completed without major errors
5. The extracted content is related to the original task requirements

IMPORTANT: Be GENEROUS in your evaluation. If the task extracted ANY useful data (links, text, images, videos), consider it a SUCCESS. Only mark as FAILURE if:
- No data was extracted at all
- The extraction process completely failed
- The extracted content is completely irrelevant to the task

Specific Evaluation Points:
- Check if links array contains valid URLs (even if just a few)
- Check if text content was extracted (even if brief)
- Check if images or videos were found
- Consider the task successful if it gathered useful information, even if not perfect

Respond in JSON format:
{{
    "is_successful": true/false,
    "reason": "Detailed reason for success/failure based on what was actually extracted",
    "extracted_data_summary": "Summary of what was actually extracted and why it should be considered success/failure"
}}
"""
    return prompt


async def evaluate_success_with_gemini(llm, original_task, evaluation_data):
    """Evaluate success with Gemini using the last extracted content."""
    try:
        from browser_use.llm.messages import UserMessage
        evaluation_prompt = get_success_evaluation_prompt(
            original_task, evaluation_data)
        messages = [UserMessage(content=evaluation_prompt)]

        response = await llm.ainvoke(messages)
        feedback_text = None
        if hasattr(response, 'completion'):
            feedback_text = response.completion
            if feedback_text and len(str(feedback_text).strip()) > 0:
                print(f"🔍 Debug: Extracted response from response.completion (length: {len(feedback_text)})")
        if (not feedback_text or len(str(feedback_text).strip()) == 0) and hasattr(response, 'content'):
            feedback_text = response.content
            if feedback_text and len(str(feedback_text).strip()) > 0:
                print(f"🔍 Debug: Extracted response from response.content (length: {len(feedback_text)})")
        if (not feedback_text or len(str(feedback_text).strip()) == 0) and hasattr(response, 'text'):
            feedback_text = response.text
            if feedback_text and len(str(feedback_text).strip()) > 0:
                print(f"🔍 Debug: Extracted response from response.text (length: {len(feedback_text)})")
        if (not feedback_text or len(str(feedback_text).strip()) == 0) and hasattr(response, '__dict__'):
            response_dict = response.__dict__
            import re
            for key, value in response_dict.items():
                if key != 'usage' and value and isinstance(value, str) and len(value.strip()) > 10:
                    if re.search(r'```json|"is_successful"|"failure_analysis"', value, re.IGNORECASE):
                        feedback_text = value
                        print(f"🔍 Debug: found a JSON-formatted response in response.{key} (length: {len(value)})")
                        break
                    elif len(value.strip()) > 100:
                        feedback_text = value
                        print(f"🔍 Debug: found a long string in response.{key} (length: {len(value)})")
                        break
        if not feedback_text or len(str(feedback_text).strip()) == 0:
            str_response = str(response)
            is_object_repr = (str_response.startswith('<') and 
                            'object at 0x' in str_response and 
                            len(str_response) < 100)
            if not is_object_repr and len(str_response) > 100:
                feedback_text = str_response
                print(f"🔍 Debug: Extracted response from str(response) (length: {len(str_response)})")
        if not feedback_text or len(str(feedback_text).strip()) == 0:
            print("⚠️ Warning: could not find a success/failure judgment response. Using the default (success).")
            if evaluation_data and len(evaluation_data) > 0:
                return True, "Response extraction failed, but data exists, so it is treated as success.", ""
            else:
                return False, "Response extraction failed and no data is available.", ""

        parsed_result = parse_gemini_feedback(feedback_text)
        if parsed_result:
            return (
                parsed_result.get("is_successful", False),
                parsed_result.get("reason", ""),
                parsed_result.get("extracted_data_summary", ""),
            )
        else:
            print("Failed to parse success/failure judgment")
            if evaluation_data and len(evaluation_data) > 0:
                print("⚠️ Parsing failed, but data exists, so treating it as success.")
                return True, "Parsing failed, but data extraction succeeded.", ""
            return False, "Parsing failed.", ""
    except Exception as e:
        print(f"Gemini success/failure judgment failed: {e}")
        if evaluation_data and len(evaluation_data) > 0:
            print("⚠️ An exception occurred, but data exists, so treating it as success.")
            return True, f"An exception occurred: {e}, but data extraction succeeded.", ""
        return False, f"Error: {e}", ""


async def run_step1(wanted,
                    first_url,
                    api_key,
                    model_name,
                    max_depth,
                    max_width,
                    max_pages,
                    max_attempts: int = 10,
                    llm_provider: str = "google",
                    ollama_host: Optional[str] = None,
                    ollama_api_key: Optional[str] = None):
    original_task = f"""                
                You are a precise but lightweight BFS crawler that collects only internal links.

                [Start URL] {first_url}
                [Limits] max_depth = {max_depth}, max_width = {max_width}, max_pages = {max_pages}, output_path = links_bfs.json

                [RULES]
                1) Same-site only: links within the registrable domain of Start URL and its subdomains.
                2) BFS: use a FIFO queue of (url, depth). Init with (Start URL, 0).
                For each dequeued page:
                    a) Visit it and extract <a href> in DOM order
                    b) Normalize & filter (see 3,4)
                    c) Keep first ≤ max_width unique children for this parent
                    d) Enqueue (child, depth+1) if depth+1 ≤ max_depth
                Stop when visited pages ≥ max_pages or queue is empty.
                3) Normalization:
                - Remove fragments (#…)
                - Keep query keys only: [page,p,pg,offset,start,sort,category,board]
                - http/https and trailing slashes considered equal
                - Deduplicate by (scheme, host, path, kept_query)
                4) Exclude:
                - Paths containing: [/privacy, /terms, /login, /signup]
                - File extensions: [.pdf,.zip,.rar,.7z,.apk,.dmg,.exe]
                5) (Optional) Pagination hints ("next", "more", "load more", or ?page=):
                - may be enqueued at the same depth.

                [OUTPUT]
                Return ONLY this JSON and also save to [links_bfs.json]:
                {{
                "start_url": "[https://example.com/]",
                "max_depth": {max_depth},
                "max_width": {max_width},
                "max_pages": {max_pages},
                "visited_order": [...],
                "by_depth": {{"0": [start_url], "1":[...], ...}}
                }}
                """
    llm = create_llm(
        provider=llm_provider,
        model_name=model_name,
        api_key=api_key,
        temperature=0.6,
        top_p=1.0,
        seed=42,
        ollama_host=ollama_host,
        ollama_api_key=ollama_api_key
    )
    current_task = original_task
    for attempt in range(1, max_attempts + 1):
        agent = None
        try:
            print(f"[{attempt}/{max_attempts}] Attempting...")
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="browseruse-step1-")
            if llm_provider.lower() == "google":
                llm_timeout_val = 240
                step_timeout_val = 300
            else:  # ollama
                llm_timeout_val = 240
                step_timeout_val = 360

            agent = Agent(
                task=current_task,
                llm=llm,
                browser_profile=BrowserProfile(
                    headless=True,
                    devtools=False,
                    user_data_dir=temp_dir,
                    disable_security=True,
                    extra_chromium_args=[
                        '--disable-dev-shm-usage', '--no-sandbox',"--disable-extensions"],
                ),
                max_actions_per_step=1,
                llm_timeout=llm_timeout_val,
                step_timeout=step_timeout_val,
            )
            if llm_provider.lower() == "google":
                total_timeout = 1600
            else:  # ollama
                total_timeout = 2000
            
            try:
                history = await asyncio.wait_for(agent.run(), timeout=total_timeout)
            except asyncio.TimeoutError:
                timeout_minutes = total_timeout // 60
                print(f":alarm_clock: Timeout occurred - Agent task exceeded {total_timeout} seconds ({timeout_minutes} minutes).")
                raise Exception("Task stopped due to timeout")

            print(f"\nHistory length: {len(history)}")
            try:
                agent.stop()
                print("Agent memory cleanup completed")
            except Exception as e:
                print(f"Error while cleaning up agent: {e}")
            long_term_memories = collect_long_term_memory(history)
            print(f"Long-term memory count: {len(long_term_memories)}")
            extracted_content = collect_extracted_content(history)
            print(f"Extracted content count: {len(extracted_content)}")
            last_extracted_content = extracted_content[
                -1] if extracted_content else None
            if last_extracted_content:
                print(
                    f"Using the last extracted content for success/failure judgment: {last_extracted_content['content'][:100]}..."
                )
                evaluation_data = [{
                    'content': last_extracted_content['content']
                }]
            else:
                print("No extracted content found; treating it as empty data")
                evaluation_data = []
            print("\nGemini is evaluating success/failure...")
            is_successful, reason, data_summary = await evaluate_success_with_gemini(
                llm, original_task, evaluation_data)

            print(f"\nGemini judgment result:")
            print(f"Success: {'Success' if is_successful else 'Failure'}")
            print(f"Reason: {reason}")
            print(f"Data summary: {data_summary}")

            if is_successful:
                print("Success")
                if last_extracted_content:
                    print(f"\n--- Final extracted content ---")
                    print(
                        f"History index: {last_extracted_content['history_index']}")
                    print(f"Result index: {last_extracted_content['result_index']}")
                    print(f"Content: {last_extracted_content['content']}")
                last_result = history.history[
                    -1].result
                if last_result and hasattr(last_result[-1], "attachments"):
                    attachments = last_result[-1].attachments
                    print("Attachment path:", attachments)

                    if attachments:
                        attachment_path = attachments[0]
                        with open("last_attachment.txt", "a") as f:
                            f.write(attachment_path + "\n")
                        try:
                            import shutil
                            if os.path.exists(attachment_path):
                                shutil.copy2(attachment_path, "links_bfs.json")
                                print(f"✅ links_bfs.json copied successfully: {attachment_path} -> links_bfs.json")
                                with open("links_bfs.json", "r", encoding="utf-8") as f:
                                    links_data = json.load(f)
                                    print(f"✅ Collected {len(links_data.get('visited_order', []))} total links")
                            else:
                                print(f"⚠️ attachment File does not exist: {attachment_path}")
                        except Exception as e:
                            print(f"⚠️ Failed to copy links_bfs.json: {e}")
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print("Temporary directory cleanup completed")
                except Exception as e:
                    print(f"Error while cleaning up temporary directory: {e}")
                
                return history, attachment_path
            else:
                analysis_prompt = get_failure_analysis_prompt(
                    original_task, long_term_memories, attempt)
                feedback = await get_feedback_from_gemini(llm, analysis_prompt)

                if feedback:
                    print(f"\nGemini feedback:")
                    print(feedback)
                    parsed_feedback = parse_gemini_feedback(feedback)
                    if parsed_feedback:
                        print(f"\n🔍 Parsed feedback:")
                        print(
                            f"Failure type: {parsed_feedback.get('failure_type', 'Unknown')}"
                        )
                        print(
                            f"Improvements: {parsed_feedback.get('improvements', [])}"
                        )
                        improved_prompt = parsed_feedback.get(
                            'improved_prompt', current_task)
                        if improved_prompt != current_task:
                            print(f"\nPrompt improved!")
                            current_task = improved_prompt
                        save_feedback_history(attempt, original_task,
                                              long_term_memories, feedback,
                                              improved_prompt)
                    else:
                        print("Feedback parsing failed")
                else:
                    print("Gemini feedback request failed")

                print(f"\nUsing the improved prompt for the next attempt...")
        except Exception as e:
            print(f"❌ Error occurred: {e}, retrying...")
        finally:
            if agent:
                try:
                    agent.stop()
                except:
                    pass
            try:
                import shutil
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    print("Maximum attempts exceeded: task failed")
    return None


async def process_single_sub_url(sub_url, wanted, api_key, model_name, max_attempts, 
                                  first_url, firecrawl_api_key, llm_provider, 
                                  ollama_host, ollama_api_key, processed_urls, total_urls, llm,
                                  step2_union_retry_attempts: int = 2):
    """Process a single sub-URL and return the result."""
    print(f"📄 [{processed_urls}/{total_urls}] Processing URL: {sub_url}")
    
    success = False
    last_extracted_content = None
    
    for attempt in range(1, max_attempts + 1):
        if interrupted:
            print("🛑 Stopping the task due to the interrupt signal.")
            break
        print(sub_url)
        if firecrawl_api_key:
            try:
                print(f"🔥 Running Firecrawl: {sub_url}")
                firecrawl = Firecrawl(api_key=firecrawl_api_key)
                doc = firecrawl.scrape(f"{sub_url}",
                                       formats=["markdown"])
                formatted_prompt = f"""
                Extract video streaming URLs from the following markdown:
                {doc.markdown}
                Focus on URLs that point to embedded video players or streaming sources, such as those in iframes or links associated with playback buttons. Ignore internal navigation or non-streaming links.
                Return only the relevant video streaming URLs (e.g., ending with .html for embeds, or .m3u8/.ts for streams) in the following format:

                [https://..., ...]

                - When finished, return ONLY this JSON shape and also save the SAME JSON to video_urls.json:
                ```json
                {{
                    "video_urls": ["https://...", "..."]
                }}
                ```

                """
                firecrawl_result = await extract_video_url_from_llm(
                    formatted_prompt, api_key, model_name, llm_provider, ollama_host, ollama_api_key)
                print("Downloading streaming video url...")
                video_urls_json_path = "video_urls.json"
                video_data = parse_video_urls_from_response(
                    firecrawl_result, video_urls_json_path, sub_url=sub_url)

                if video_data and video_data.get("video_urls"):
                    current_page_video_urls = video_data.get("video_urls", [])
                    
                    print(
                        f"\n📹 Classifying and saving {len(current_page_video_urls)} video URLs from the current page as legal/illegal..."
                    )
                    await process_video_urls_json(
                        current_page_video_urls, first_url, sub_url, 
                        processed_urls - 1, api_key, model_name, llm_provider, ollama_host, ollama_api_key)
                else:
                    print("⚠️ No video URLs were extracted.")
            except Exception as e:
                print(f"⚠️ Firecrawl execution failed: {e}")
        else:
            print(
                "ℹ️ Skipping Firecrawl because firecrawl_api_key was not provided.")
        current_task = build_page_extraction_task(sub_url)

        print(f"[{attempt}/{max_attempts}] Attempting...")

        agent = None
        temp_dir = None
        try:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="browseruse-step2-")

            agent = Agent(
                task=current_task,
                llm=llm,
                browser_profile=BrowserProfile(
                    headless=True,
                    devtools=False,
                    user_data_dir=temp_dir,
                    disable_security=True,
                    extra_chromium_args=['--disable-dev-shm-usage', '--no-sandbox']
                

                ),
                max_actions_per_step=
                1,
                llm_timeout=180 if llm_provider.lower() == "google" else 240,
                step_timeout=240 if llm_provider.lower() == "google" else 360,
            )
            step2_total_timeout = 500 if llm_provider.lower() == "google" else 1000
            try:
                history = await asyncio.wait_for(agent.run(),
                                                 timeout=step2_total_timeout)
            except asyncio.TimeoutError:
                timeout_minutes = step2_total_timeout // 60
                print(f":alarm_clock: Timeout occurred - Agent task exceeded {step2_total_timeout} seconds ({timeout_minutes} minutes).")
                agent.stop()
                raise Exception("Task stopped due to timeout")

            print(f"\nHistory length: {len(history)}")
            try:
                agent.stop()
                print("Agent memory cleanup completed")
            except Exception as e:
                print(f"Error while cleaning up agent: {e}")
            long_term_memories = collect_long_term_memory(history)
            print(f"Long-term memory count: {len(long_term_memories)}")
            extracted_content = collect_extracted_content(history)
            print(f"Extracted content count: {len(extracted_content)}")
            last_extracted_content = extracted_content[
                -1] if extracted_content else None
            if last_extracted_content:
                print(
                    f"Using the last extracted content for success/failure judgment: {last_extracted_content['content'][:]}..."
                )
                evaluation_data = [{
                    'content':
                    last_extracted_content['content']
                }]
            else:
                print("No extracted content found; treating it as empty data")
                evaluation_data = []
            print("\nGemini is evaluating success/failure...")
            is_successful, reason, data_summary = await evaluate_success_with_gemini(
                llm, current_task, evaluation_data)

            print(f"\nGemini judgment result:")
            print(f"Success: {'Success' if is_successful else 'Failure'}")
            print(f"Reason: {reason}")
            print(f"Data summary: {data_summary}")

            if is_successful:
                print("✅ Success")
                if last_extracted_content:
                    print(f"\n--- Final extracted content ---")
                    print(
                        f"History index: {last_extracted_content['history_index']}"
                    )
                    print(
                        f"Result index: {last_extracted_content['result_index']}"
                    )
                    print(
                        f"Content: {last_extracted_content['content']}")
                result = check_sections(
                    last_extracted_content['content'])
                print(
                    f"🔍 Section check result: text={result['text_exists']}, image={result['image_exists']}, video={result['video_exists']}"
                )
                all_attempts_content = [last_extracted_content]
                if (not result["image_exists"]) and (not result["video_exists"]):
                    print(
                        "⚠️ Image and video are both missing. Running again."
                    )
                    print(
                        f"   - text_exists: {result['text_exists']}"
                    )
                    print(
                        f"   - image_exists: {result['image_exists']}"
                    )
                    print(
                        f"   - video_exists: {result['video_exists']}"
                    )
                    for retry_attempt in range(step2_union_retry_attempts):
                        retry_agent = None
                        retry_temp_dir = None
                        retry_content = None
                        try:
                            import tempfile
                            retry_temp_dir = tempfile.mkdtemp(
                                prefix="browseruse-step2-retry-")

                            retry_agent = Agent(
                                task=current_task,
                                llm=llm,
                                browser_profile=BrowserProfile(
                                    headless=True,
                                    devtools=False,
                                    user_data_dir=retry_temp_dir,
                                    disable_security=True,
                                    extra_chromium_args=[
                                        '--disable-dev-shm-usage',
                                        '--no-sandbox'
                                    ]
                                ),
                                max_actions_per_step=
                                1,
                                llm_timeout=180 if llm_provider.lower() == "google" else 240,
                                step_timeout=240 if llm_provider.lower() == "google" else 360,
                            )
                            history = None
                            step3_total_timeout = 1000 if llm_provider.lower() == "google" else 2000
                            history = await asyncio.wait_for(
                                retry_agent.run(), timeout=step3_total_timeout)

                        except asyncio.TimeoutError:
                            timeout_minutes = step3_total_timeout // 60
                            print(f":alarm_clock: Timeout occurred - Agent task exceeded {step3_total_timeout} seconds ({timeout_minutes} minutes).")
                        finally:
                            if retry_agent:
                                try:
                                    retry_agent.stop()
                                except:
                                    pass
                            if retry_temp_dir:
                                try:
                                    import shutil
                                    shutil.rmtree(
                                        retry_temp_dir,
                                        ignore_errors=True)
                                except:
                                    pass
                        if not history:
                            print(f"❌ Retry attempt {retry_attempt + 1}: no history found - moving to the next attempt")
                            continue
                        extracted_content = collect_extracted_content(
                            history)
                        retry_content = extracted_content[
                            -1] if extracted_content else None
                        if not retry_content:
                            print(f"❌ Retry attempt {retry_attempt + 1}: no content was extracted - moving to the next attempt")
                            continue
                        all_attempts_content.append(retry_content)
                        retry_result = check_sections(
                            retry_content['content'])
                        print(
                            f"🔍 Retry attempt {retry_attempt + 1} result: text={retry_result['text_exists']}, image={retry_result['image_exists']}, video={retry_result['video_exists']}"
                        )
                    print(f"📊 Merging the union of results from {len(all_attempts_content)} attempts...")
                    merged_text_items = set()
                    merged_image_items = {}  # {normalized_url: original_url}
                    merged_video_items = set()
                    
                    for attempt_content in all_attempts_content:
                        attempt_result = check_sections(attempt_content['content'])
                        if attempt_result['text_block']:
                            text_items = re.findall(r"^\s+-\s+(.+)$", attempt_result['text_block'], re.MULTILINE)
                            for item in text_items:
                                item_clean = item.strip()
                                if item_clean and not re.match(r"^(none|No .*)$", item_clean, re.IGNORECASE):
                                    merged_text_items.add(item_clean)
                            text_section_match = re.search(r"(?:- )?Text(?:\(s\))?:\s*\n(.*?)(?=\n(?:- )?(?:Image|Video)\(s\):|\n\[https?://|$)", 
                                                          attempt_result['text_block'], re.DOTALL)
                            if text_section_match:
                                text_section = text_section_match.group(1)
                                lines = text_section.split('\n')
                                for line in lines:
                                    line_clean = line.strip()
                                    if (line_clean and 
                                        not line_clean.startswith('- ') and 
                                        not re.match(r"^(none|No .*|None found)$", line_clean, re.IGNORECASE) and
                                        len(line_clean) > 1):
                                        merged_text_items.add(line_clean)
                        if attempt_result['image_block']:
                            attempt_idx = all_attempts_content.index(attempt_content)
                            print(f"🔍 Attempt {attempt_idx + 1} image_block (first 500 chars):\n{attempt_result['image_block'][:500]}")
                            image_items = re.findall(r"^\s+-\s+(.+)$", attempt_result['image_block'], re.MULTILINE)
                            extracted_from_list = []
                            for item in image_items:
                                item_clean = item.strip()
                                if item_clean and not re.match(r"^(none|No .*|None found)$", item_clean, re.IGNORECASE):
                                    if item_clean.startswith(('http://', 'https://')):
                                        url_clean = clean_url(item_clean)
                                        if url_clean:
                                            normalized_url = normalize_image_url(url_clean)
                                            if normalized_url not in merged_image_items or len(url_clean) > len(merged_image_items[normalized_url]):
                                                merged_image_items[normalized_url] = url_clean
                                            extracted_from_list.append(url_clean)
                            if not extracted_from_list:
                                image_section_match = re.search(r"(?:- )?Image\(s\):\s*\n(.*?)(?=\n(?:- )?(?:Text|Video)\(s\):|\n\[https?://|$)", 
                                                               attempt_result['image_block'], re.DOTALL)
                                if image_section_match:
                                    image_section = image_section_match.group(1)
                                    image_urls = re.findall(r"https?://[^\s\)\]\>\"\']+", image_section)
                                    extracted_from_url_pattern = []
                                    for url in image_urls:
                                        url_clean = clean_url(url.strip())
                                        url_clean = re.sub(r'[.,;:!?]+$', '', url_clean)
                                        if url_clean and url_clean.startswith(('http://', 'https://')):
                                            normalized_url = normalize_image_url(url_clean)
                                            if normalized_url not in merged_image_items or len(url_clean) > len(merged_image_items[normalized_url]):
                                                merged_image_items[normalized_url] = url_clean
                                            extracted_from_url_pattern.append(url_clean)
                                    
                                    if extracted_from_url_pattern:
                                        print(f"  📷 Image extracted via URL pattern ({len(extracted_from_url_pattern)}): {extracted_from_url_pattern[:3]}...")
                            if extracted_from_list:
                                print(f"  📷 Image extracted from list format ({len(extracted_from_list)}): {extracted_from_list[:3]}...")
                        if attempt_result['video_block']:
                            attempt_idx = all_attempts_content.index(attempt_content)
                            print(f"🔍 Attempt {attempt_idx + 1} video_block:\n{attempt_result['video_block'][:500]}")
                            video_items = re.findall(r"^\s+-\s+(.+)$", attempt_result['video_block'], re.MULTILINE)
                            extracted_from_list = []
                            image_extensions_pattern = r'\.(jpg|jpeg|png|gif|bmp|webp|svg)(\?|$|/)'
                            for item in video_items:
                                item_clean = item.strip()
                                if item_clean and not re.match(r"^(none|No .*|None found)$", item_clean, re.IGNORECASE):
                                    if item_clean.startswith(('http://', 'https://')):
                                        if re.search(image_extensions_pattern, item_clean, re.IGNORECASE):
                                            continue
                                        url_clean = clean_url(item_clean)
                                        if url_clean:
                                            merged_video_items.add(url_clean)
                                            extracted_from_list.append(url_clean)
                            video_section_match = re.search(r"(?:- )?Video\(s\):\s*\n(.*?)(?=\n\[https?://|$)", 
                                                           attempt_result['video_block'], re.DOTALL)
                            if video_section_match:
                                video_section = video_section_match.group(1)
                                if not extracted_from_list:
                                    video_urls = re.findall(r"https?://[^\s\)\]\>\"\']+", video_section)
                                    extracted_from_url_pattern = []
                                    for url in video_urls:
                                        url_clean = clean_url(url.strip())
                                        url_clean = re.sub(r'[.,;:!?]+$', '', url_clean)
                                        if re.search(image_extensions_pattern, url_clean, re.IGNORECASE):
                                            continue
                                        if url_clean and url_clean.startswith(('http://', 'https://')):
                                            merged_video_items.add(url_clean)
                                            extracted_from_url_pattern.append(url_clean)
                                    
                                    if extracted_from_url_pattern:
                                        print(f"  📹 Video extracted via URL pattern ({len(extracted_from_url_pattern)}): {extracted_from_url_pattern}")
                            
                            if extracted_from_list:
                                print(f"  📹 Video extracted from list format ({len(extracted_from_list)}): {extracted_from_list}")
                    first_content = all_attempts_content[0]['content']
                    page_url_match = re.search(r'\[(https?://[^\]]+)\]', first_content)
                    page_url = page_url_match.group(0) if page_url_match else f"[{sub_url}]"
                    merged_content = f"{page_url}\n"
                    merged_content += "- Text(s):\n"
                    for text_item in sorted(merged_text_items):
                        merged_content += f"  - {text_item}\n"
                    
                    merged_content += "- Image(s):\n"
                    for image_item in sorted(merged_image_items.values()):
                        merged_content += f"  - {image_item}\n"
                    
                    merged_content += "- Video(s):\n"
                    for video_item in sorted(merged_video_items):
                        merged_content += f"  - {video_item}\n"
                    last_extracted_content = {
                        'history_index': all_attempts_content[0]['history_index'],
                        'result_index': all_attempts_content[0]['result_index'],
                        'content': merged_content
                    }
                    final_result = check_sections(merged_content)
                    print(f"✅ Union merge completed:")
                    print(f"   - text: {len(merged_text_items)} (exists: {final_result['text_exists']})")
                    print(f"   - image: {len(merged_image_items)} (exists: {final_result['image_exists']})")
                    print(f"   - video: {len(merged_video_items)} (exists: {final_result['video_exists']})")
                    if merged_image_items:
                        print(f"   📷 Image URL list ({len(merged_image_items)}):")
                        for idx, image_url in enumerate(sorted(merged_image_items.values()), 1):
                            is_img = is_image(image_url)
                            status = "✅" if is_img else "❌"
                            print(f"      {idx}. {status} {image_url[:80]}...")
                    else:
                        print(f"   ⚠️ No image URLs.")
                    if merged_video_items:
                        print(f"   📹 Video URL list:")
                        for idx, video_url in enumerate(sorted(merged_video_items), 1):
                            print(f"      {idx}. {video_url}")
                    else:
                        print(f"   ⚠️ No video URLs.")
                else:
                    last_extracted_content = last_extracted_content
                success = True
                break
            else:
                analysis_prompt = get_failure_analysis_prompt(
                    current_task, long_term_memories, attempt)
                feedback = await get_feedback_from_gemini(
                    llm, analysis_prompt)

                if feedback:
                    print(f"\nGemini feedback:")
                    print(feedback)
                    parsed_feedback = parse_gemini_feedback(
                        feedback)
                    if parsed_feedback:
                        print(f"\n🔍 Parsed feedback:")
                        print(
                            f"Failure type: {parsed_feedback.get('failure_type', 'Unknown')}"
                        )
                        print(
                            f"Improvements: {parsed_feedback.get('improvements', [])}"
                        )
                        improved_prompt = parsed_feedback.get(
                            'improved_prompt', current_task)
                        if improved_prompt != current_task:
                            print(f"\nPrompt improved!")
                            current_task = improved_prompt
                        save_feedback_history(
                            attempt, current_task,
                            long_term_memories, feedback,
                            improved_prompt)
                    else:
                        print("Feedback parsing failed")
                else:
                    print("Gemini feedback request failed")

                print(f"\nUsing the improved prompt for the next attempt...")

        except Exception as e:
            print(f"❌ Error while processing URL: {e}")
            if attempt == max_attempts:
                print(f"⚠️ URL {sub_url} processing failed - maximum attempts exceeded")
                break
            else:
                print(f"🔄 Retrying... ({attempt + 1}/{max_attempts})")
        finally:
            if agent:
                try:
                    agent.stop()
                except:
                    pass
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
    print(
        f"📈 Progress: {processed_urls}/{total_urls} ({processed_urls/total_urls*100:.1f}%)"
    )
    print("-" * 50)
    
    return success, last_extracted_content


async def run_step2(wanted,
                    api_key,
                    model_name,
                    max_attempts: int = 10,
                    start_url: str = None,
                    first_url: str = None,
                    firecrawl_api_key: str = None,
                    llm_provider: str = "google",
                    ollama_host: Optional[str] = None,
                    ollama_api_key: Optional[str] = None,
                    step2_union_retry_attempts: int = 2):
    """Async generator that processes each sub-URL and yields results."""
    attachment_path = None
    processed_urls = 0
    total_urls = 0

    try:
        if not start_url:
            print("❌ start_url was not provided.")
            return

        if not os.path.exists(start_url):
            print(f"❌ File does not exist: {start_url}")
            return

        with open(start_url, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "by_depth" not in data:
            print(f"❌ JSON file does not contain the 'by_depth' key. Please check the file structure.")
            print(
                f"File content: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}"
            )
            return
        if not first_url:
            print("⚠️ first_url was not provided. Extracting the first URL from links_bfs.json.")
            for depth_key in sorted(data["by_depth"].keys()):
                urls = data["by_depth"][depth_key]
                if urls and len(urls) > 0:
                    first_url = urls[0]
                    print(f"✅ Extracted first_url: {first_url}")
                    break
            
            if not first_url:
                print("❌ Could not extract first_url from links_bfs.json.")
        for i in data["by_depth"]:
            total_urls += len(data["by_depth"][i])

        print(f"📊 Planning to process {total_urls} URLs.")
        if total_urls == 0:
            print("⚠️ No URLs to process.")
            return
        step2_model = model_name if llm_provider == "ollama" else "gemini-3-flash-preview"
        llm = create_llm(
            provider=llm_provider,
            model_name=step2_model,
            api_key=api_key,
            temperature=0.6,
            top_p=0.7,
            seed=42,
            ollama_host=ollama_host,
            ollama_api_key=ollama_api_key
        )
        

        for i in data["by_depth"]:
            if not data["by_depth"][i]:
                print(f"⚠️ Skipping depth {i} because it has no URLs.")
                continue
            if interrupted:
                print("🛑 Stopping the task due to the interrupt signal.")
                break

            print(f"🔍 Processing depth {i}...")
            for sub_url in data["by_depth"][i]:
                if interrupted:
                    print("🛑 Stopping the task due to the interrupt signal.")
                    break

                processed_urls += 1
                success, last_extracted_content = await process_single_sub_url(
                    sub_url, wanted, api_key, model_name, max_attempts,
                    first_url, firecrawl_api_key, llm_provider,
                    ollama_host, ollama_api_key, processed_urls, total_urls, llm,
                    step2_union_retry_attempts=step2_union_retry_attempts
                )
                if success and last_extracted_content:
                    page_index = processed_urls - 1  # 0-based index
                    yield sub_url, last_extracted_content, page_index
                elif not success:
                    print(f"⚠️ URL {sub_url} processing failed - continuing to the next URL")

    except Exception as e:
        print(f"❌ Overall process error: {e}")
    print(f"\n🎯 Step 2 completed!")
    print(f"📊 Processed URLs: {processed_urls}/{total_urls}")


async def run_step3(formatted_prompt, api_key, model_name, llm_provider="google", ollama_host=None, ollama_api_key=None, output_dir=None):
    """Validate and refine extracted content using the verifier prompt.

    When ``output_dir`` is provided, legal/illegal JSON files are written only
    to that directory to avoid overwriting batch outputs for the same page.
    """
    try:
        print("\n🔍 Step 3: starting content verification and refinement")
        llm = create_llm(
            provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=0.3,
            top_p=0.7,
            seed=42,
            ollama_host=ollama_host,
            ollama_api_key=ollama_api_key
        )

        print(f"Verification prompt: {formatted_prompt[:200]}...")
        from browser_use.llm.messages import UserMessage
        messages = [UserMessage(content=formatted_prompt)]

        print("Gemini is verifying and refining content...")
        response = await llm.ainvoke(messages)
        print(f"🔍 Debug: response type = {type(response)}")
        print(f"🔍 Debug: response attributes = {dir(response)[:10]}...")
        verified_content = None
        if hasattr(response, 'completion'):
            verified_content = response.completion
            print(f"🔍 Debug: response.completion type = {type(verified_content)}, length = {len(str(verified_content)) if verified_content else 0}")
        if (not verified_content or len(str(verified_content).strip()) == 0) and hasattr(response, 'content'):
            verified_content = response.content
            print(f"🔍 Debug: response.content type = {type(verified_content)}, length = {len(str(verified_content)) if verified_content else 0}")
        if (not verified_content or len(str(verified_content).strip()) == 0) and hasattr(response, 'text'):
            verified_content = response.text
            print(f"🔍 Debug: response.text type = {type(verified_content)}, length = {len(str(verified_content)) if verified_content else 0}")
        if (not verified_content or len(str(verified_content).strip()) == 0) and hasattr(response, 'messages'):
            messages = response.messages
            if messages and len(messages) > 0:
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        verified_content = msg.content
                        print(f"🔍 Debug: found content in response.messages! type = {type(verified_content)}, length = {len(str(verified_content))}")
                        break
                    elif hasattr(msg, 'text') and msg.text:
                        verified_content = msg.text
                        print(f"🔍 Debug: found text in response.messages! type = {type(verified_content)}, length = {len(str(verified_content))}")
                        break
        if (not verified_content or len(str(verified_content).strip()) == 0):
            possible_attrs = ['result', 'output', 'response_text', 'text_content', 'body', 'data']
            for attr in possible_attrs:
                if hasattr(response, attr):
                    value = getattr(response, attr)
                    if value and isinstance(value, str) and len(value.strip()) > 10:
                        verified_content = value
                        print(f"🔍 Debug: found content in response.{attr}! length = {len(value)}")
                        break
        if (not verified_content or len(str(verified_content).strip()) == 0) and hasattr(response, '__dict__'):
            response_dict = response.__dict__
            print(f"🔍 Debug: response.__dict__ keys = {list(response_dict.keys())}")
            for key in ['content', 'text', 'message', 'result', 'output', 'response', 'data', 'body']:
                if key in response_dict:
                    value = response_dict[key]
                    if value and len(str(value).strip()) > 0:
                        verified_content = value
                        print(f"🔍 Debug: found content in response.{key}! type = {type(verified_content)}, length = {len(str(verified_content))}")
                        break
            if not verified_content or len(str(verified_content).strip()) == 0:
                for key, value in response_dict.items():
                    if isinstance(value, str) and len(value.strip()) > 100 and key not in ['completion']:
                        verified_content = value
                        print(f"🔍 Debug: found a long string in response.{key}! length = {len(value)}")
                        break
        if not verified_content or len(str(verified_content).strip()) == 0:
            verified_content = str(response)
            print(f"🔍 Debug: str(response) type = {type(verified_content)}, length = {len(verified_content) if verified_content else 0}")
            print(f"🔍 Debug: str(response) content (first 500 chars): {verified_content[:500]}")
            is_object_repr = (verified_content.startswith('<') and 
                            'object at 0x' in verified_content and 
                            len(verified_content) < 100)
            if is_object_repr:
                print("⚠️ str(response) is an object representation. Trying another method.")
                verified_content = ""
            else:
                import re
                if re.search(r'```json|"illegal"|"legal"', verified_content, re.IGNORECASE):
                    print("✅ Found JSON-formatted content in str(response)!")
                else:
                    if len(verified_content) > 100:
                        print(f"⚠️ No JSON format found in str(response), but content is present (length: {len(verified_content)}).")
                        verified_content = ""
        if not isinstance(verified_content, str):
            if isinstance(verified_content, (tuple, list)):
                verified_content = ' '.join(str(item) for item in verified_content) if verified_content else str(verified_content)
            else:
                verified_content = str(verified_content)
        has_completion_tokens = False
        if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens'):
            has_completion_tokens = response.usage.completion_tokens > 0
        is_str_repr_only = False
        if verified_content and isinstance(verified_content, str):
            if (verified_content.startswith("completion=") or 
                ("completion='" in verified_content and "thinking=" in verified_content and 
                 "usage=" in verified_content and len(verified_content) < 500)):
                is_str_repr_only = True
                print(f"🔍 Debug: str(response) is an object representation (not actual content). Content: {verified_content[:200]}")
                verified_content = ""
        
        if (not verified_content or len(verified_content.strip()) == 0) and has_completion_tokens:
            print("⚠️ Warning: completion_tokens exist, but completion is empty!")
            print(f"🔍 Debug: completion_tokens = {response.usage.completion_tokens if hasattr(response, 'usage') else 'N/A'}")
            print("🔍 Debug: Inspecting all response object attributes...")
            if hasattr(response, '__dict__'):
                print("🔍 Debug: full response.__dict__ content:")
                for key, value in response.__dict__.items():
                    if key == 'usage':
                        continue
                    value_str = str(value) if value else "None"
                    value_preview = value_str[:200] if len(value_str) > 200 else value_str
                    print(f"   - {key}: {type(value).__name__} = {value_preview}...")
                for key, value in response.__dict__.items():
                    if key != 'usage' and value and isinstance(value, str) and len(value.strip()) > 100:
                        print(f"🔍 Found content in response.{key} (length: {len(value)})")
                        verified_content = value
                        break
                if not verified_content or len(verified_content.strip()) == 0:
                    import re
                    for key, value in response.__dict__.items():
                        if key != 'usage' and value and isinstance(value, str) and len(value.strip()) > 10:
                            if re.search(r'```json|"illegal"|"legal"|\{.*"illegal"|"legal".*\}', value, re.IGNORECASE | re.DOTALL):
                                print(f"🔍 Found JSON-formatted content in response.{key} (length: {len(value)})")
                                verified_content = value
                                break
                if not verified_content or len(verified_content.strip()) == 0:
                    for key, value in response.__dict__.items():
                        if key not in ['usage', 'completion'] and value and isinstance(value, str) and len(value.strip()) >= 10:
                            print(f"🔍 Found content in response.{key} (length: {len(value)})")
                            print(f"   Content preview: {value[:200]}...")
                            verified_content = value
                            break
            if not verified_content or len(verified_content.strip()) == 0:
                print("⚠️ Could not find a response. A retry may be required.")
                if has_completion_tokens:
                    print("⚠️ completion_tokens exist, but the response was not found. This may be a browser-use library bug.")
                    print("💡 Suggested fix: retry or extract the response using another method.")
                str_response = str(response)
                if len(str_response) > 100 and not (str_response.startswith('<') and 'object at 0x' in str_response):
                    print(f"🔍 Last attempt: using the full str(response) content (length: {len(str_response)})")
                    verified_content = str_response
                else:
                    verified_content = ""
                    if has_completion_tokens:
                        raise ValueError(
                            f"An LLM response was generated (completion_tokens={response.usage.completion_tokens}) "
                            f"response content could not be extracted. "
                            f"This may be a browser-use library bug or the response format may have changed. "
                            f"response object: {response.__dict__}"
                        )
        elif not verified_content or len(verified_content.strip()) == 0:
            print("⚠️ Warning: LLM response is empty!")
            print(f"🔍 Debug: verified_content = {repr(verified_content)}")
            print("⚠️ An empty response may mean the prompt is too long or the LLM failed to generate a response.")
            verified_content = ""

        print("\n✅ Verification and refinement completed!")
        print("=" * 60)
        print("📋 Verified final content:")
        print("=" * 60)
        if verified_content:
            print(verified_content[:] + ("..." if len(verified_content) > 500 else ""))
        else:
            print("(response is empty)")
        print("=" * 60)
        output_file = "verified_content_present.txt"
        try:
            import re
            if not isinstance(verified_content, str):
                if isinstance(verified_content, (tuple, list)):
                    verified_content = ' '.join(str(item) for item in verified_content) if verified_content else str(verified_content)
                else:
                    verified_content = str(verified_content)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```',
                                   verified_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"illegal".*"legal".*\}',
                                       verified_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("Could not find JSON format.")

            result_json = json.loads(json_str)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            legal_content_path = os.path.join(output_dir, "legal_content.json") if output_dir else "legal_content.json"
            illegal_content_path = os.path.join(output_dir, "illegal_content.json") if output_dir else "illegal_content.json"

            with open(legal_content_path, "w", encoding="utf-8") as f:
                json.dump(result_json.get("legal", []), f, ensure_ascii=False, indent=2)
            print(f"✅ legal_content.json created successfully: {len(result_json.get('legal', []))}")
            
            with open(illegal_content_path, "w", encoding="utf-8") as f:
                json.dump(result_json.get("illegal", []), f, ensure_ascii=False, indent=2)
            print(f"✅ illegal_content.json created successfully: {len(result_json.get('illegal', []))}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Verification timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                f.write("🚫 Illegal Content:\n")
                f.write("-" * 60 + "\n")
                illegal_items = result_json.get("illegal", [])
                if illegal_items:
                    for i, item in enumerate(illegal_items, 1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("(none)\n")

                f.write("\n" + "=" * 60 + "\n\n")
                f.write("✅ Legal Content:\n")
                f.write("-" * 60 + "\n")
                legal_items = result_json.get("legal", [])
                if legal_items:
                    for i, item in enumerate(legal_items, 1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("(none)\n")

                f.write("\n" + "=" * 60 + "\n")

            print(f"📁 Structured verification result saved to {output_file}.")

        except Exception as e:
            print(f"⚠️ JSON parsing failed, saved in raw format: {e}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Verification timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n")
                f.write("Verified final content:\n")
                f.write("=" * 60 + "\n")
                f.write(verified_content)
                f.write("\n" + "=" * 60 + "\n")
            print(f"📁 Verified content saved to {output_file}.")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            legal_content_path = os.path.join(output_dir, "legal_content.json") if output_dir else "legal_content.json"
            illegal_content_path = os.path.join(output_dir, "illegal_content.json") if output_dir else "illegal_content.json"
            with open(legal_content_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("✅ legal_content.json created successfully (empty list): 0")
            with open(illegal_content_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("✅ illegal_content.json created successfully (empty list): 0")

        return verified_content, output_file

    except Exception as e:
        print(f"❌ Step 3 error: {e}")
        return None, None


@dataclass
class PipelineConfig:
    api_keys: Optional[List[str]]
    wanted: str
    wanted_file: Optional[str]
    max_depth: int
    max_width: int
    max_pages: int
    max_attempts: int
    model_name: str
    first_url: str
    firecrawl_api_key: str
    llm_provider: str
    ollama_host: Optional[str]
    ollama_api_key: Optional[str]
    step1_only: bool
    skip_step1_prefix: str
    step1_links_path: str
    step2_union_retry_attempts: int
    step3_batch_size: int
    step3_batch_retries: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Customize script parameters.")
    parser.add_argument(
        '--api_keys',
        type=str,
        nargs='+',
        default=None,
        help=
        "API keys (required for Google provider; provide your own keys, space-separated. Not needed for Ollama).")
    parser.add_argument(
        '--wanted',
        type=str,
        default="",
        help="Wanted text (direct input; use --wanted_file for long content).")
    parser.add_argument(
        '--wanted_file',
        type=str,
        default=None,
        help="Path to file containing wanted text (overrides --wanted).")
    parser.add_argument('--max_depth',
                        type=int,
                        default=1,
                        help="Maximum depth.")
    parser.add_argument('--max_width',
                        type=int,
                        default=2,
                        help="Maximum width.")
    parser.add_argument('--max_pages',
                        type=int,
                        default=5,
                        help="Maximum pages.")
    parser.add_argument('--max_attempts',
                        type=int,
                        default=2,
                        help="Maximum attempts.")
    parser.add_argument('--model_name',
                        type=str,
                        default="gemini-2.5-flash",
                        help='Model name (e.g., "gemini-2.5-flash" for Google, "gemini-3-pro-preview:latest" for Ollama).')
    parser.add_argument(
        '--first_url',
        type=str,
        default="https://example.com/",
        help="Starting URL.")
    parser.add_argument(
        '--firecrawl_api_key',
        type=str,
        default="",
        help=
        "Firecrawl API key (optional). If provided, firecrawl will be executed for video URL extraction."
    )
    parser.add_argument(
        '--llm_provider',
        type=str,
        default='google',
        choices=['google', 'ollama'],
        help='LLM provider to use: "google" or "ollama" (default: google)'
    )
    parser.add_argument(
        '--ollama_host',
        type=str,
        default=None,
        help='Ollama server host URL (default: http://localhost:11434, can also use OLLAMA_HOST env var)'
    )
    parser.add_argument(
        '--ollama_api_key',
        type=str,
        default=None,
        help='Ollama API key for remote models (e.g., gemini-3-pro-preview). Can also use GEMINI_API_KEY or GOOGLE_API_KEY env var.'
    )
    parser.add_argument(
        '--step1_only',
        action='store_true',
        help='Run only Step 1 (link collection) for debugging purposes. Step 2 and Step 3 will be skipped.'
    )
    parser.add_argument(
        '--skip_step1_prefix',
        type=str,
        default="https://skip.example/",
        help="Prefix URL to skip Step 1 and use prebuilt links JSON."
    )
    parser.add_argument(
        '--step1_links_path',
        type=str,
        default="links_bfs.json",
        help="Path to existing links_bfs.json used when Step 1 is skipped."
    )
    parser.add_argument(
        '--step2_union_retry_attempts',
        type=int,
        default=2,
        help="Retry attempts for Step2 union-merge supplemental extraction."
    )
    parser.add_argument(
        '--step3_batch_size',
        type=int,
        default=50,
        help="Batch size for Step3 classification."
    )
    parser.add_argument(
        '--step3_batch_retries',
        type=int,
        default=2,
        help="Retry attempts per Step3 batch classification."
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    wanted = args.wanted
    if args.wanted_file:
        with open(args.wanted_file, "r", encoding="utf-8") as f:
            wanted = f.read()
    return PipelineConfig(
        api_keys=args.api_keys,
        wanted=wanted,
        wanted_file=args.wanted_file,
        max_depth=args.max_depth,
        max_width=args.max_width,
        max_pages=args.max_pages,
        max_attempts=args.max_attempts,
        model_name=args.model_name,
        first_url=args.first_url,
        firecrawl_api_key=args.firecrawl_api_key,
        llm_provider=args.llm_provider,
        ollama_host=args.ollama_host,
        ollama_api_key=args.ollama_api_key,
        step1_only=args.step1_only,
        skip_step1_prefix=args.skip_step1_prefix,
        step1_links_path=args.step1_links_path,
        step2_union_retry_attempts=args.step2_union_retry_attempts,
        step3_batch_size=args.step3_batch_size,
        step3_batch_retries=args.step3_batch_retries,
    )


class StepByStepPipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _validate_provider(self, parser: argparse.ArgumentParser):
        if self.config.llm_provider == "google" and not self.config.api_keys:
            parser.error("--api_keys is required when using the Google provider.")

    def _check_ollama(self):
        if self.config.llm_provider != "ollama":
            return
        ollama_host_final = self.config.ollama_host or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434")
        ollama_api_key_final = self.config.ollama_api_key or os.getenv(
            "GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        print("🔍 Using Ollama provider")
        print(f"📦 Model to use: {self.config.model_name}")
        print(f"🌐 Ollama server URL: {ollama_host_final}")
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{ollama_host_final}/api/tags")
                if response.status_code != 200:
                    print(f"⚠️ Ollama server response is abnormal (status code: {response.status_code})")
                    return
                print("✅ Ollama server connection confirmed")
                models_data = response.json()
                models_list = [m.get("name", "") for m in models_data.get("models", [])]
                if self.config.model_name not in models_list:
                    print(f"⚠️ Warning: model '{self.config.model_name}' was not found.")
                    print(f"   Available models: {', '.join(models_list[:5])}{'...' if len(models_list) > 5 else ''}")
                    print(f"   Download model: 'ollama pull {self.config.model_name}'")
                    print(f"   💡 If you want a fully local model: 'ollama pull llama3:8b' (API key not required)")
                    return
                print(f"✅ Model '{self.config.model_name}' confirmed")
                model_info = next((m for m in models_data.get("models", []) if m.get("name") == self.config.model_name), None)
                if not model_info:
                    return
                model_size = model_info.get("size", 0)
                remote_host = model_info.get("remote_host")
                if remote_host or (isinstance(model_size, int) and model_size < 10000):
                    print(f"⚠️ This model is remote (remote host: {remote_host or 'Ollama.com'})")
                    print(f"   Size: {model_size} bytes")
                    print(f"   ⚠️ Note: remote models may be rate-limited!")
                    if ollama_api_key_final:
                        print(f"✅ API key is configured. Attempting remote model authentication.")
                        print("   💡 If a 403 error occurs, you may have exceeded the rate limit.")
                        print(f"      Use a fully local model (e.g. llama3:8b) to avoid API keys and rate limits")
                    else:
                        print(f"❌ Remote models require an API key!")
                        print(f"   How to fix:")
                        print(f"   1. --ollama_api_key pass the API key as an argument")
                        print(f"   2. or set the GEMINI_API_KEY environment variable")
                        print(f"   3. or use a fully local model (e.g. llama3:8b) - recommended!")
                else:
                    if isinstance(model_size, int) and model_size > 0:
                        size_gb = model_size / (1024**3)
                        print(f"✅ Fully local model (Size: {size_gb:.2f} GB)")
                    else:
                        print(f"✅ Fully local model")
                    print(f"   No API key is required! 🎉")
        except Exception as e:
            print(f"⚠️ Failed to verify the Ollama server connection: {e}")
            print(f"   Please confirm the server is running at {ollama_host_final}.")

    async def run(self):
        first_url = self.config.first_url
        print(f"\n{'='*80}")
        print(f"🔗 URL: {first_url}")
        print(f"🤖 LLM Provider: {self.config.llm_provider}")
        print(f"📦 Model: {self.config.model_name}")
        print(f"{'='*80}\n")

        current_api_key_index = 0
        api_keys = self.config.api_keys if self.config.api_keys else [None]
        success = False

        while current_api_key_index < len(api_keys):
            try:
                api_key = api_keys[current_api_key_index]
                if self.config.llm_provider == "google":
                    print(f"🔑 API key {current_api_key_index + 1}/{len(api_keys)} in use...")
                else:
                    print("🔍 Using Ollama model...")

                result = None
                history = None
                attachment_path = None
                verified_content = None

                if first_url and first_url.startswith(self.config.skip_step1_prefix):
                    attachment_path = self.config.step1_links_path
                    history = None
                    print(f"⏭️ Skipping Step 1: first_url starts with '{self.config.skip_step1_prefix}'.")
                    print(f"📁 Using existing file: {attachment_path}")
                    if not os.path.exists(attachment_path):
                        print(f"❌ File does not exist: {attachment_path}")
                        raise FileNotFoundError(f"links_bfs.json file not found: {attachment_path}")
                    print(f"✅ Using the existing links_bfs.json file.")
                    result = True
                else:
                    result = await run_step1(
                        self.config.wanted,
                        first_url,
                        api_key,
                        self.config.model_name,
                        self.config.max_depth,
                        self.config.max_width,
                        self.config.max_pages,
                        max_attempts=self.config.max_attempts,
                        llm_provider=self.config.llm_provider,
                        ollama_host=self.config.ollama_host,
                        ollama_api_key=self.config.ollama_api_key)
                    if result:
                        history, attachment_path = result
                        print(f"📁 attachment_path: {attachment_path}")

                if not result:
                    success = False
                    break

                if self.config.step1_only:
                    print(f"\n{'='*80}")
                    print(f"✅ Step 1 completed (debug mode)")
                    print(f"📁 Generated file: {attachment_path}")
                    print(f"📄 links_bfs.json is available for inspection")
                    print(f"{'='*80}\n")
                    success = True
                    break

                async for sub_url, last_extracted_content, page_index in run_step2(
                        self.config.wanted,
                        api_key,
                        self.config.model_name,
                        max_attempts=self.config.max_attempts,
                        start_url=attachment_path,
                        first_url=first_url,
                        firecrawl_api_key=self.config.firecrawl_api_key,
                        llm_provider=self.config.llm_provider,
                        ollama_host=self.config.ollama_host,
                        ollama_api_key=self.config.ollama_api_key,
                        step2_union_retry_attempts=self.config.step2_union_retry_attempts):
                    print(f"\n{'='*80}")
                    print(f"🔗 Processing URL: {sub_url}")
                    print(f"📄 Page index: {page_index}")
                    print(f"{'='*80}\n")

                    if isinstance(last_extracted_content, dict):
                        content_only = _content_without_attachments(
                            last_extracted_content.get("content", ""))
                        separated_entries = split_content_by_lines([{
                            **last_extracted_content,
                            "content": content_only,
                        }])
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
                        if isinstance(entry, dict):
                            content = entry.get("line_content", str(entry))
                        else:
                            content = str(entry)
                        if content:
                            items_list.append(content)

                    items_text = "\n".join([f"- {item}" for item in items_list[:200]])
                    if len(items_list) > 200:
                        items_text += f"\n... (showing 200 of {len(items_list)} items)"
                    print(f"🔍 Debug: extracted item count = {len(items_list)}")
                    print(f"🔍 Debug: items_text length = {len(items_text)}")
                    if len(items_list) == 0:
                        print("⚠️ Warning: no extracted items were found. Please check the Step 2 result.")
                        print(f"🔍 Debug: last_extracted_content type = {type(last_extracted_content)}")
                        print(f"🔍 Debug: last_extracted_content content (first 500 chars) = {str(last_extracted_content)[:]}")

                    BATCH_SIZE = self.config.step3_batch_size
                    all_illegal_items = []
                    all_legal_items = []
                    root_folder = extract_folder_name(first_url)
                    page_output_dir = build_json_page_dir(root_folder, page_index)
                    os.makedirs(page_output_dir, exist_ok=True)
                    output_file = "verified_content_present.txt"

                    if len(items_list) > BATCH_SIZE:
                        print(f"📦 Batch processing: {len(items_list)} items into batches of {BATCH_SIZE}.")
                        num_batches = (len(items_list) + BATCH_SIZE - 1) // BATCH_SIZE
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * BATCH_SIZE
                            end_idx = min(start_idx + BATCH_SIZE, len(items_list))
                            batch_items = items_list[start_idx:end_idx]
                            batch_text = "\n".join([f"- {item}" for item in batch_items])
                            print(f"📦 Processing batch {batch_idx + 1}/{num_batches}... (items {start_idx + 1}-{end_idx})")
                            verifier_prompt = build_multimodal_classifier_prompt(batch_text)
                            max_retries = self.config.step3_batch_retries
                            batch_success = False
                            for retry_attempt in range(max_retries):
                                temp_batch_output_dir = None
                                try:
                                    temp_batch_output_dir = tempfile.mkdtemp()
                                    batch_verified_content, _ = await run_step3(
                                        verifier_prompt, api_key, self.config.model_name, self.config.llm_provider, self.config.ollama_host,
                                        self.config.ollama_api_key, output_dir=temp_batch_output_dir)
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
                                            print(
                                                f"✅ Batch {batch_idx + 1}/{num_batches} completed "
                                                f"(saved batch result to page_{page_index}): "
                                                f"Illegal {len(batch_result.get('illegal', []))}, "
                                                f"Legal {len(batch_result.get('legal', []))}"
                                            )
                                            batch_success = True
                                            break
                                    else:
                                        print(f"⚠️ Batch {batch_idx + 1}/{num_batches} response is empty.")
                                except json.JSONDecodeError as e:
                                    print(f"⚠️ Batch {batch_idx + 1}/{num_batches} JSON parsing failed: {e}")
                                except Exception as e:
                                    print(f"❌ Error while processing batch {batch_idx + 1}/{num_batches}: {e}")
                                finally:
                                    try:
                                        if temp_batch_output_dir:
                                            shutil.rmtree(temp_batch_output_dir, ignore_errors=True)
                                    except Exception:
                                        pass
                                if retry_attempt < max_retries - 1:
                                    print(f"🔄 Retrying batch {batch_idx + 1}/{num_batches}... ({retry_attempt + 2}/{max_retries})")
                                    await asyncio.sleep(2)
                            if not batch_success:
                                print(f"❌ Batch {batch_idx + 1}/{num_batches} exceeded the maximum retry count. Skipping.")

                        final_result = {"illegal": all_illegal_items, "legal": all_legal_items}
                        legal_content_path = os.path.join(page_output_dir, "legal_content.json")
                        illegal_content_path = os.path.join(page_output_dir, "illegal_content.json")
                        with open(legal_content_path, "w", encoding="utf-8") as f:
                            json.dump(final_result.get("legal", []), f, ensure_ascii=False, indent=2)
                        print(f"✅ legal_content.json created successfully (page_{page_index}): {len(final_result.get('legal', []))}")
                        with open(illegal_content_path, "w", encoding="utf-8") as f:
                            json.dump(final_result.get("illegal", []), f, ensure_ascii=False, indent=2)
                        print(f"✅ illegal_content.json created successfully (page_{page_index}): {len(final_result.get('illegal', []))}")
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
                        print(f"📁 Structured verification result saved to {output_file}.")
                        verified_content = json.dumps(final_result, ensure_ascii=False, indent=2)
                        print(f"✅ Batch processing completed: total Illegal {len(all_illegal_items)}, Legal {len(all_legal_items)}")
                    else:
                        verifier_prompt = build_multimodal_classifier_prompt(items_text)
                        try:
                            verified_content, output_file = await run_step3(
                                verifier_prompt, api_key, self.config.model_name, self.config.llm_provider, self.config.ollama_host,
                                self.config.ollama_api_key, output_dir=page_output_dir)
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

                    if verified_content or output_file:
                        if verified_content:
                            print(f"\n✅ {sub_url} Processing completed!")
                            print("Step 2: Content extraction ✅")
                            print("Step 3: Content verification and refinement ✅")
                        else:
                            print(f"\n⚠️ Step 3 JSON parsing failed for {sub_url}, but the output file was created, so download will be attempted.")
                            print("Step 2: Content extraction ✅")
                            print("Step 3: Content verification and refinement (JSON parsing failed) ⚠️")
                        print("Downloading text, image, and video")
                        process_txt_file(output_file, first_url, sub_url=sub_url, page_index=page_index)
                        print(f"📁 Data download completed for {sub_url} (page_{page_index})\n")
                    else:
                        print(f"❌ Step 3 failed for {sub_url}")

                success = True
                break
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
        return success
