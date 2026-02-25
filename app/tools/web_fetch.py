from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote_plus

import feedparser
import httpx
from cachetools import TTLCache

from app.tools.registry import ToolResult
from app.config import get_settings

import re
_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(s: str) -> str:
    return _TAG_RE.sub("", s).replace("\xa0", " ").strip()


# Cache results to avoid refetching the same query repeatedly during dev/testing.
# Key: (query, max_items, days)
_CACHE: TTLCache = TTLCache(maxsize=256, ttl=600)  # 10 minutes

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fetch_text_data(query: str, *, max_items: int = 20, days: int = 7) -> ToolResult:
    """
    Fetch recent text snippets from RSS (v0):
    - Uses Google News RSS query feed (broad coverage).
    - Deduplicates by URL/title.
    - Returns a list of items with title/url/published/text.
    """
    if not query or not query.strip():
        return ToolResult(ok=False, error="Empty query")

    settings = get_settings()
    cfg = settings.fetch

    days = int(cfg.get("days", days))
    max_items = int(cfg.get("max_items", max_items))
    hl = str(cfg.get("hl", "en-US"))
    gl = str(cfg.get("gl", "US"))
    ceid = str(cfg.get("ceid", "US:en"))


    max_items = max(1, min(int(max_items), 50))
    days = max(1, min(int(days), 30))

    cache_key = (query.strip().lower(), max_items, days)
    if cache_key in _CACHE:
        return ToolResult(ok=True, data={"source": "cache", "items": _CACHE[cache_key]})

    # Google News RSS search feed. (This is a public RSS endpoint.)
    # We keep the URL inside code (good practice for reproducibility).
    q = quote_plus(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={q}+when:{days}d&hl={hl}&gl={gl}&ceid={ceid}"

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(rss_url)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
    except Exception as e:
        return ToolResult(ok=False, error=f"Fetch failed: {type(e).__name__}: {e}")

    cutoff = _now_utc() - timedelta(days=days)

    seen = set()
    items: List[Dict[str, Any]] = []

    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or "").strip()

        if not title and not summary:
            continue

        # Parse published time if present; otherwise keep None.
        published_dt = None
        if entry.get("published_parsed"):
            try:
                published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                published_dt = None

        # Filter by date when available
        if published_dt is not None and published_dt < cutoff:
            continue

        # Dedup key
        key = link or title
        if not key or key in seen:
            continue
        seen.add(key)

        text_raw = " ".join([t for t in [title, summary] if t]).strip()
        text = _strip_html(text_raw)


        items.append(
            {
                "title": title,
                "url": link,
                "published_utc": published_dt.isoformat() if published_dt else None,
                "text_raw": text_raw,
                "text": text,
            }
        )

        if len(items) >= max_items:
            break

    _CACHE[cache_key] = items
    return ToolResult(ok=True, data={"source": "rss", "rss_url": rss_url, "items": items})
