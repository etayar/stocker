from __future__ import annotations

import re
from typing import Dict, List, Tuple

from app.tools.registry import ToolResult

# Very small alias map (you can grow it later)
# Key: normalized company name keyword -> (ticker, canonical company name)
_NAME_TO_TICKER: Dict[str, Tuple[str, str]] = {
    "nvidia": ("NVDA", "NVIDIA"),
    "microsoft": ("MSFT", "Microsoft"),
    "apple": ("AAPL", "Apple"),
    "amazon": ("AMZN", "Amazon"),
    "alphabet": ("GOOGL", "Alphabet"),
    "google": ("GOOGL", "Alphabet"),
    "meta": ("META", "Meta"),
    "tesla": ("TSLA", "Tesla"),
}

# Common all-caps words that are NOT tickers
_STOP_TICKERS = {
    "A", "I", "AN", "THE", "AND", "OR", "OF", "TO", "IN", "ON", "FOR", "WITH", "THIS", "THAT", "WEEK"
}

# Regex for tickers like NVDA, MSFT, GOOGL, TSLA (1-5 letters), also allow $NVDA
_TICKER_RE = re.compile(r"(?<![A-Z0-9])\$?([A-Z]{1,5})(?![A-Z0-9])")

def extract_ticker_query(text: str) -> ToolResult:
    """
    Extract probable stock tickers / company mentions and produce a clean search query.
    v0 rules:
      - Tickers: ALLCAPS tokens of length 1..5 (optionally prefixed with $)
      - Company names: simple keyword matching (nvidia, microsoft, etc.)
      - Query: "TICKER OR CompanyName stock"
    """
    if not text or not text.strip():
        return ToolResult(ok=False, error="Empty text")

    raw = text.strip()

    # 1) Find tickers
    tickers: List[str] = []
    for m in _TICKER_RE.finditer(raw):
        t = m.group(1)
        if t in _STOP_TICKERS:
            continue
        # avoid extremely short junk
        if len(t) == 1:
            continue
        if t not in tickers:
            tickers.append(t)

    # 2) Find company name aliases (case-insensitive)
    lowered = raw.lower()
    company_hits: List[str] = []
    for name_key, (ticker, canonical) in _NAME_TO_TICKER.items():
        if name_key in lowered:
            if canonical not in company_hits:
                company_hits.append(canonical)
            if ticker not in tickers:
                tickers.append(ticker)

    # 3) Build query
    parts: List[str] = []
    for t in tickers:
        parts.append(t)
    for c in company_hits:
        parts.append(f"\"{c}\"")

    if parts:
        base = " OR ".join(parts)
        query = f"({base}) stock OR earnings OR shares"
        confidence = 0.8 if tickers else 0.6
    else:
        # fallback: use original text but trimmed (still better than nothing)
        query = raw
        confidence = 0.2

    return ToolResult(
        ok=True,
        data={
            "tickers": tickers,
            "company_names": company_hits,
            "query": query,
            "confidence": confidence,
        },
    )
