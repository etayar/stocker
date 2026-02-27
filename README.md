# Stocker

> Time-series + LLM sentiment + technical analysis → stock buy/sell signal (0–1 score).

## Overview

Stocker combines three independent scoring components and compounds them via geometric mean into a single actionable signal:

```
signal = (ts_score × llms_score × ta_score)^(1/3)
```

A score near 1 = strong buy, near 0 = strong sell. One weak indicator drags the entire score down.

## Components

| Component | Source | Output |
|-----------|--------|--------|
| Time-Series (TS) | Price history | `ts_score ∈ [0,1]` |
| LLM Sentiment (LLMS) | News / Reddit via FinBERT | `llms_score ∈ [0,1]` |
| Technical Analysis (TA) | RSI, MACD, Bollinger Bands | `ta_score ∈ [0,1]` |

## Stack

- **Python 3.12**
- **Data**: pandas, numpy, yfinance, NewsAPI, PRAW
- **Models**: FinBERT (transformers), PyTorch, scikit-learn
- **API**: FastAPI + Uvicorn
- **Storage**: PostgreSQL (SQLAlchemy)

## Project Structure

```
stocker/
├── data/
│   ├── fetchers/          # Price & news data fetchers
│   └── storage/           # PostgreSQL persistence layer
├── models/
│   ├── ts_model.py        # Time-series model → ts_score
│   ├── sentiment_model.py # FinBERT sentiment → llms_score
│   ├── ta_model.py        # Technical indicators → ta_score
│   └── scorer.py          # Geometric mean compound scorer
├── pipeline/
│   ├── runner.py          # Orchestrates full pipeline
│   └── preprocessors/     # Price & text preprocessing
├── api/
│   ├── main.py            # FastAPI entrypoint
│   ├── routes/            # API route handlers
│   └── schemas.py         # Pydantic request/response models
├── tests/                 # Mirrors source structure
├── config.yaml            # All configurable parameters
└── .env.example           # Required environment variables
```

## Setup

```bash
# 1. Clone and enter directory
git clone <repo> && cd stocker

# 2. Create virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start API server
uvicorn api.main:app --reload
```

## Usage

```bash
# Query signal for a stock over the next N days
curl "http://localhost:8000/signal?ticker=AAPL&horizon=5"
```

## Configuration

All parameters are in `config.yaml`. Environment variables in `.env` override config values.

## MVP Scope

- Single stock at a time
- Daily granularity
- Configurable prediction horizon (N days)
- Signal only — no live trading

## Testing

```bash
pytest tests/
```
