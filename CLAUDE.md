# PROTECTED FILE — DO NOT DELETE, MOVE, OR MODIFY THIS FILE UNLESS EXPLICITLY INSTRUCTED BY THE USER

# Stocker - Claude Instructions

## Project Goal
Time series (TS, stocks prices over time) + LLM sentiment (LMMS) + textual analysis (TA) to predict stock dynamics.

## Stack
- Python 3.12
- pandas, numpy for data
- transformers for LLM sentiment
- scikit-learn / pytorch for modeling
- FastAPI for backend
- PostgreSQL for storage

## Structure
/data, /models, /api, /pipeline

## Rules
- Always write tests
- No hardcoded API keys
- Modular pipeline components
- Configurable variables by yaml + env overwriting 

## Scoring & Compounding
Each component - TS, LLMS or TA will eventually provide a quantity score between 0 and 1 which is how likely should 
I buy or sell in the period defined by the user when queried the system. 
Compound a formula - f:[ts_score, llms_score, ta_score] -> [0, 1] s.t:
    f(ts_score, llms_score, ta_score) = (ts_score * llms_score * ta_score)^(1/3)
    In general, for n quantities q1, q2, ..., qn: f(q1, q2, ..., qn) = (q1 * q2 * ... * qn)^(1/n) -> pretty robust, 
    since one quantity near zero will drag down the entire score, so we won't get a buy action if one indicator signals 
    danger. Moreover, if q1 = q2 = ... = qn := q => f(q1, q2, ..., qn) = q.

## Data Sources
- Stock prices: Yahoo Finance (yfinance) / Alpha Vantage / Polygon.io
- News/text: NewsAPI / Finnhub / Reddit (PRAW)
- Sentiment model: FinBERT (preferred over generic BERT for finance)

## TS Model - Full Architecture

### Model Hierarchy
1. **Base**: Pretrained Chronos (amazon/chronos-t5-small, HuggingFace, frozen)
2. **General model**: Chronos fine-tuned on broad stock universe (many tickers)
   - Rebuilt every 14 days via background scheduler
   - Saved to: models/general/model + metadata.json
3. **Ticker model**: General model fine-tuned on specific ticker
   - Built on-demand when user requests a ticker
   - Cache TTL: 7 days (config.yaml → model.ts.cache_ttl_days)
   - If ticker model exists AND age < 7 days → skip fine-tuning, load & predict
   - Saved to: models/cache/{TICKER}/model + metadata.json

### Walk-Forward Validation (NO random splits, NO static train/val)
- Data is split into small temporally-ordered folds
- Each fold: |—train window—|—val window—|
- Folds slide forward through time (no overlap between val windows)
- Train window size: config.yaml → data.walk_forward.train_window_days
- Val window size:   config.yaml → data.walk_forward.val_window_days  
- Step size:         config.yaml → data.walk_forward.step_days
- NEVER shuffle, NEVER use future data, NEVER re-use val data as train

### Leakage Prevention
- Scaler fitted on each train fold only
- Val fold transformed with train-fitted scaler
- Scaler saved to models/cache/{TICKER}/scaler + models/general/scaler

### Automation - Everything is automatic, zero manual steps
On user request for ticker X:
  1. Check models/cache/{X}/metadata.json
     - EXISTS and age < 7 days → load model → predict → ts_score. DONE.
     - MISSING or stale →
  2. Fetch full price history via yfinance (auto, up to today)
  3. Build walk-forward folds automatically
  4. Load general model from models/general/
  5. Fine-tune general model on ticker X using walk-forward folds
  6. Save model + scaler + metadata.json to models/cache/{X}/
  7. Predict → ts_score

Every 14 days (background scheduler):
  1. Fetch price history for all tickers in config.yaml → data.universe
  2. Build walk-forward folds for each ticker
  3. Load pretrained Chronos base
  4. Fine-tune on full universe
  5. Save to models/general/ + update metadata.json

### Model Registry
- models/
  ├── general/
  │   ├── model/          # general fine-tuned Chronos weights
  │   ├── scaler/         # fitted scaler
  │   └── metadata.json   # { "last_trained": "...", "universe": [...], "n_tickers": N }
  └── cache/
      └── {TICKER}/
          ├── model/      # ticker-specific weights
          ├── scaler/     # ticker-specific scaler
          └── metadata.json  # { "ticker": "...", "fine_tuned_at": "...", "data_until": "..." }


## Agentic Layer

A natural language agent sits in front of the pipeline and translates 
human prompts into structured query parameters.

**Example:**
- Input: "Should I buy Tesla for the next 2 weeks?"
- Output: `{ "ticker": "TSLA", "horizon": 14, "granularity": "daily" }`

### Agent Responsibilities
- Extract ticker from company name or symbol
- Infer prediction horizon from natural language ("next week" → 7)
- Validate and normalize parameters
- Pass structured params to the pipeline runner
- Return human-readable explanation of the signal score

### Stack Addition
- LLM: Claude API (via anthropic SDK) for parameter extraction
- New module: `api/agent.py`
- New route: `POST /ask` (accepts free-text, returns signal + explanation)

### Flow
User prompt → agent.py → structured params → pipeline/runner.py → scores → 
scorer.py → signal → agent formats human-readable response → user

## MVP Scope (be explicit!)
- Single stock at a time
- Daily granularity
- Prediction horizon: next N days (user configurable)
- No live trading — signal only

## What NOT to do
- No monolithic scripts
- No Jupyter notebooks in production code
- No hardcoded tickers or dates



