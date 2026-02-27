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



