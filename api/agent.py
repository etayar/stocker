"""
api.agent

Agent functions that bridge free-text user prompts and the Stocker pipeline.

Two public functions:
  parse_query(prompt, client, config) -> QueryParams
    Uses Anthropic tool use (forced tool call) to extract ticker, horizon,
    and granularity from a natural-language prompt.

  explain_signal(result, prompt, client, config) -> str
    Generates a human-readable explanation of the pipeline result using a
    short Claude completion.

Neither function imports anthropic directly; they accept a duck-typed client
so tests can inject a plain MagicMock without installing the SDK.
"""

from __future__ import annotations

import json

from api.schemas import QueryParams

# ── Tool schema passed to Anthropic for structured extraction ──────────────────

_EXTRACT_TOOL = {
    "name": "extract_query_params",
    "description": (
        "Extract structured stock query parameters from a natural-language prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Uppercase stock ticker symbol, e.g. AAPL, TSLA, NVDA. "
                    "Convert company names to their ticker (e.g. Apple → AAPL)."
                ),
            },
            "horizon": {
                "type": "integer",
                "description": (
                    "Prediction horizon in calendar days. "
                    "Interpret 'next week' as 7, 'next month' as 30, etc."
                ),
                "minimum": 1,
                "maximum": 365,
            },
            "granularity": {
                "type": "string",
                "enum": ["daily"],
                "description": "Always 'daily' for now.",
            },
        },
        "required": ["ticker", "horizon", "granularity"],
    },
}

_PARSE_SYSTEM = (
    "You are a financial query parser. "
    "Given a user prompt, call extract_query_params with the correct values. "
    "Always convert company names to their official ticker symbol. "
    "If the horizon is ambiguous, default to 7 days."
)

_EXPLAIN_SYSTEM = (
    "You are a concise financial analyst assistant. "
    "Explain a stock signal score in 2-3 plain-English sentences. "
    "Mention the individual component scores and what the compound signal means. "
    "Do not give investment advice; state that signals are for informational purposes only."
)


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_query(prompt: str, client: object, config: dict) -> QueryParams:
    """Extract structured query parameters from *prompt* using the Anthropic API.

    Parameters
    ----------
    prompt  : Natural-language user question (e.g. "Should I buy Tesla next week?").
    client  : Anthropic client instance (duck-typed for testability).
    config  : Loaded config dict; uses ``config["agent"]`` for model and token limits.

    Returns
    -------
    QueryParams with ticker, horizon, granularity.

    Raises
    ------
    ValueError  : If the model does not return a tool_use block.
    """
    agent_cfg = config.get("agent", {})
    model = agent_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = int(agent_cfg.get("max_tokens_parse", 256))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_PARSE_SYSTEM,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_query_params"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the tool_use block from the response.
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise ValueError(
            "Agent did not return structured parameters. "
            f"Raw response: {response.content}"
        )

    raw = tool_block.input
    return QueryParams(
        ticker=raw["ticker"].upper(),
        horizon=int(raw["horizon"]),
        granularity=raw.get("granularity", "daily"),
    )


def explain_signal(
    result: dict,
    prompt: str,
    client: object,
    config: dict,
) -> str:
    """Generate a human-readable explanation of *result* for the original *prompt*.

    Parameters
    ----------
    result  : Dict returned by ``run_pipeline`` (includes scores and signal).
    prompt  : Original user question (for context).
    client  : Anthropic client instance (duck-typed).
    config  : Loaded config dict.

    Returns
    -------
    Plain-English explanation string.
    """
    agent_cfg = config.get("agent", {})
    model = agent_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = int(agent_cfg.get("max_tokens_explain", 512))

    user_message = (
        f"User question: {prompt}\n\n"
        f"Pipeline result:\n"
        f"  Ticker:         {result['ticker']}\n"
        f"  Horizon:        {result['horizon']} days\n"
        f"  TS score:       {result['ts_score']:.3f}\n"
        f"  Sentiment score:{result['llms_score']:.3f}\n"
        f"  TA score:       {result['ta_score']:.3f}\n"
        f"  Compound score: {result['compound_score']:.3f}\n"
        f"  Signal:         {result['signal']}\n"
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
