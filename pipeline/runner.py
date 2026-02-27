"""
pipeline.runner

Top-level pipeline orchestrator. Wires together all fetch, preprocess,
model, and scoring steps into a single callable.

Reads configuration from config.yaml (merged with env overrides).

Public interface:
  run_pipeline(ticker: str, horizon: int, config: dict) -> dict
    Executes the full pipeline for one stock.
    Returns a result dict:
    {
      "ticker": str,
      "run_date": str (ISO-8601),
      "horizon": int,
      "ts_score": float,
      "llms_score": float,
      "ta_score": float,
      "compound_score": float,
      "signal": "BUY" | "SELL" | "HOLD",
    }

  load_config(config_path: str = "config.yaml") -> dict
    Loads YAML config and applies env var overrides.
"""
