"""
models package

Three independent scoring models, each producing a score in [0, 1]:
  - ts_model        : time-series model → ts_score
  - sentiment_model : FinBERT LLM sentiment → llms_score
  - ta_model        : technical analysis indicators → ta_score

Plus the compound scorer:
  - scorer          : geometric mean f(ts, llms, ta) = (ts·llms·ta)^(1/3)
"""
