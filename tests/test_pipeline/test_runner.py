"""
tests.test_pipeline.test_runner

Tests for pipeline.runner.

Test cases to implement:
  - test_run_pipeline_returns_expected_keys
      run_pipeline() result dict has keys:
      [ticker, run_date, horizon, ts_score, llms_score,
       ta_score, compound_score, signal].
  - test_run_pipeline_scores_in_range
      All score fields ∈ [0, 1].
  - test_run_pipeline_signal_valid
      signal field is one of {"BUY", "SELL", "HOLD"}.
  - test_load_config_reads_yaml
      load_config() returns a dict with expected top-level keys.
  - test_load_config_env_override
      Setting STOCK_TICKER env var overrides config.yaml value.
  - test_run_pipeline_with_mocked_components
      Mock all three models to return fixed scores; verify compound
      score matches expected geometric mean.
"""
