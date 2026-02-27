"""
api.schemas

Pydantic request and response models for the Stocker API.

Models:
  SignalRequest
    Fields: ticker (str), horizon (int, default=5)

  ComponentScores
    Fields: ts_score (float), llms_score (float), ta_score (float)

  SignalResponse
    Fields:
      ticker         : str
      run_date       : datetime
      horizon        : int
      ts_score       : float  ∈ [0, 1]
      llms_score     : float  ∈ [0, 1]
      ta_score       : float  ∈ [0, 1]
      compound_score : float  ∈ [0, 1]
      signal         : Literal["BUY", "SELL", "HOLD"]

  HistoricalScoreRecord
    Fields: same as SignalResponse plus id (int)

  ErrorResponse
    Fields: detail (str)
"""
