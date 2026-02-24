from app.tools.registry import ToolRegistry
from app.tools.sentiment import sentiment_score
from app.tools.web_fetch import fetch_text_data
from app.tools.ticker_extract import extract_ticker_query

TOOLS = ToolRegistry()
TOOLS.register("sentiment_score", sentiment_score)
TOOLS.register("fetch_text_data", fetch_text_data)
TOOLS.register("extract_ticker_query", extract_ticker_query)
