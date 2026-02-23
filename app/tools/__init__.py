from app.tools.registry import ToolRegistry
from app.tools.sentiment import sentiment_score

TOOLS = ToolRegistry()
TOOLS.register("sentiment_score", sentiment_score)
