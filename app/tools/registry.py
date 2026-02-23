from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: Optional[str] = None

ToolFn = Callable[..., ToolResult]

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = fn

    def get(self, name: str) -> ToolFn:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[str]:
        return sorted(self._tools.keys())
