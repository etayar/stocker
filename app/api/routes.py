from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "stocker"}


class OrchestrateRequest(BaseModel):
    prompt: str


class OrchestrateResponse(BaseModel):
    planned_components: list[str]
    result: str


@router.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate(req: OrchestrateRequest):
    prompt_lower = req.prompt.lower()

    planned = ["llm"]
    tool_outputs = []

    if "sentiment" in prompt_lower:
        planned.append("tool:sentiment_score")
        from app.tools import TOOLS
        result = TOOLS.get("sentiment_score")(text=req.prompt)
        tool_outputs.append({"tool": "sentiment_score", "result": result.model_dump()})

    if "document" in prompt_lower or "pdf" in prompt_lower:
        planned.append("rag")

    # v0: still echo, but now includes real tool output
    return OrchestrateResponse(
        planned_components=planned,
        result=f"Echo (v0): {req.prompt}\nTool outputs: {tool_outputs}",
    )
