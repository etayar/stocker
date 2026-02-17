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
    # v0: simple rule-based "planner" (we'll replace later with real orchestrator)
    prompt_lower = req.prompt.lower()

    planned = ["llm"]
    if "sentiment" in prompt_lower:
        planned.append("tool:sentiment")
    if "document" in prompt_lower or "pdf" in prompt_lower:
        planned.append("rag")

    return OrchestrateResponse(
        planned_components=planned,
        result=f"Echo (v0): {req.prompt}",
    )
