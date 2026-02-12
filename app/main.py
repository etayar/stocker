from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Stocker")

@app.get("/health")
def health():
    return {"status": "ok", "service": "stocker"}


class OrchestrateRequest(BaseModel):
    prompt: str


class OrchestrateResponse(BaseModel):
    planned_components: list[str]
    result: str


@app.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate(req: OrchestrateRequest):
    # v0: "planning" is just simple rules, no LLM yet
    prompt_lower = req.prompt.lower()

    planned = ["llm"]  # in the future: llm, rag, tools, models, etc.

    if "sentiment" in prompt_lower:
        planned.append("tool:sentiment")

    if "document" in prompt_lower or "pdf" in prompt_lower:
        planned.append("rag")

    return OrchestrateResponse(
        planned_components=planned,
        result=f"Echo (v0): {req.prompt}",
    )
