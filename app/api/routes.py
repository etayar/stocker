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

    from app.tools import TOOLS

    # v1: sentiment means "fetch recent text then score it"
    if "sentiment" in prompt_lower:
        planned.append("tool:fetch_text_data")
        planned.append("tool:sentiment_score")

        # naive query: use the prompt as the search query (we'll improve later: extract ticker/entity)
        fetch_res = TOOLS.get("fetch_text_data")(query=req.prompt, max_items=20, days=7)
        tool_outputs.append({"tool": "fetch_text_data", "result": fetch_res.model_dump()})

        agg = {"score": 0.0, "confidence": 0.0, "scored_items": 0}
        if fetch_res.ok and fetch_res.data and fetch_res.data.get("items"):
            scores = []
            confs = []

            for it in fetch_res.data["items"]:
                text = it.get("text") or ""
                sres = TOOLS.get("sentiment_score")(text=text)
                if sres.ok and sres.data:
                    scores.append(float(sres.data.get("score", 0.0)))
                    confs.append(float(sres.data.get("confidence", 0.0)))

            if scores:
                agg["score"] = sum(scores) / len(scores)
                agg["confidence"] = sum(confs) / len(confs) if confs else 0.0
                agg["scored_items"] = len(scores)

        tool_outputs.append({"tool": "sentiment_aggregate", "result": {"ok": True, "data": agg}})

    if "document" in prompt_lower or "pdf" in prompt_lower:
        planned.append("rag")

    return OrchestrateResponse(
        planned_components=planned,
        result=f"Echo (v0): {req.prompt}\nTool outputs: {tool_outputs}",
    )

