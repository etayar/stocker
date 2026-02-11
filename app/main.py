from fastapi import FastAPI

app = FastAPI(title="Stocker")

@app.get("/health")
def health():
    return {"status": "ok", "service": "stocker"}
