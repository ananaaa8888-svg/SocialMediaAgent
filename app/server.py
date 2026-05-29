"""
Social Media Monitoring Agent — Web Server
==========================================
Custom FastAPI backend that replaces the old Streamlit dashboard.
Serves a hand-built static frontend (app/static/) and a small JSON API.

Run with:
    cd SocialMediaAgent
    uvicorn app.server:app --reload
or simply:
    python -m app.server
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import engine

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Social Media Monitoring Agent", docs_url="/api/docs")


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"          # hybrid | bm25 | semantic
    alpha: float = 0.5
    top_k: int = 200


class SummaryRequest(BaseModel):
    result_id: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/search")
def search(req: SearchRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    top_k = max(10, min(int(req.top_k), 500))
    alpha = max(0.0, min(float(req.alpha), 1.0))
    return engine.run_search(q, mode=req.mode, alpha=alpha, top_k=top_k)


@app.post("/api/summary")
def summary(req: SummaryRequest):
    try:
        return engine.run_abstractive(req.result_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Static assets (css/js). Mounted last so it doesn't shadow the API routes.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
