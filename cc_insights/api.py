"""FastAPI surface."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .app import build_orchestrator, build_session_store
from .schemas import AskRequest, AskResponse


app = FastAPI(title="cc-insights")
_orchestrator = None
_session = None


@app.on_event("startup")
def _startup() -> None:
    global _orchestrator, _session
    _orchestrator = build_orchestrator()
    _session = build_session_store()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if _orchestrator is None or _session is None:
        raise HTTPException(503, "Not initialized")
    sid = _session.ensure_session(req.session_id)
    try:
        answer, query_id, latency_ms = _orchestrator.ask(req.question)
    except Exception as e:
        raise HTTPException(500, f"agent error: {e}") from e
    _session.record(sid, req.question, answer.model_dump())
    return AskResponse(answer=answer, latency_ms=latency_ms, query_id=query_id)


@app.post("/feedback")
def feedback(query_id: str, value: str) -> dict:
    if _session is None:
        raise HTTPException(503, "Not initialized")
    if value not in {"up", "down"}:
        raise HTTPException(400, "value must be 'up' or 'down'")
    _session.feedback(query_id, value)
    return {"ok": True}
