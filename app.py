"""Web server: a JSON API over the RAG pipeline + the single-page UI.

Run:  .venv\\Scripts\\python app.py   then open http://localhost:8000
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    # Tiny .env reader. Project .env (./.env) wins; parent ../.env fills anything
    # missing, so shared keys (OPENROUTER_API_KEY, TYPESAFE_API_KEY, ...) can live
    # outside this repo — e.g. when you run several apps off one API account.
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_dotenv()

from typing import Literal  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from rag import embedder, gemini_llm, generator, jev, local_llm, pipeline, summary_store  # noqa: E402

BENCH_RESULTS = ROOT / "data" / "bench_results.json"

app = FastAPI(title="Tiny RAG")


class AskRequest(BaseModel):
    question: str
    top_k: int = 12
    engines: list[Literal["claude", "local", "jev", "gemini"]] = ["jev", "local"]
    doc: str | None = None  # search only this book: a doc_id, or a regex on its file name
    # On-the-fly Jev override for THIS query only: None = env/UI default,
    # True = force Jev on (even if the UI pills dropped it), False = force off.
    jev: bool | None = None
    # Earlier turns of this chat, [{"q": question, "a": answer}], oldest first: follow-ups
    # ("e come muore?") are rewritten with them. Empty after "New chat".
    history: list[dict] = []


class AskDocRequest(BaseModel):
    question: str
    model: str | None = None  # one OpenRouter model id instead of pipeline.ASK_DOC_MODELS


class AskAllRequest(BaseModel):
    question: str
    engine: Literal["cloud", "local", "jev"] | None = None  # defaults to pipeline.ASK_ALL_ENGINE
    model: str | None = None  # for engine=cloud, one OpenRouter model id instead of ASK_ALL_MODELS


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        "llm_enabled": generator.llm_enabled(),
        "llm_model": generator.MODEL,
        "jev_enabled": jev.jev_enabled(),
        "jev_reason": jev.jev_reason(),
        "gemini_enabled": gemini_llm.gemini_enabled(),
        "gemini_model": gemini_llm.MODEL,
        "local_model": local_llm.MODEL,
        "local": local_llm.local_status(),
        "embed_model": embedder.MODEL_NAME,
        "documents": pipeline.store.documents(),
        "total_chunks": len(pipeline.store.chunks),
    }


@app.get("/api/documents")
def list_documents(q: str | None = None):
    return {"documents": pipeline.store.documents(q)}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...), chunk_size: int = Form(800), overlap: int = Form(150)):
    if not 100 <= chunk_size <= 4000:
        raise HTTPException(400, "chunk_size must be between 100 and 4000")
    results = []
    for f in files:
        try:
            results.append(pipeline.ingest(f.filename, f.file.read(), chunk_size, overlap))
        except ValueError as e:
            results.append({"name": f.filename, "error": str(e)})
    return {"results": results}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = pipeline.store.delete_doc(doc_id)
    if not removed:
        raise HTTPException(404, "Document not found")
    summary_store.delete(doc_id)
    return {"removed_chunks": removed}


@app.get("/api/documents/{doc_id}/summary")
def get_summary(doc_id: str):
    entry = summary_store.get(doc_id)
    if not entry:
        raise HTTPException(404, "No summary yet — POST to this URL to build one")
    return entry


@app.post("/api/documents/{doc_id}/summary")
def build_summary(doc_id: str):
    result = pipeline.summarize_doc(doc_id)
    if "error" in result:
        raise HTTPException(404 if result["error"] == "Document not found" else 502, result["error"])
    return result


@app.post("/api/documents/{doc_id}/ask")
def ask_doc(doc_id: str, req: AskDocRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    result = pipeline.ask_doc(doc_id, req.question.strip(), req.model)
    if "error" in result:
        raise HTTPException(404 if result["error"] == "Document not found" else 502, result["error"])
    return result


@app.post("/api/ask-all")
def ask_all(req: AskAllRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    result = pipeline.ask_all(req.question.strip(), req.engine, req.model)
    if "error" in result:
        raise HTTPException(502, result["error"])
    return result


def _ask_args(req: AskRequest) -> tuple[str, int, list, str | None, list]:
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    engines = list(dict.fromkeys(req.engines))  # dedupe, keep order
    if req.jev is False:
        engines = [e for e in engines if e != "jev"]
    elif req.jev is True and "jev" not in engines:
        engines.append("jev")
    if not engines:
        raise HTTPException(400, "Pick at least one engine")
    doc_id = None
    if req.doc:
        docs = pipeline.store.documents()
        try:
            match = [d for d in docs if d["doc_id"] == req.doc] or pipeline.store.documents(req.doc)
        except re.error as e:
            raise HTTPException(400, f"`doc` is not a valid regex: {e}")
        if len(match) != 1:
            raise HTTPException(400, f"`doc` must match exactly one document, matched {len(match)}")
        doc_id = match[0]["doc_id"]
    return req.question.strip(), max(1, min(req.top_k, 16)), engines, doc_id, req.history


@app.post("/api/ask")
def ask(req: AskRequest):
    return pipeline.ask(*_ask_args(req))


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Same as /api/ask, streamed as NDJSON events (see pipeline.ask_stream)."""
    events = pipeline.ask_stream(*_ask_args(req))
    return StreamingResponse((json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                             media_type="application/x-ndjson")


@app.get("/bench")
def bench_page():
    return FileResponse(ROOT / "static" / "bench.html")


@app.get("/api/bench")
def bench_results():
    if not BENCH_RESULTS.exists():
        raise HTTPException(404, "No benchmark results yet. Run bench.py first.")
    return FileResponse(BENCH_RESULTS, media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
