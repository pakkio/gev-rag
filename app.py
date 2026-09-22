"""Web server: a JSON API over the RAG pipeline + the single-page UI.

Run:  .venv\\Scripts\\python app.py   then open http://localhost:8000
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    # Tiny .env reader so you can put ANTHROPIC_API_KEY=... in rag-app/.env
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_dotenv()

from typing import Literal  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from rag import embedder, generator, jev, local_llm, pipeline  # noqa: E402

BENCH_RESULTS = ROOT / "data" / "bench_results.json"

app = FastAPI(title="Tiny RAG")


class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    engines: list[Literal["claude", "local", "jev"]] = ["claude"]


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        "llm_enabled": generator.llm_enabled(),
        "llm_model": generator.MODEL,
        "jev_enabled": jev.jev_enabled(),
        "local_model": local_llm.MODEL,
        "local": local_llm.local_status(),
        "embed_model": embedder.MODEL_NAME,
        "documents": pipeline.store.documents(),
        "total_chunks": len(pipeline.store.chunks),
    }


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
    return {"removed_chunks": removed}


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    engines = list(dict.fromkeys(req.engines))  # dedupe, keep order
    if not engines:
        raise HTTPException(400, "Pick at least one engine")
    return pipeline.ask(req.question.strip(), max(1, min(req.top_k, 10)), engines)


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
