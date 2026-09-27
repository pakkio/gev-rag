"""Cache for whole-document summaries (map-reduce output), keyed by doc_id.
Separate from the chunk vector store: a summary answers "what is this
document about", which per-chunk retrieval can't (see rag/summarizer.py).
"""
import json
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILE = DATA_DIR / "summaries.json"

_lock = threading.Lock()


def _load() -> dict:
    if FILE.exists():
        return json.loads(FILE.read_text(encoding="utf-8"))
    return {}


def get(doc_id: str) -> dict | None:
    with _lock:
        return _load().get(doc_id)


def set(doc_id: str, entry: dict):
    with _lock:
        data = _load()
        data[doc_id] = entry
        DATA_DIR.mkdir(exist_ok=True)
        FILE.write_text(json.dumps(data), encoding="utf-8")


def delete(doc_id: str):
    with _lock:
        data = _load()
        if data.pop(doc_id, None) is not None:
            FILE.write_text(json.dumps(data), encoding="utf-8")
