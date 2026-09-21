"""STEP 4 (ingest) / STEP 2 (query): VECTOR STORE — save vectors, find nearest ones.

A deliberately tiny "vector database": one numpy matrix + a JSON list of chunk
metadata, persisted to ./data. Search = dot product against every row
(brute force). Fine for thousands of chunks; swap in Chroma/FAISS/pgvector
when you outgrow it.
"""
import json
import threading
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.lock = threading.Lock()
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks = []  # [{doc_id, doc_name, chunk_index, text}]
        self._load()

    def _load(self):
        vec_file, meta_file = DATA_DIR / "vectors.npy", DATA_DIR / "chunks.json"
        if vec_file.exists() and meta_file.exists():
            self.vectors = np.load(vec_file)
            self.chunks = json.loads(meta_file.read_text(encoding="utf-8"))

    def _save(self):
        DATA_DIR.mkdir(exist_ok=True)
        np.save(DATA_DIR / "vectors.npy", self.vectors)
        (DATA_DIR / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    def add(self, vectors: np.ndarray, chunks: list):
        with self.lock:
            self.vectors = np.vstack([self.vectors, vectors])
            self.chunks.extend(chunks)
            self._save()

    def delete_doc(self, doc_id: str) -> int:
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            removed = len(self.chunks) - len(keep)
            self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self.chunks = [self.chunks[i] for i in keep]
            self._save()
            return removed

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> list:
        with self.lock:
            if not self.chunks:
                return []
            scores = self.vectors @ query_vec  # cosine similarity for each chunk
            top = np.argsort(-scores)[:top_k]
            return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def documents(self) -> list:
        docs = {}
        for c in self.chunks:
            d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "name": c["doc_name"], "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())
