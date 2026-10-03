"""STEP 4 (ingest) / STEP 2 (query): VECTOR STORE — save vectors, find nearest ones.

A deliberately tiny "vector database": one numpy matrix + a JSON list of chunk
metadata, persisted to ./data. Search = dot product against every row
(brute force). Fine for thousands of chunks; swap in Chroma/FAISS/pgvector
when you outgrow it.
"""
import json
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
def word_regex(term: str) -> re.Pattern:
    """Case-insensitive whole-word match: "Anna" must not hit "Annabelle"."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


# A capitalized word after a lowercase letter, digit or comma and one space: mid-sentence, so
# capitalized because it's a name, not because it starts a sentence.
_MID_CAPITAL = re.compile(r"(?<=[a-zà-ÿ0-9,;]) ([A-ZÀ-Ý][a-zà-ÿ'’]{2,})")
_LOWER_WORD = re.compile(r"\b([a-zà-ÿ][a-zà-ÿ'’]{2,})\b")


BOOST = 0.1  # per question name found in a chunk; cosine scores of top hits differ by ~0.01-0.05


class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.lock = threading.Lock()
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks = []  # [{doc_id, doc_name, chunk_index, text}]
        self._names = (None, {})  # (chunk count it was built for, lowercase -> capitalized form)
        self._positions = (None, {})  # (chunk count, (doc_id, chunk_index) -> position)
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
            # An empty store takes the model's dimension instead of the 384 default.
            self.vectors = vectors if not len(self.chunks) else np.vstack([self.vectors, vectors])
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

    def search(self, query_vec: np.ndarray, top_k: int = 4, doc_id: str | None = None,
               boost_terms: list[str] = ()) -> list:
        """Top-k chunks by cosine similarity, optionally within one document. Each boost term
        (a name from the question) found in a chunk adds BOOST to its score: an embedding
        model barely separates a minor character's passages ("Justin") from the
        rest of an Italian novel, while the name itself pins them down."""
        with self.lock:
            if not self.chunks:
                return []
            idx = (np.arange(len(self.chunks)) if doc_id is None else
                   np.array([i for i, c in enumerate(self.chunks) if c["doc_id"] == doc_id], dtype=int))
            if len(idx) == 0:
                return []
            scores = self.vectors[idx] @ query_vec  # cosine similarity for each chunk
            for term in boost_terms:
                rx = word_regex(term)
                present = np.array([bool(rx.search(self.chunks[i]["text"])) for i in idx])
                # A name in >10% of the searched chunks ("Bovary" inside Madame Bovary) says
                # nothing about which passage answers; boosting it would bury the rare one.
                if 0 < present.sum() <= 0.1 * len(idx):
                    scores = scores + BOOST * present
            top = np.argsort(-scores)[:top_k]
            return [{**self.chunks[idx[j]], "score": float(scores[j])} for j in top]

    def mentions(self, query_vec: np.ndarray, term: str, doc_ids: set, limit: int) -> tuple[int, list]:
        """Every chunk of doc_ids naming `term` as a whole word: (how many there are, the
        `limit` most similar to the query, in reading order). For questions about a person
        or thing, whose description is spread over many passages rather than in the top-k."""
        rx = word_regex(term)
        with self.lock:
            idx = np.array([i for i, c in enumerate(self.chunks)
                            if c["doc_id"] in doc_ids and rx.search(c["text"])], dtype=int)
            if len(idx) == 0:
                return 0, []
            scores = self.vectors[idx] @ query_vec
            keep = sorted(idx[np.argsort(-scores)[:limit]].tolist(),
                          key=lambda i: (self.chunks[i]["doc_id"], self.chunks[i]["chunk_index"]))
            return len(idx), [{**self.chunks[i], "score": float(self.vectors[i] @ query_vec)} for i in keep]

    def with_neighbors(self, chunk: dict, span: int = 1) -> str:
        """A chunk's text with the `span` chunks before and after it in the same document: a fact
        often straddles a chunk boundary, so checking a claim against its cited chunk alone
        rejects true sentences ("Justin appeared pale on the pharmacy doorstep", 0.39)."""
        if chunk.get("chunk_index") is None:  # a book card: whole already
            return chunk["text"]
        built_for, pos = self._positions
        if built_for != len(self.chunks):
            pos = {(c["doc_id"], c["chunk_index"]): i for i, c in enumerate(self.chunks)}
            self._positions = (len(self.chunks), pos)
        parts = [self.chunks[pos[(chunk["doc_id"], k)]]["text"]
                 for k in range(chunk["chunk_index"] - span, chunk["chunk_index"] + span + 1)
                 if (chunk["doc_id"], k) in pos]
        return "\n".join(parts) or chunk["text"]

    def proper_nouns(self) -> dict:
        """lowercase -> Capitalized for the library's names: words written capitalized mid-sentence
        at least 5 times and more often than in lowercase ("Justin", "Achab"; not "Era").
        Lets a question typed in lowercase ("chi era justin?") still be matched by name. Built
        on first use (~seconds for 50K chunks) and again after uploads or deletions."""
        built_for, names = self._names
        if built_for == len(self.chunks):
            return names
        texts = [c["text"] for c in self.chunks]
        capital, forms = Counter(), {}
        for t in texts:
            for w in _MID_CAPITAL.findall(t):
                capital[w.lower()] += 1
                forms.setdefault(w.lower(), w)
        candidates = {w for w, n in capital.items() if n >= 5}
        lower = Counter(w for t in texts for w in _LOWER_WORD.findall(t) if w in candidates)
        names = {w: forms[w] for w in candidates if capital[w] > lower[w]}
        self._names = (len(texts), names)
        return names

    def documents(self, pattern: str | None = None) -> list:
        docs = {}
        for c in self.chunks:
            d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "name": c["doc_name"],
                                               "chunks": 0, "pages": c.get("pages")})
            d["chunks"] += 1
        result = list(docs.values())
        if pattern:
            rx = re.compile(pattern, re.IGNORECASE)
            result = [d for d in result if rx.search(d["name"])]
        return result
