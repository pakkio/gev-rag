"""The two RAG pipelines, wired together. Each step records a trace entry
(name, duration, details) so the UI can show exactly what happened.

  INGEST:  file -> load -> chunk -> embed -> store
  QUERY:   question -> embed -> retrieve -> augment (build prompt) -> generate   (Claude)
                                        \-> select (pick a sentence)              (Jev)
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import chunker, embedder, generator, jev, loader
from .store import VectorStore

store = VectorStore()


class Trace:
    def __init__(self):
        self.steps = []

    def step(self, name: str, started: float, **details):
        self.steps.append({"step": name, "ms": round((time.perf_counter() - started) * 1000, 1), **details})


def ingest(filename: str, data: bytes, chunk_size: int, overlap: int) -> dict:
    trace = Trace()

    t = time.perf_counter()
    loaded = loader.load_text(filename, data)
    trace.step("load", t, characters=len(loaded["text"]), pages=loaded["pages"])
    if not loaded["text"].strip():
        raise ValueError("No text could be extracted (scanned PDFs need OCR).")

    t = time.perf_counter()
    chunks = chunker.chunk_text(loaded["text"], chunk_size, overlap)
    trace.step("chunk", t, count=len(chunks), chunk_size=chunk_size, overlap=overlap,
               preview=chunks[:3])

    t = time.perf_counter()
    vectors = embedder.embed_passages(chunks)
    trace.step("embed", t, model=embedder.MODEL_NAME, vectors=len(vectors),
               dimensions=int(vectors.shape[1]), sample=[round(float(x), 4) for x in vectors[0][:8]])

    t = time.perf_counter()
    doc_id = uuid.uuid4().hex[:8]
    store.add(vectors, [
        {"doc_id": doc_id, "doc_name": filename, "chunk_index": i, "text": c}
        for i, c in enumerate(chunks)
    ])
    trace.step("store", t, doc_id=doc_id, total_chunks_in_store=len(store.chunks))

    return {"doc_id": doc_id, "name": filename, "trace": trace.steps}


def retrieve(question: str, top_k: int, trace: Trace) -> list:
    t = time.perf_counter()
    qvec = embedder.embed_query(question)
    trace.step("embed_query", t, model=embedder.MODEL_NAME, dimensions=len(qvec),
               sample=[round(float(x), 4) for x in qvec[:8]])

    t = time.perf_counter()
    hits = store.search(qvec, top_k)
    trace.step("retrieve", t, top_k=top_k, searched=len(store.chunks),
               hits=[{"doc_name": h["doc_name"], "chunk_index": h["chunk_index"],
                      "score": round(h["score"], 4), "text": h["text"]} for h in hits])
    return hits


def answer_with_claude(question: str, hits: list, trace: Trace) -> dict:
    t = time.perf_counter()
    prompt = generator.build_prompt(question, hits)
    trace.step("augment", t, system=generator.SYSTEM_PROMPT, prompt=prompt, prompt_chars=len(prompt))

    t = time.perf_counter()
    result = generator.generate(prompt)
    trace.step("generate", t, model=result["model"] or "(disabled)", usage=result["usage"])
    return {"engine": "claude", "answer": result["answer"], "abstained": result.get("abstained"),
            "ms": round(sum(s["ms"] for s in trace.steps[-2:]), 1), "usage": result["usage"]}


def answer_with_jev(question: str, hits: list, trace: Trace) -> dict:
    t = time.perf_counter()
    result = jev.fast_answer(question, hits)
    trace.step("select", t, **{k: v for k, v in result.items() if k != "answer"})
    return {"engine": "jev", "answer": result["answer"], "abstained": result["abstained"],
            "ms": trace.steps[-1]["ms"], "usage": result["usage"],
            "answerable": result["answerable"], "confidence": result["confidence"]}


def ask(question: str, top_k: int, mode: str = "claude") -> dict:
    trace = Trace()
    hits = retrieve(question, top_k, trace)
    if not hits:
        return {"answers": [{"engine": mode, "answer": "No documents yet. Upload a file first."}],
                "sources": [], "trace": trace.steps}

    engines = {"claude": [answer_with_claude], "jev": [answer_with_jev],
               "both": [answer_with_claude, answer_with_jev]}[mode]
    # Each engine gets its own trace so their timings don't mix; in "both"
    # mode they run in parallel, as a real app would.
    branch_traces = [Trace() for _ in engines]
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        answers = list(pool.map(lambda fe: fe[0](question, hits, fe[1]), zip(engines, branch_traces)))
    for bt in branch_traces:
        trace.steps.extend(bt.steps)

    sources = [{"n": i, "doc_name": h["doc_name"], "chunk_index": h["chunk_index"],
                "score": round(h["score"], 4)} for i, h in enumerate(hits, start=1)]
    return {"answers": answers, "sources": sources, "trace": trace.steps,
            "retrieval_ms": round(sum(s["ms"] for s in trace.steps[:2]), 1)}
