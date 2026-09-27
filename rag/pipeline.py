"""The two RAG pipelines, wired together. Each step records a trace entry
(name, duration, details) so the UI can show exactly what happened.

  INGEST:  file -> load -> chunk -> embed -> store
  QUERY:   question -> embed -> retrieve -> augment (build prompt) -> generate   (Claude, or a
                                        |                                          local LLM)
                                        \-> select (pick a sentence)              (Jev)
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import chunker, embedder, gemini_llm, generator, jev, loader, local_llm, summarizer, summary_store
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


def answer_with_local(question: str, hits: list, trace: Trace) -> dict:
    t = time.perf_counter()
    prompt = generator.build_prompt(question, hits)
    trace.step("augment", t, system=generator.SYSTEM_PROMPT, prompt=prompt, prompt_chars=len(prompt))

    t = time.perf_counter()
    result = local_llm.generate(prompt)
    trace.step("generate_local", t, model=result["model"] or "(disabled)", usage=result["usage"])
    return {"engine": "local", "answer": result["answer"], "abstained": result["abstained"],
            "ms": round(sum(s["ms"] for s in trace.steps[-2:]), 1), "usage": result["usage"]}


def answer_with_gemini(question: str, hits: list, trace: Trace) -> dict:
    t = time.perf_counter()
    prompt = generator.build_prompt(question, hits)
    trace.step("augment", t, system=generator.SYSTEM_PROMPT, prompt=prompt, prompt_chars=len(prompt))

    t = time.perf_counter()
    result = gemini_llm.generate(prompt)
    trace.step("generate_gemini", t, model=result["model"] or "(disabled)", usage=result["usage"])
    return {"engine": "gemini", "answer": result["answer"], "abstained": result["abstained"],
            "ms": round(sum(s["ms"] for s in trace.steps[-2:]), 1), "usage": result["usage"]}


def answer_with_jev(question: str, hits: list, trace: Trace) -> dict:
    t = time.perf_counter()
    result = jev.fast_answer(question, hits)
    trace.step("select", t, **{k: v for k, v in result.items() if k != "answer"})
    return {"engine": "jev", "answer": result["answer"], "abstained": result["abstained"],
            "ms": trace.steps[-1]["ms"], "usage": result["usage"],
            "answerable": result["answerable"], "confidence": result["confidence"]}


ENGINES = {"claude": answer_with_claude, "local": answer_with_local, "jev": answer_with_jev,
           "gemini": answer_with_gemini}


def ask(question: str, top_k: int, engines: list) -> dict:
    trace = Trace()
    hits = retrieve(question, top_k, trace)
    if not hits:
        return {"answers": [{"engine": engines[0], "answer": "No documents yet. Upload a file first."}],
                "sources": [], "trace": trace.steps}

    # Each engine gets its own trace so their timings don't mix; when several
    # are selected they run in parallel, as a real app would.
    branch_traces = [Trace() for _ in engines]
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        answers = list(pool.map(lambda et: ENGINES[et[0]](question, hits, et[1]), zip(engines, branch_traces)))
    for bt in branch_traces:
        trace.steps.extend(bt.steps)

    sources = [{"n": i, "doc_name": h["doc_name"], "chunk_index": h["chunk_index"],
                "score": round(h["score"], 4)} for i, h in enumerate(hits, start=1)]
    return {"answers": answers, "sources": sources, "trace": trace.steps,
            "retrieval_ms": round(sum(s["ms"] for s in trace.steps[:2]), 1)}


def _doc_chunks(doc_id: str) -> tuple[str, list[str]] | None:
    """Ordered chunk texts for a doc_id, for whole-document reads (as opposed to
    retrieve()'s top-k similarity search). Returns (doc_name, texts) or None."""
    doc_chunks = [c for c in store.chunks if c["doc_id"] == doc_id]
    if not doc_chunks:
        return None
    doc_chunks.sort(key=lambda c: c["chunk_index"])
    return doc_chunks[0]["doc_name"], [c["text"] for c in doc_chunks]


def summarize_doc(doc_id: str) -> dict:
    """Map-reduce (or, when it fits, a single call) over a whole document into one
    cached summary. Unlike retrieve/ask, this reads every chunk, not just the top-k."""
    found = _doc_chunks(doc_id)
    if found is None:
        return {"error": "Document not found"}
    doc_name, texts = found

    result = summarizer.summarize(doc_name, texts)
    if result["summary"] is None:
        return {"error": result["error"], "trace": result["trace"]}

    entry = {"doc_id": doc_id, "doc_name": doc_name, "summary": result["summary"],
              "model": result["model"], "cost_usd": result["cost_usd"]}
    summary_store.set(doc_id, entry)
    return {**entry, "trace": result["trace"]}


ASK_DOC_SYSTEM = (
    "You are given the full text of a document. Answer the following question precisely "
    "and concisely, grounded only in the document's content. If the answer is not in the "
    "document, say so clearly rather than guessing. Answer in the same language as the "
    "question.\n\nQuestion: {question}"
)


def ask_doc(doc_id: str, question: str) -> dict:
    """One question against the WHOLE document (not top-k retrieval) — for questions
    retrieve()/ask() structurally can't answer, like 'what is this book about' or
    'does it mention X anywhere'. Not cached (unlike summarize_doc): the answer
    depends on the question, so there's nothing fixed to cache per doc_id."""
    found = _doc_chunks(doc_id)
    if found is None:
        return {"error": "Document not found"}
    doc_name, texts = found

    if not summarizer.fits_single_call(texts):
        return {"error": "Document too large for a single whole-document call "
                          "(no map-reduce fallback wired for open-ended questions yet)."}

    out = summarizer.whole_document(texts, ASK_DOC_SYSTEM.format(question=question), max_tokens=1500)
    if out["result"] is None:
        return {"error": out["error"], "trace": out["trace"]}
    return {"doc_id": doc_id, "doc_name": doc_name, "question": question, "answer": out["result"],
            "model": out["model"], "cost_usd": out["cost_usd"], "trace": out["trace"]}
