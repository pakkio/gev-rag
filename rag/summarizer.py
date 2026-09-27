"""Whole-document questions — the thing per-chunk retrieval can't answer.
`/api/ask` only ever sees the top-k chunks most similar to the question, so
"what is this document about?" or "list every X" has no single matching
chunk to retrieve. This reads the whole document instead of searching for
one piece of it, two ways:

  WHOLE_DOCUMENT: the entire document fits in one call — gemini-2.5-flash-lite
      has a 1M-token context window, so anything up to a few million
      characters just goes in as one request. Fast (one round trip) and
      usually the cheaper option too, since map-reduce repeats the full text
      once per batch plus the reduce passes. This is the common case; use it
      whenever the document fits.
  MAP_REDUCE: batch chunks by character budget -> summarize each batch ->
      reduce the batch summaries, recursively in groups, until one is left.
      Only needed for documents too large even for a 1M-token window. A
      single flat reduce over hundreds of verbose batch outputs (e.g. a
      "list every X" prompt, as opposed to a short summary) has no natural
      stopping point and can run to the model's max output tokens,
      degenerating into repeating itself — hierarchical reduce keeps each
      call's input and expected output small enough that it actually
      finishes.

Uses the gemini engine (cheapest wired-up LLM).
"""
import time
from concurrent.futures import ThreadPoolExecutor

from . import gemini_llm

SINGLE_CALL_CHAR_BUDGET = 3_000_000  # ~750K tokens, safely under the 1.05M-token context window
SINGLE_CALL_TIMEOUT = 180

BATCH_CHAR_BUDGET = 10000
REDUCE_GROUP_CHAR_BUDGET = 15000
MAP_WORKERS = 8
MAP_MAX_TOKENS = 600
REDUCE_MAX_TOKENS = 4000
FINAL_REDUCE_MAX_TOKENS = 8000  # the last, single-group reduce is the user-facing output

DIRECT_SUMMARY_SYSTEM = (
    "You are given the full text of a document. Write one short paragraph (4-6 sentences) "
    "describing what the document as a whole is about: its subject/genre, main content, "
    "and how it's structured. Write in the same language as the document. No preamble."
)
MAP_SYSTEM = (
    "You summarize one excerpt from a longer document. Write 2-3 concise sentences "
    "covering the concrete topics, events, or claims in this excerpt. "
    "Write in the same language as the excerpt. No preamble."
)
REDUCE_SYSTEM = (
    "You are given ordered section summaries covering an entire document, from start "
    "to end. Write one short paragraph (4-6 sentences) describing what the document as "
    "a whole is about: its subject/genre, main content, and how it's structured. "
    "Write in the same language as the summaries. No preamble."
)


DEFAULT_FREQUENCY_PENALTY = 0.4  # curbs the greedy-decoding repetition loop (see gemini_llm.chat)


def fits_single_call(chunks: list[str]) -> bool:
    return sum(len(c) for c in chunks) <= SINGLE_CALL_CHAR_BUDGET


def whole_document(chunks: list[str], system_prompt: str, max_tokens: int = FINAL_REDUCE_MAX_TOKENS,
                    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY) -> dict:
    """One call over the entire document. Only call this when fits_single_call() is
    true — the caller decides, since the right fallback (map_reduce, with what
    prompts) is context-dependent."""
    text = "\n\n".join(chunks)
    t0 = time.perf_counter()
    result = gemini_llm.chat(system_prompt, text, max_tokens=max_tokens, timeout=SINGLE_CALL_TIMEOUT,
                              frequency_penalty=frequency_penalty)
    trace = [{"step": "whole_document", "ms": round((time.perf_counter() - t0) * 1000, 1),
              "chunks": len(chunks), "chars": len(text), "model": result["model"],
              "tokens": result["usage"], "cost_usd": result["usage"]["cost_usd"] if not result["error"] else None}]
    if result["error"]:
        return {"result": None, "error": result["answer"], "trace": trace}
    return {"result": result["answer"], "model": result["model"],
            "cost_usd": result["usage"]["cost_usd"], "trace": trace}


def _batch(items: list[str], char_budget: int) -> list[str]:
    batches, current, size = [], [], 0
    for item in items:
        if current and size + len(item) > char_budget:
            batches.append("\n\n".join(current))
            current, size = [], 0
        current.append(item)
        size += len(item)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _reduce_level(labeled: list[str], reduce_system: str, label: str, trace: list) -> list[dict] | dict:
    """One level of reduction: group `labeled` by char budget, reduce each group in
    parallel. Returns the level's raw results (list) so the caller can decide whether
    another level is needed, or an {"error": ...} dict on failure."""
    groups = _batch(labeled, REDUCE_GROUP_CHAR_BUDGET)
    # A single group means this call produces the final, user-facing result — give it
    # more room than an intermediate merge, which gets consolidated again later anyway.
    max_tokens = FINAL_REDUCE_MAX_TOKENS if len(groups) == 1 else REDUCE_MAX_TOKENS
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(MAP_WORKERS, len(groups))) as pool:
        results = list(pool.map(
            lambda g: gemini_llm.chat(reduce_system, g, max_tokens=max_tokens,
                                       frequency_penalty=DEFAULT_FREQUENCY_PENALTY), groups))

    errors = [r["answer"] for r in results if r["error"]]
    if errors:
        return {"error": errors[0]}

    cost = sum(r["usage"]["cost_usd"] for r in results)
    trace.append({"step": label, "ms": round((time.perf_counter() - t0) * 1000, 1),
                  "groups_in": len(labeled), "calls": len(groups), "model": results[0]["model"],
                  "tokens": {"input": sum(r["usage"]["input_tokens"] for r in results),
                             "output": sum(r["usage"]["output_tokens"] for r in results)},
                  "cost_usd": round(cost, 6)})
    return results


def map_reduce(chunks: list[str], map_system: str = MAP_SYSTEM, reduce_system: str = REDUCE_SYSTEM) -> dict:
    """Generic map-reduce over every chunk of a document. `summarize()` (the cached,
    general-purpose summary) is the default-prompt case; pass custom map_system/
    reduce_system for other whole-document questions (e.g. "list every prediction")."""
    trace = []
    t0 = time.perf_counter()
    batches = _batch(chunks, BATCH_CHAR_BUDGET)
    trace.append({"step": "batch", "ms": round((time.perf_counter() - t0) * 1000, 1),
                  "chunks": len(chunks), "batches": len(batches), "char_budget": BATCH_CHAR_BUDGET})

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAP_WORKERS) as pool:
        map_results = list(pool.map(
            lambda b: gemini_llm.chat(map_system, b, max_tokens=MAP_MAX_TOKENS,
                                       frequency_penalty=DEFAULT_FREQUENCY_PENALTY), batches))

    errors = [r["answer"] for r in map_results if r["error"]]
    if errors:
        return {"result": None, "error": errors[0], "trace": trace}

    map_cost = sum(r["usage"]["cost_usd"] for r in map_results)
    trace.append({"step": "map", "ms": round((time.perf_counter() - t0) * 1000, 1),
                  "calls": len(batches), "model": map_results[0]["model"],
                  "tokens": {"input": sum(r["usage"]["input_tokens"] for r in map_results),
                             "output": sum(r["usage"]["output_tokens"] for r in map_results)},
                  "cost_usd": round(map_cost, 6)})

    # Reduce, recursively, until one group is left. Each level's total input is
    # bounded by REDUCE_GROUP_CHAR_BUDGET per call, so it always terminates.
    level = [f"[section {i+1}] {r['answer']}" for i, r in enumerate(map_results)]
    reduce_cost = 0.0
    level_num = 1
    while True:
        result = _reduce_level(level, reduce_system, f"reduce_L{level_num}", trace)
        if isinstance(result, dict) and "error" in result:
            return {"result": None, "error": result["error"], "trace": trace}
        reduce_cost += sum(r["usage"]["cost_usd"] for r in result)
        if len(result) == 1:
            final = result[0]
            break
        level = [f"[group {i+1}] {r['answer']}" for i, r in enumerate(result)]
        level_num += 1

    total_cost = round(map_cost + reduce_cost, 6)
    return {"result": final["answer"], "model": final["model"], "cost_usd": total_cost, "trace": trace}


def summarize(doc_name: str, chunks: list[str]) -> dict:
    out = whole_document(chunks, DIRECT_SUMMARY_SYSTEM, max_tokens=1000) if fits_single_call(chunks) \
        else map_reduce(chunks)
    if out["result"] is None:
        return {"summary": None, "error": out["error"], "trace": out["trace"]}
    return {"summary": out["result"], "doc_name": doc_name, "model": out["model"],
            "cost_usd": out["cost_usd"], "trace": out["trace"]}
