r"""The two RAG pipelines, wired together. Each step records a trace entry
(name, duration, details) so the UI can show exactly what happened.

  INGEST:  file -> load -> chunk -> embed -> store
  QUERY:   question -> embed -> retrieve -> augment (build prompt) -> generate   (Claude, or a
                                        |                                          local LLM)
                                        \-> select (pick a sentence)              (Jev)
"""
import os
import logging
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import chapters, chunker, embedder, gemini_llm, generator, jev, loader, local_llm, summarizer, summary_store
from .store import VectorStore, word_regex

store = VectorStore()
log = logging.getLogger("uvicorn.error")  # uvicorn's logger: shows up in the server log


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
        {"doc_id": doc_id, "doc_name": filename, "chunk_index": i, "text": c, "pages": loaded["pages"]}
        for i, c in enumerate(chunks)
    ])
    trace.step("store", t, doc_id=doc_id, total_chunks_in_store=len(store.chunks))

    return {"doc_id": doc_id, "name": filename, "trace": trace.steps}


# Capitalized words that aren't names: question words and sentence starters.
_NOT_NAMES = {"chi", "che", "cosa", "come", "quale", "quali", "quando", "dove", "perché", "quanti",
              "quante", "in", "il", "la", "lo", "le", "gli", "un", "una", "di", "del", "della", "nel",
              "nella", "who", "what", "which", "how", "when", "where", "why", "the", "is", "are", "does",
              "did", "do", "was", "were", "tell", "describe", "explain", "give", "list", "name",
              "can", "could", "summarize", "dimmi", "descrivi", "spiega", "elenca", "riassumi"}


def question_names(question: str) -> list[str]:
    """Proper names in the question ("Justin", "Hippolyte"), used to boost chunks containing them:
    capitalized words, and lowercase ones that are names in the library ("chi era justin?")."""
    names = store.proper_nouns()
    words = re.findall(r"\b[\wÀ-ÿ][\wà-ÿ'’]{2,}", question)
    found = [names.get(w.lower(), w) for w in words
             if w.lower() not in _NOT_NAMES and (w[0].isupper() or w.lower() in names)]
    return list(dict.fromkeys(found))


def retrieve(question: str, top_k: int, trace: Trace, doc_id: str | None = None) -> list:
    t = time.perf_counter()
    qvec = embedder.embed_query(question)
    trace.step("embed_query", t, model=embedder.MODEL_NAME, dimensions=len(qvec),
               sample=[round(float(x), 4) for x in qvec[:8]])

    t = time.perf_counter()
    names = question_names(question)
    hits = store.search(qvec, top_k, doc_id=doc_id, boost_terms=names)
    cards = summary_store.all()
    for h in hits:
        h["chapter"] = chapters.chapter_of(cards.get(h["doc_id"], {}).get("chapters", []), h["chunk_index"])
    trace.step("retrieve", t, top_k=top_k, searched=len(store.chunks), doc_id=doc_id, boosted_names=names,
               hits=[{"doc_name": h["doc_name"], "chunk_index": h["chunk_index"], "chapter": h["chapter"],
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


def answer_with_jev(question: str, hits: list, trace: Trace, emit=None, cancel=None) -> dict:
    """Two answers in parallel. Jev selects one sentence (with its neighbours): a verbatim quote
    in ~0.5 s, but by design short, and one sentence can't cover a broad question ("what does
    the book predict about the 20th century?"). Meanwhile an LLM drafts a detailed, cited answer
    that Jev checks sentence by sentence (verified_answer). The result carries both: "quote" is
    Jev's selection, "answer" the checked draft, or the quote when there's no usable draft.
    emit(event) streams both as they happen (ask_stream); None = no stream."""
    t0 = time.perf_counter()
    emit = emit or (lambda event: None)
    draft_trace = Trace()
    with ThreadPoolExecutor(max_workers=1) as pool:
        draft = (pool.submit(verified_answer, question, hits, draft_trace, emit, None, cancel)
                 if _can_verify() else None)
        t = time.perf_counter()
        result = jev.fast_answer(question, hits)
        trace.step("select", t, **{k: v for k, v in result.items() if k != "answer"})
        out = {"engine": "jev", "answer": result["answer"], "abstained": result["abstained"],
               "ms": trace.steps[-1]["ms"], "usage": result["usage"],
               "answerable": result["answerable"], "confidence": result["confidence"]}
        if result["abstained"]:
            emit({"type": "jev_abstained", "engine": "jev", "answerable": result["answerable"],
                  "confidence": result["confidence"]})
        elif result["usage"]:
            out["quote"] = result["answer"]
            emit({"type": "jev_quote", "engine": "jev", "text": result["answer"], "ms": out["ms"],
                  "answerable": result["answerable"], "confidence": result["confidence"]})
        verified = draft.result() if draft else None
    trace.steps.extend(draft_trace.steps)
    if verified:
        usages = [u for u in (result["usage"], verified.pop("flash_usage"), verified.pop("jev_usage")) if u]
        out.update(verified, abstained=False, ms=round((time.perf_counter() - t0) * 1000, 1),
                   usage={k: round(sum(u[k] for u in usages), 6) for k in ("input_tokens", "output_tokens", "cost_usd")})
    return out


# The detailed answer next to Jev's quote. For a question naming someone ("Chi è Justin?") the
# passages are every one naming him: no single sentence describes Justin, his portrait is spread
# over ~50 passages. Otherwise they're the retrieved chunks plus the book cards, which answer
# broad questions ("previsioni sul XX secolo"). An LLM writes from them, citing passages per
# sentence, and Jev checks each sentence against
# the passages it cites as soon as the sentence is complete, and sorts it into three:
#   fact   supported >= FACT_AT and stated as fact (not a character's guess) -> kept as is
#   guess  supported >= GUESS_AT but not a fact: embroidered, or someone's suspicion in the
#          story (Homais teasing that Justin loves Félicité) -> kept as [[...]]
#   drop   less supported than that -> removed
# Swept on 41 draft sentences about four characters: 0.4-0.7 is where partly-true, embellished
# sentences sit ("an unrequited love for Emma", 0.49); below 0.4 they were mostly invented.
MENTIONS_LIMIT = 40  # ~32K chars of passages: enough for a minor character, cheap for the LLM
MENTIONS_MIN = 5
MENTIONS_MAX = 120  # Justin: 47 passages; Emma: 670, where her mentions no longer pick anything out
FACT_AT, GUESS_AT, FACTUAL_AT = 0.7, 0.4, 0.6
# The draft writer: the chat Flash model (deepseek-v4-flash, reasoning off). Over 12 drafts on six
# questions Jev confirmed 57% of its sentences as facts and removed 8; ling-3.0-flash was 4x
# faster (2.3 s vs ~18 s median) but got 36% confirmed, 23 removed, and 0 facts on Justin twice
# ("Homais's adopted son"). The draft streams sentence by sentence after Jev's instant quote,
# so its length shows less. ling is the fallback if the first model fails before writing.
VERIFIED_MODEL = os.environ.get("VERIFIED_MODEL", gemini_llm.MODEL)
VERIFIED_FALLBACK = "inclusionai/ling-3.0-flash"
VERIFIED_SYSTEM = (
    "You are given numbered passages from a library of books, chosen for the question: either "
    "every passage naming the person or thing it asks about, or the passages most related to it "
    "plus a summary card of each book. Answer the question in detail, in the same language as the "
    "question (for a person or thing: who or what it is, their role, relationships, what they do "
    "and how they are described). Start with who it is and its part in the story's key events "
    "(what it does that changes the main characters' fate), then the rest. Write 5 to 10 "
    "sentences, most important first, each adding a "
    "concrete detail from the passages (events, scenes, names, places, words quoted), not a "
    "restatement of the same summary. Keep each sentence to one fact from one or two passages, so "
    "it can be checked against them. Take details from the book's own passages; use the summary "
    "cards only for context. State the content directly: never mention passages, cards or "
    "summaries in the answer. End every "
    "sentence with the numbers of the passages it is based on, like [3] or [3][7]. Each sentence "
    "must say only what its cited passages say about this person or thing, not about someone "
    "else; leave out what the passages don't make clear. No closing summary sentence. If the "
    f"passages say nothing about it, reply only \"{generator.ABSTAIN_PREFIX}\""
)
# Sentence ends: after . ! ? or a citation's ], when a new sentence or list item starts. Not
# before "[", so a citation after the period ("... Homais. [4]") stays with its sentence.
_ANSWER_SENTENCE = re.compile(r"(?<=[.!?\]])\s+(?=[A-ZÀ-Ý\"'(*\-])")


# Remarks about the sources rather than the book ("come descritto nel riassunto della trama",
# "la card precisa che", "as the summary says"): the model makes them despite the prompt, and
# the reader doesn't know what a card or passage is. Removed from each drafted sentence.
_SOURCE_WORDS = r"(?:riassunt\w*|sintesi|sched\w*|card|passagg\w*|bran\w*|trama fornita|summar\w*|passages?|excerpts?|overview)"
# A clause opened by a comma takes its closing comma with it: "Ahab, as the summary says, is" -> "Ahab is".
_META_CLAUSE = re.compile(r"\s*(,)?\s*\b(?:come|secondo|stando|as|according to)\b[^,.;\[]*?\b" + _SOURCE_WORDS
                          + r"\b[^,.;\[]*(?(1),?)", re.I)
_META_LEAD = re.compile(r"^(?:(?:la|il|lo|l'|questa|questo|the|this)\s*)?" + _SOURCE_WORDS + r"\b[^,.;\[]*?\b(?:che|that)\b[\s,]*", re.I)


def _strip_source_talk(sentence: str) -> str:
    out = _META_LEAD.sub("", _META_CLAUSE.sub("", sentence))
    out = re.sub(r"(\w)\[", r"\1 [", out).strip(" ,;")
    return out[:1].upper() + out[1:] if out else sentence


def _can_verify() -> bool:
    return gemini_llm.gemini_enabled() and jev.jev_enabled()


def _complete_sentences(buffer: str, final: bool) -> tuple[list[tuple[str, bool]], str]:
    """Split the finished sentences off a streaming buffer: ([(sentence, ends_its_line)], rest).
    The last sentence of an unfinished line stays in rest: it may still grow."""
    *lines, rest = buffer.split("\n")
    if final:
        lines, rest = lines + [rest], ""
    out = []
    for line in lines:
        parts = [s.strip() for s in _ANSWER_SENTENCE.split(line) if s.strip()]
        out += [(s, i == len(parts) - 1) for i, s in enumerate(parts)]
    parts = _ANSWER_SENTENCE.split(rest)
    out += [(s.strip(), False) for s in parts[:-1] if s.strip()]
    return out, parts[-1]


def _status(supported: float | None, factual: float | None) -> str:
    if supported is None or supported < GUESS_AT:
        return "drop"
    return "fact" if supported >= FACT_AT and factual >= FACTUAL_AT else "guess"


def _as_guess(sentence: str, cited: list) -> str:
    """"[[sentence]] [3][7]": the doubtful part in double brackets, its citations outside, so a
    "]" from a citation can't close the brackets early."""
    return f"[[{re.sub(r'\s*\[\d+\]', '', sentence).strip()}]] " + "".join(f"[{n}]" for n in cited)


def _no_fallback(trace: Trace, t: float, reason: str) -> None:
    trace.step("fallback", t, reason=reason)
    return None


def verified_answer(question: str, hits: list, trace: Trace, emit=None, cancel=None,
                    outer_cancel=None) -> dict | None:
    emit = emit or (lambda event: None)
    cancel = cancel or threading.Event()
    t = time.perf_counter()
    if not _can_verify():
        return _no_fallback(trace, t, "needs both the LLM (OpenRouter) and Jev on")
    qvec = embedder.embed_query(question)
    doc_ids = {h["doc_id"] for h in hits}
    # The subject is the rarest name that isn't part of a book's title: in "Chi è Justin in
    # Madame Bovary?" it's Justin, not Bovary (everywhere in that book) nor Madame (rare in the
    # Italian text, which says "signora", but it only names the book).
    titles = " ".join(h["doc_name"] for h in hits).lower()
    names = question_names(question)
    found = [(n, *store.mentions(qvec, n, doc_ids, MENTIONS_LIMIT)) for n in names if n.lower() not in titles]
    # Mentions only pick out the right passages for a name in MENTIONS_MIN..MENTIONS_MAX of them.
    # Above, it's a protagonist (Emma: 670) and its mentions say nothing about which passages
    # answer; below, it's often not the subject at all ("Captain" in "Who is Captain Ahab?":
    # 2 English passages, while the Italian Moby Dick says "Achab"). The retrieved ones do better.
    found = [f for f in found if MENTIONS_MIN <= f[1] <= MENTIONS_MAX]
    if found:
        name, total, passages = min(found, key=lambda f: f[1])
    else:  # no name, or none found: the retrieved chunks and book cards
        name, total, passages = None, len(hits), hits[:MENTIONS_LIMIT]
    trace.step("mentions", t, entity=name or "(retrieved passages)", total=total, used=len(passages),
               docs=sorted({p["doc_name"] for p in passages}))
    if cancel.is_set():
        return None
    emit({"type": "draft_start", "engine": "jev", "entity": name, "passages": len(passages)})

    sentences, verdicts, jev_usages, jev_model, lock = [], {}, [], [None], threading.Lock()

    def check(i: int, sentence: str, cited: list):
        # Each cited chunk with its neighbours: on 41 draft sentences this turned 11 true ones from
        # dropped/[[ ]] into facts (or dropped into [[ ]]) and only 2 the other way.
        v = jev.verify([(re.sub(r"\s*\[\d+\]", "", sentence), [store.with_neighbors(passages[n - 1]) for n in cited])])
        p, f = (None, None) if v["error"] else (v["supported"][0], v["factual"][0])
        status = _status(p, f)
        with lock:
            verdicts[i] = (status, p, f)
            if v["usage"]:
                jev_usages.append(v["usage"])
                jev_model[0] = v["model"]
        if not cancel.is_set():
            emit({"type": "verdict", "engine": "jev", "i": i, "status": status, "supported": p, "factual": f,
                  "text": _as_guess(sentence, cited) if status == "guess" else sentence})

    t = time.perf_counter()
    prompt, buffer, result = generator.build_prompt(question, passages), "", None
    with ThreadPoolExecutor(max_workers=4) as checks:
        def finished(done: list):
            for s, ends_line in done:
                s = _strip_source_talk(s)
                i = len(sentences)
                cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", s) if 1 <= int(n) <= len(passages)})
                sentences.append((s, ends_line, cited))
                emit({"type": "draft_sentence", "engine": "jev", "i": i, "text": s, "ends_line": ends_line})
                if cited:
                    checks.submit(check, i, s, cited)
                else:  # an uncited sentence can't be checked: dropped
                    verdicts[i] = ("drop", None, None)
                    emit({"type": "verdict", "engine": "jev", "i": i, "status": "drop", "supported": None,
                          "factual": None, "text": s})

        for model in dict.fromkeys((VERIFIED_MODEL, VERIFIED_FALLBACK)):
            for kind, x in gemini_llm.chat_stream(VERIFIED_SYSTEM, prompt, max_tokens=2500, timeout=90, model=model,
                                                  cancel=cancel, reasoning=False):
                if outer_cancel is not None and outer_cancel.is_set():
                    cancel.set()
                if kind == "delta":
                    done, buffer = _complete_sentences(buffer + x, final=False)
                    finished(done)
                    emit({"type": "draft_tail", "engine": "jev", "text": _META_CLAUSE.sub("", buffer)})
                else:
                    result = x
            if not result["error"] or sentences or cancel.is_set():
                break
        if cancel.is_set():
            return None
        done, buffer = _complete_sentences(buffer, final=True)
        finished(done)
        emit({"type": "draft_tail", "engine": "jev", "text": ""})
        trace.step("generate_gemini", t, model=result.get("model") or "(error)", usage=result.get("usage"))
        t = time.perf_counter()  # verify's own time: the checks still running once the draft is written
    if result["error"]:
        return _no_fallback(trace, t, f"LLM error: {result['answer']}")
    if generator.is_abstained(result["answer"]):
        return _no_fallback(trace, t, "the LLM found nothing about it in the passages")

    status = {i: verdicts.get(i, ("drop", None, None))[0] for i in range(len(sentences))}
    kept = [(_as_guess(s, c) if status[i] == "guess" else s, ends_line, c)
            for i, (s, ends_line, c) in enumerate(sentences) if status[i] != "drop"]
    counts = {k: sum(v == k for v in status.values()) for k in ("fact", "guess", "drop")}
    jev_usage = {k: round(sum(u[k] for u in jev_usages), 6) for k in ("input_tokens", "output_tokens", "cost_usd")}
    trace.step("verify", t, model=jev_model[0], usage=jev_usage, thresholds={"fact": FACT_AT, "guess": GUESS_AT, "factual": FACTUAL_AT},
               counts=counts,
               claims=[{"sentence": s, "status": status[i], "supported": verdicts.get(i, (None, None, None))[1],
                        "factual": verdicts.get(i, (None, None, None))[2]} for i, (s, _, _) in enumerate(sentences)])
    # Suppositions only make sense around confirmed facts: a draft where Jev confirmed none
    # ("Justin was Homais's adopted son", kept as [[...]] at 0.45) is unreliable as a whole.
    if not counts["fact"]:
        return _no_fallback(trace, t, f"Jev confirmed no fact in the draft ({counts['guess']} suppositions, "
                                      f"{counts['drop']} removed)")
    answer = "".join(s + ("\n" if ends_line else " ") for s, ends_line, _ in kept).strip()
    cited = sorted({n for _, _, c in kept for n in c})
    return {"answer": answer, "mode": "verified", "verified": f"{counts['fact']}/{len(sentences)}",
            "guesses": counts["guess"], "dropped": counts["drop"],
            "cited": [{"n": n, "doc_name": passages[n - 1]["doc_name"],
                       "chunk_index": passages[n - 1]["chunk_index"]} for n in cited],
            "flash_usage": result["usage"], "jev_usage": jev_usage}


ENGINES = {"claude": answer_with_claude, "local": answer_with_local, "jev": answer_with_jev,
           "gemini": answer_with_gemini}


def _attach_cards(hits: list) -> list:
    """Append the book card (whole-document summary) of up to two retrieved docs.

    Top-k chunk retrieval can't answer definitional questions ("who is Anna
    Karenina?"): the intro sentence sits at rank ~90 while scene chunks cluster
    at the top. The card states it directly, so Jev can select it verbatim and
    the generate engines can cite it. Benchmarks call retrieve/ENGINES directly
    and don't go through ask(), so they keep comparing raw retrieval.
    """
    out = list(hits)
    cards = [x for x in out if x.get("is_card")]
    for h in hits:
        if len(cards) >= 2:
            break
        if any(x["doc_id"] == h["doc_id"] for x in cards):
            continue
        entry = summary_store.get(h["doc_id"])
        if entry and entry.get("summary"):
            out.append({"doc_id": h["doc_id"], "doc_name": h["doc_name"], "chunk_index": None,
                        "chapter": "book card", "score": h["score"], "is_card": True,
                        "text": entry["summary"]})
            cards.append(out[-1])
    return out


# Follow-up questions in a chat ("e come muore?") are rewritten into a standalone question from
# the conversation before anything else runs, so retrieval and every engine, Jev included, see
# who or what it is about. The UI sends the last 20 turns; "New chat" (Ctrl+K) sends none.
CONDENSE_SYSTEM = (
    "You rewrite the latest question of a conversation about books so it can be understood on its "
    "own. Use the conversation only to resolve what it refers to (he, she, it, that book, the "
    "character, \"e lui?\", \"come muore?\", \"and his wife?\"): add the names of the people and "
    "books meant. Keep the language and the meaning of the question; don't answer it. If it is "
    "already self-contained or changes topic, return it unchanged. Reply with the question only."
)
CONDENSE_MODEL = "inclusionai/ling-3.0-flash"  # a one-line rewrite: fast beats smart (~0.5 s)
HISTORY_TURNS, HISTORY_ANSWER_CHARS = 20, 600  # ~4K tokens at most: still ~$0.0001 and < 1 s with ling


def condense(question: str, history: list | None, trace: Trace) -> str:
    """The question made standalone using the chat history, or unchanged without history."""
    if not history or not gemini_llm.gemini_enabled():
        return question
    t = time.perf_counter()
    convo = "\n".join(f"Q: {h.get('q', '')}\nA: {(h.get('a') or '')[:HISTORY_ANSWER_CHARS]}"
                      for h in history[-HISTORY_TURNS:])
    r = gemini_llm.chat(CONDENSE_SYSTEM, f"Conversation:\n{convo}\n\nLatest question: {question}",
                        max_tokens=200, timeout=20, model=CONDENSE_MODEL, reasoning=False)
    rewritten = question if r["error"] else r["answer"].strip().strip('"').splitlines()[0].strip() or question
    trace.step("condense", t, model=r.get("model") or "(error)", usage=r.get("usage"), original=question,
               rewritten=rewritten, turns=len(history[-HISTORY_TURNS:]))
    return rewritten


# Jev reads more chunks than the generate engines: its input costs ~$0.04/MTok, and on
# book_questions/ 24 chunks gave 18/30 right answers vs 15/30 with 12.
JEV_TOP_K = int(os.environ.get("JEV_TOP_K", "24"))


def ask(question: str, top_k: int, engines: list, doc_id: str | None = None, history: list | None = None) -> dict:
    trace = Trace()
    question = condense(question, history, trace)
    depth = max(top_k, JEV_TOP_K) if "jev" in engines else top_k
    deep = retrieve(question, depth, trace, doc_id)
    if not deep:
        return {"answers": [{"engine": engines[0], "answer": "No documents yet. Upload a file first."}],
                "sources": [], "trace": trace.steps}
    hits = _attach_cards(deep[:top_k])
    jev_hits = _attach_cards(deep)

    # Each engine gets its own trace so their timings don't mix; when several
    # are selected they run in parallel, as a real app would.
    branch_traces = [Trace() for _ in engines]
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        answers = list(pool.map(lambda et: ENGINES[et[0]](question, jev_hits if et[0] == "jev" else hits, et[1]),
                                zip(engines, branch_traces)))
    costs = cost_lines("chat", trace.steps)
    for engine, answer, bt in zip(engines, answers, branch_traces):
        costs += _with_costs(answer, cost_lines(engine, bt.steps))["costs"]
    total = log_costs(question, trace.steps, costs)
    for bt in branch_traces:
        trace.steps.extend(bt.steps)

    sources = [{"n": i, "doc_name": h["doc_name"], "chunk_index": h["chunk_index"], "chapter": h["chapter"],
                "score": round(h["score"], 4)} for i, h in enumerate(hits, start=1)]
    return {"answers": answers, "sources": sources, "trace": trace.steps, "costs": costs, "total_cost_usd": total,
            "retrieval_ms": round(sum(s["ms"] for s in trace.steps if s["step"] in ("embed_query", "retrieve")), 1)}


# What each paid (or free) call in a question was for, per engine: read back from the trace,
# so calls whose result was thrown away (a draft Jev rejected entirely) still count.
_COST_TASKS = {
    ("claude", "generate"): "risposta dai passaggi recuperati",
    ("local", "generate_local"): "risposta dai passaggi recuperati (GPU locale, gratis)",
    ("gemini", "generate_gemini"): "risposta dai passaggi recuperati",
    ("jev", "select"): "citazione: sceglie la frase che risponde",
    ("jev", "generate_gemini"): "risposta dettagliata: scrive la bozza",
    ("jev", "verify"): "risposta dettagliata: verifica le frasi",
    ("chat", "condense"): "chat: riscrive la domanda col contesto",
}


def cost_lines(engine: str, steps: list) -> list:
    out = []
    for s in steps:
        task, u = _COST_TASKS.get((engine, s["step"])), s.get("usage")
        if not task or not u:
            continue
        if s["step"] == "verify":
            task += f" ({len(s.get('claims', []))})"
        out.append({"engine": engine, "model": s.get("model") or "?", "task": task,
                    "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                    "cost_usd": u.get("cost_usd") or 0.0})
    return out


def _with_costs(answer: dict, lines: list) -> dict:
    """Attach the cost lines; the answer's own cost becomes their sum, so it includes calls
    whose output was dropped (a rejected draft still costs what it cost)."""
    answer["costs"] = lines
    if answer.get("usage") and lines:
        answer["usage"] = {**answer["usage"], "cost_usd": round(sum(c["cost_usd"] for c in lines), 6)}
    return answer


def log_costs(question: str, retrieval_steps: list, lines: list) -> float:
    """Print a question's cost breakdown in the server log, one line per call, and the total."""
    embed = next((s for s in retrieval_steps if s["step"] == "embed_query"), None)
    rows = ([{"engine": "ricerca", "model": embed["model"], "task": "embedding della domanda (GPU locale, gratis)",
              "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}] if embed else []) + lines
    total = round(sum(r["cost_usd"] for r in rows), 6)
    out = [f"── costi · {question[:70]!r} " + "─" * 20]
    for r in rows:
        tok = f"{r['input_tokens']}+{r['output_tokens']} tok" if r["input_tokens"] or r["output_tokens"] else "-"
        out.append(f"   {r['engine']:<8} {r['model'][:38]:<38} {r['task'][:58]:<58} {tok:>14}  ${r['cost_usd']:.6f}")
    out.append(f"   {'TOTALE':<8} {'':<38} {'':<58} {'':>14}  ${total:.6f}")
    log.info("\n".join(out))
    return total


def _stream_gemini(question: str, hits: list, trace: Trace, emit, cancel) -> dict:
    """answer_with_gemini, streaming its tokens through emit as they arrive."""
    t = time.perf_counter()
    prompt = generator.build_prompt(question, hits)
    trace.step("augment", t, system=generator.SYSTEM_PROMPT, prompt=prompt, prompt_chars=len(prompt))
    t = time.perf_counter()
    result = None
    for kind, x in gemini_llm.chat_stream(generator.SYSTEM_PROMPT, prompt, max_tokens=1500, cancel=cancel,
                                          reasoning=gemini_llm.FLASH_REASONING):
        if kind == "delta":
            emit({"type": "delta", "engine": "gemini", "text": x})
        else:
            result = x
    trace.step("generate_gemini", t, model=result["model"] or "(disabled)", usage=result["usage"])
    return {"engine": "gemini", "answer": result["answer"],
            "abstained": None if result["error"] else generator.is_abstained(result["answer"]),
            "ms": round(sum(s["ms"] for s in trace.steps[-2:]), 1), "usage": result["usage"]}


def ask_stream(question: str, top_k: int, engines: list, doc_id: str | None = None, history: list | None = None):
    """ask() as a stream of events, for the UI: sources first, then each engine's tokens
    (Flash), Jev's draft being written and checked, and each final answer as it lands;
    "done" carries the full trace. Engines run in parallel; the slowest never holds up the rest."""
    trace = Trace()
    original, question = question, condense(question, history, trace)
    if question != original:
        yield {"type": "condensed", "question": question}
    depth = max(top_k, JEV_TOP_K) if "jev" in engines else top_k
    deep = retrieve(question, depth, trace, doc_id)
    if not deep:
        yield {"type": "answer", "answer": {"engine": engines[0], "answer": "No documents yet. Upload a file first."}}
        yield {"type": "done", "trace": trace.steps}
        return
    hits = _attach_cards(deep[:top_k])
    jev_hits = _attach_cards(deep)
    yield {"type": "sources", "engines": engines,
           "sources": [{"n": i, "doc_name": h["doc_name"], "chunk_index": h["chunk_index"], "chapter": h["chapter"],
                        "score": round(h["score"], 4)} for i, h in enumerate(hits, start=1)]}

    events, cancel = queue.Queue(), threading.Event()
    traces = {e: Trace() for e in engines}

    def run(engine: str):
        try:
            if engine == "gemini":
                answer = _stream_gemini(question, hits, traces[engine], events.put, cancel)
            elif engine == "jev":
                answer = answer_with_jev(question, jev_hits, traces[engine], events.put, cancel)
            else:
                answer = ENGINES[engine](question, hits, traces[engine])
        except Exception as e:  # one engine failing must not end the stream for the others
            answer = {"engine": engine, "answer": f"Error: {e}", "abstained": None, "ms": None, "usage": None}
        _with_costs(answer, cost_lines(engine, traces[engine].steps))
        events.put({"type": "answer", "answer": answer})
        events.put(None)  # this engine is finished

    pool = ThreadPoolExecutor(max_workers=len(engines))
    try:
        for e in engines:
            pool.submit(run, e)
        running = len(engines)
        while running:
            event = events.get()
            if event is None:
                running -= 1
            else:
                yield event
    finally:
        cancel.set()  # client gone or done: stop any stream still open
        pool.shutdown(wait=False)
    costs = cost_lines("chat", trace.steps) + [c for e in engines for c in cost_lines(e, traces[e].steps)]
    total = log_costs(question, trace.steps, costs)
    for e in engines:
        trace.steps.extend(traces[e].steps)
    yield {"type": "done", "trace": trace.steps, "costs": costs, "total_cost_usd": total}


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
              "chapters": result["chapters"], "model": result["model"], "cost_usd": result["cost_usd"]}
    summary_store.set(doc_id, entry)
    return {**entry, "trace": result["trace"]}


# OpenRouter models tried in order until one answers: a model can reject the request (a book
# longer than its context) or return an empty answer, and the next one then takes over.
# Picked by two benchmarks (13 library questions incl. traps, 4+1 whole-book questions):
# ling-3.0-flash scored full marks on library questions at ~0.05 cents (gemini-2.5-flash: same
# score, ~0.6 cents) but has a 262K context, too small for long novels; deepseek-v4-flash reads
# 1M tokens at ~3 cents per whole-book question. Comma-separated in the env.
DEFAULT_CLOUD_MODELS = "inclusionai/ling-3.0-flash,deepseek/deepseek-v4-flash"


def _models_env(name: str) -> list[str]:
    models = [m.strip() for m in os.environ.get(name, DEFAULT_CLOUD_MODELS).split(",") if m.strip()]
    return models or DEFAULT_CLOUD_MODELS.split(",")


ASK_ALL_MODELS = _models_env("ASK_ALL_MODELS")


def _first_answer(models: list[str], call) -> dict:
    """call(model) -> gemini_llm.chat-style result; returns the first non-error one, with the
    failures of the models before it, or the last error if every model failed."""
    failed = []
    for m in models:
        r = call(m)
        if not r["error"]:
            return {**r, "failed": failed}
        failed.append({"model": m, "error": r["answer"]})
    return {"error": True, "answer": failed[-1]["error"], "failed": failed}


ASK_ALL_SYSTEM = (
    "You are given reference cards for every book in a library, numbered [1]..[n]. Answer the "
    "question that follows the cards using only these cards, and cite the books you rely on as [n]. "
    "Answer directly, without disclaimers, whenever the cards contain the answer; only if they "
    "genuinely don't, say so instead of guessing. Answer in the same language as the question."
)


# "cloud" (ASK_ALL_MODELS via OpenRouter), "local" (Ollama) or "jev" (one sentence from a card). Local needs a context as big as all
# the cards (~30K tokens), which doesn't fit an 8 GB GPU next to the model: Ollama then silently
# runs it on the CPU, at minutes per question.
ASK_ALL_ENGINE = os.environ.get("ASK_ALL_ENGINE", "cloud")
ASK_ALL_NUM_CTX = 40960  # 26 cards are ~30K tokens; Ollama silently truncates past num_ctx


def ask_all(question: str, engine: str | None = None, model: str | None = None) -> dict:
    """A question about the whole library ("which novels end with the hero's death?"), answered
    from the cached per-book cards in one LLM call. Books without a card are reported, not
    silently ignored, since the answer can't account for them."""
    cards = summary_store.all()
    docs = store.documents()
    have = [d for d in docs if d["doc_id"] in cards]
    missing = [d["name"] for d in docs if d["doc_id"] not in cards]
    if not have:
        return {"error": "No book cards yet — build them with `just summarize-all`."}
    engine = engine or ASK_ALL_ENGINE
    t = time.perf_counter()
    if engine == "jev":
        return _ask_all_jev(question, have, cards, missing, t)
    labeled = [f"[{i}] {d['name']}\n{cards[d['doc_id']]['summary']}" for i, d in enumerate(have, start=1)]
    # The question goes after the ~30K tokens of cards, not in the system prompt before them:
    # placed first, flash answered "the cards say nothing about Moby Dick" with card [22] on it.
    content = "\n\n".join(labeled + [f"Question: {question}"])
    if engine == "cloud":
        r = _first_answer([model] if model else ASK_ALL_MODELS,
                          lambda m: gemini_llm.chat(ASK_ALL_SYSTEM, content, max_tokens=2000, timeout=120, model=m))
    else:
        r = {**local_llm.chat(ASK_ALL_SYSTEM, content, num_ctx=ASK_ALL_NUM_CTX, max_tokens=2000), "failed": []}
    if r["error"]:
        return {"error": r["answer"], "failed": r["failed"]}
    return {"question": question, "answer": r["answer"], "books": [d["name"] for d in have],
            "missing_cards": missing, "model": r["model"], "failed": r["failed"],
            "cost_usd": r["usage"]["cost_usd"], "ms": round((time.perf_counter() - t) * 1000), "usage": r["usage"]}


_card_vectors: dict = {}  # doc_id -> (summary, vector); cards change only on summarize


def _ask_all_jev(question: str, have: list, cards: dict, missing: list, t: float) -> dict:
    """Jev picks one sentence from the cards closest to the question. It can't combine cards
    (no lists like "all Russian authors"), but for one-fact questions it answers in ~0.3 s
    with a sentence copied from a card, so it can't invent anything."""
    for d in have:
        summary = cards[d["doc_id"]]["summary"]
        if _card_vectors.get(d["doc_id"], (None,))[0] != summary:
            _card_vectors[d["doc_id"]] = (summary, embedder.embed_passages([f"{d['name']}\n{summary}"])[0])
    qvec = embedder.embed_query(question)
    names = question_names(question)
    scored = sorted(have, reverse=True, key=lambda d: float(_card_vectors[d["doc_id"]][1] @ qvec)
                    + 0.1 * sum(bool(word_regex(n).search(cards[d["doc_id"]]["summary"])) for n in names))
    top = scored[:3]
    hits = [{"text": cards[d["doc_id"]]["summary"], "doc_name": d["name"], "is_card": True} for d in top]
    r = jev.fast_answer(question, hits)
    if r["usage"] is None:
        return {"error": r["answer"]}
    return {"question": question, "answer": r["answer"], "books": [d["name"] for d in top],
            "missing_cards": missing, "model": "jev", "failed": [], "abstained": r["abstained"],
            "answerable": r["answerable"], "confidence": r["confidence"], "cost_usd": r["usage"]["cost_usd"],
            "ms": round((time.perf_counter() - t) * 1000), "usage": r["usage"]}


# Same chain as ASK_ALL_MODELS: ling-3.0-flash for books that fit its 262K context, deepseek-v4-flash
# (1M) for the long ones it rejects. Long-context recall still varies: on a minor character (Justin
# in Madame Bovary) gemini-2.5-flash-lite attached young Charles's description to him and
# deepseek-v4-flash said he's mentioned once (he's in 50 passages); only xiaomi/mimo-v2.5 (~2x the
# cost, ~1 min) got him right — pass it as `model` for questions about minor details.
ASK_DOC_MODELS = _models_env("ASK_DOC_MODELS")
ASK_DOC_SYSTEM = (
    "You are given the full text of a document, followed by a question. Answer it precisely "
    "and in detail (the direct answer first, then the supporting facts and context), grounded only in the document's content, and make sure every detail you "
    "report is about the person or thing asked, not someone else. If the answer is not in the "
    "document, say so clearly rather than guessing. Answer in the same language as the question."
)


def ask_doc(doc_id: str, question: str, model: str | None = None) -> dict:
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

    # Question after the book, as in ask_all: the book prefix is then cacheable across questions.
    content = "\n\n".join(texts + [f"Question: {question}"])
    r = _first_answer([model] if model else ASK_DOC_MODELS,
                      lambda m: gemini_llm.chat(ASK_DOC_SYSTEM, content, max_tokens=1500,
                                                timeout=summarizer.SINGLE_CALL_TIMEOUT, model=m))
    if r["error"]:
        return {"error": r["answer"], "failed": r["failed"]}
    return {"doc_id": doc_id, "doc_name": doc_name, "question": question, "answer": r["answer"],
            "model": r["model"], "failed": r["failed"], "cost_usd": r["usage"]["cost_usd"]}
