"""JEV FAST ANSWER (alternative to Augment + Generate) — select, don't generate.

Instead of asking an LLM to write an answer, code splits the retrieved chunks
into candidate sentences and Jev (TypeSafe's System One model) makes two
judgments in ONE request, answered in parallel:

  answerable  (Noul)    Do the passages contain the answer at all?
  sentence    (Choice)  Which candidate sentence answers the question?

The answer is copied verbatim from your documents, so it cannot be invented.
The trade-off: you get the evidence sentence, not a fluent rewritten answer.
"""
import os
import re

from typesafe_sdk import (
    Choice,
    Noul,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)

NONE = "none of these sentences"
MAX_CANDIDATES = 254  # Choice allows up to 255 options, one of which is NONE
# Thresholds swept on book_questions/ (40 questions, multilingual embeddings, 24 chunks):
# answerable >= 0.7 and confidence >= 0.5 answered 20/30 right, 3 wrong, 2/10 traps; the
# previous 0.8/0.6 answered 18/30 with the same errors (it only dropped right picks).
ANSWERABLE_THRESHOLD = 0.7
# Estimate per TypeSafe's published rate (Jev 1.13, checked 2026-09-21):
# $0.042 per million INPUT tokens, output free. Not a bill — verify the current
# rate and your actual usage in the TypeSafe console. Override via env.
JEV_PRICE_PER_MTOK = float(os.environ.get("JEV_PRICE_PER_MTOK", "0.042"))
# When the Noul says "answerable" but no sentence is a confident match (< 0.5),
# the pick is a near-miss ("Justin?" -> a scene about Justin, not who he is).
# Prefer an honest abstain over a weak sentence.
MIN_PICK_CONFIDENCE = 0.5
# The answer is the picked sentence plus its neighbours in the same passage, so it reads
# as an excerpt with context rather than one isolated line. Still verbatim text.
CONTEXT_BEFORE, CONTEXT_AFTER = 1, 2

# Split after . ! ? when the next sentence starts like a sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# Book cards are markdown: split bullet/header lines ("* **Anna Karenina:** …")
# before collapsing newlines, so each bullet is its own candidate sentence.
_MARKDOWN_BREAK = re.compile(r"\n(?=[*#\-])")
_client = None


# External hard-off knob, e.g. to save tokens while keeping the key: JEV_ENABLED=0
# in .env (or false/no/off) turns Jev off without touching code or the key.
_JEV_KNOB = os.environ.get("JEV_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def jev_enabled() -> bool:
    return _JEV_KNOB and bool(os.environ.get("TYPESAFE_API_KEY"))


def jev_reason() -> str | None:
    """Why Jev is off, for the UI badge; None when it's on."""
    if not _JEV_KNOB:
        return "off: JEV_ENABLED=0 in .env"
    if not os.environ.get("TYPESAFE_API_KEY"):
        return "off (no key)"
    return None


def _get_client() -> TypeSafeClient:
    global _client
    if _client is None:
        _client = TypeSafeClient()
    return _client


def _blocks(hits: list) -> list[list[str]]:
    """Each hit split into blocks of sentences, then sentence punctuation after collapsing
    newlines. Book cards (is_card) break at every line: their lines are bullets or
    "Name: description" entries, with or without markdown. Prose chunks break only before
    markdown bullets/headers, since PDF text wraps mid-sentence. A block is the span an
    answer's context may extend over, so one character's line never pulls in the next."""
    out = []
    for h in hits:
        text = h["text"].replace("\r", "")
        for block in (text.split("\n") if h.get("is_card") else _MARKDOWN_BREAK.split(text)):
            sentences = [s.strip() for s in _SENTENCE_BREAK.split(block.replace("\n", " ")) if s.strip()]
            if sentences:
                out.append(sentences)
    return out


def candidate_sentences(hits: list) -> list:
    seen, out = set(), []
    for block in _blocks(hits):
        for s in block:
            if len(s) >= 20 and s not in seen:
                seen.add(s)
                out.append(s)
    return out[:MAX_CANDIDATES]


def with_context(sentence: str, hits: list) -> str:
    """The picked sentence with CONTEXT_BEFORE/AFTER neighbours from the first block holding it."""
    for block in _blocks(hits):
        if sentence in block:
            i = block.index(sentence)
            return " ".join(block[max(0, i - CONTEXT_BEFORE):i + 1 + CONTEXT_AFTER])
    return sentence


def fast_answer(question: str, hits: list) -> dict:
    candidates = candidate_sentences(hits)
    base = {"candidates": len(candidates), "model": None, "usage": None,
            "answerable": None, "confidence": None, "abstained": None}
    if not jev_enabled():
        return {**base, "answer": f"Jev is off ({jev_reason()})."}

    criteria = {s: None for s in candidates}
    criteria[NONE] = "None of these sentences answers the question."
    try:
        response = _get_client().system_one(
            state={"question": question, "passages": [h["text"] for h in hits]},
            questions={
                "answerable": Noul(
                    instructions="Do `passages` contain the specific information needed to answer `question`?"
                ),
                "sentence": Choice(
                    instructions="Which sentence most directly answers `question`, based on `passages`?",
                    criteria=criteria,
                ),
            },
        )
    except TypeSafeAuthenticationError:
        return {**base, "answer": "Invalid TypeSafe API key. Check TYPESAFE_API_KEY."}
    except TypeSafeRateLimitError:
        return {**base, "answer": "Rate limited by TypeSafe. Try again shortly."}
    except TypeSafeError as e:
        return {**base, "answer": f"TypeSafe error: {e}"}

    answerable = response.nouls["answerable"].noul
    pick = response.choices["sentence"]
    abstained = (answerable < ANSWERABLE_THRESHOLD or pick.choice == NONE
                 or pick.confidence < MIN_PICK_CONFIDENCE)
    return {
        **base,
        "answer": "Not in the documents." if abstained else with_context(pick.choice, hits),
        "abstained": abstained,
        "answerable": round(answerable, 4),
        "picked": pick.choice,
        "confidence": round(pick.confidence, 4),
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                  "cost_usd": round(response.usage.input_tokens * JEV_PRICE_PER_MTOK / 1e6, 6)},
    }


def verify(claims: list[tuple[str, list[str]]]) -> dict:
    """For each (claim, cited passages), all in one request: Jev's probability that the
    passages support the claim ("supported"), and that they state it as a fact rather than
    as a character's guess, suspicion, joke or opinion ("factual": Homais teasing that Justin
    loves Félicité supports "Homais suspects it", not "Justin loves Félicité"). Used to check
    an LLM-written answer sentence by sentence, so what Jev can't select (a description
    spread over many passages) can be written and still be checked against the text."""
    base = {"supported": None, "factual": None, "model": None, "usage": None}
    if not jev_enabled():
        return {**base, "error": f"Jev is off ({jev_reason()})."}
    state, questions = {}, {}
    for i, (claim, passages) in enumerate(claims):
        state[f"claim_{i}"], state[f"passages_{i}"] = claim, passages
        questions[f"supported_{i}"] = Noul(
            instructions=f"Is every detail of `claim_{i}` stated or directly implied by `passages_{i}`, "
                         f"about the same person or thing?")
        questions[f"factual_{i}"] = Noul(
            instructions=f"Does `claim_{i}` state things the way `passages_{i}` do? Answer no if it states "
                         f"as fact what the passages give only as a character's guess, suspicion, joke, "
                         f"rumour or opinion. A claim that itself reports that someone says, predicts or "
                         f"believes something counts as stated the way the passages do, if that report "
                         f"is accurate.")
    try:
        response = _get_client().system_one(state=state, questions=questions)
    except TypeSafeError as e:  # base class of the auth and rate-limit errors too
        return {**base, "error": f"TypeSafe error: {e}"}
    return {"supported": [round(response.nouls[f"supported_{i}"].noul, 4) for i in range(len(claims))],
            "factual": [round(response.nouls[f"factual_{i}"].noul, 4) for i in range(len(claims))],
            "model": response.model, "error": None,
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                      "cost_usd": round(response.usage.input_tokens * JEV_PRICE_PER_MTOK / 1e6, 6)}}
