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
MAX_CANDIDATES = 200  # Choice allows up to 255 options
ANSWERABLE_THRESHOLD = 0.5

# Split after . ! ? when the next sentence starts like a sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_client = None


def jev_enabled() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def _get_client() -> TypeSafeClient:
    global _client
    if _client is None:
        _client = TypeSafeClient()
    return _client


def candidate_sentences(hits: list) -> list:
    seen, out = set(), []
    for h in hits:
        for s in _SENTENCE_BREAK.split(h["text"].replace("\n", " ")):
            s = s.strip()
            if len(s) >= 20 and s not in seen:
                seen.add(s)
                out.append(s)
    return out[:MAX_CANDIDATES]


def fast_answer(question: str, hits: list) -> dict:
    candidates = candidate_sentences(hits)
    base = {"candidates": len(candidates), "model": None, "usage": None,
            "answerable": None, "confidence": None, "abstained": None}
    if not jev_enabled():
        return {**base, "answer": "Jev is off (no TYPESAFE_API_KEY set)."}

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
    abstained = answerable < ANSWERABLE_THRESHOLD or pick.choice == NONE
    return {
        **base,
        "answer": "Not in the documents." if abstained else pick.choice,
        "abstained": abstained,
        "answerable": round(answerable, 4),
        "picked": pick.choice,
        "confidence": round(pick.confidence, 4),
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }
