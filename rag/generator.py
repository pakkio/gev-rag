"""STEP 3-4 (query): AUGMENT + GENERATE — put retrieved chunks in the prompt, ask the LLM.

This is the "A" and "G" in RAG: the model answers from YOUR documents instead
of only its training data, and cites which chunk each fact came from.
"""
import os

import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")  # cheap: citation QA doesn't need Opus
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}  # claude-haiku-4-5, USD
ABSTAIN_PREFIX = "Not in the documents."
# Smaller/weaker models sometimes paraphrase the refusal instead of using
# ABSTAIN_PREFIX verbatim (e.g. "The text does not specify...") despite the
# system prompt instructing them to start with it exactly. Check a couple of
# common paraphrases too, so the abstained flag isn't wrong just because a
# model didn't follow the format instruction to the letter.
_ABSTAIN_MARKERS = (ABSTAIN_PREFIX.lower(), "does not specify", "does not mention", "not stated in")

SYSTEM_PROMPT = f"""You answer questions using only the numbered context passages provided.
- Cite passages inline like [1] or [2][3] right after the facts they support.
- If the context does not contain the answer, begin your reply with "{ABSTAIN_PREFIX}" and do not guess.
- Be concise."""


def llm_enabled() -> bool:
    # Off by default: the API is billed separately from a Claude subscription and
    # needs prepaid credits. Set CLAUDE_ENABLED=true in .env once you have them.
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return has_key and os.environ.get("CLAUDE_ENABLED", "false").lower() == "true"


def is_abstained(answer: str) -> bool:
    return any(m in answer.strip().lower()[:120] for m in _ABSTAIN_MARKERS)


def build_prompt(question: str, hits: list) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {h['doc_name']}, chunk {h['chunk_index']}"
        f"{', ' + h['chapter'] if h.get('chapter') else ''})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def generate(prompt: str) -> dict:
    if not llm_enabled():
        return {
            "answer": "Claude is disabled. To use it, add ANTHROPIC_API_KEY and CLAUDE_ENABLED=true to .env "
                      "(the Anthropic API needs prepaid credits). The retrieved chunks are in the trace.",
            "model": None, "usage": None,
        }

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        return {"answer": "Invalid API key. Check ANTHROPIC_API_KEY.", "model": None, "usage": None}
    except anthropic.RateLimitError:
        return {"answer": "Rate limited by the API. Try again shortly.", "model": None, "usage": None}
    except anthropic.APIStatusError as e:
        return {"answer": f"API error {e.status_code}: {e.message}", "model": None, "usage": None}
    except anthropic.APIConnectionError:
        return {"answer": "Could not reach the Anthropic API. Check your network.", "model": None, "usage": None}

    if response.stop_reason == "refusal":
        answer = "The model declined to answer this question."
    else:
        answer = "".join(b.text for b in response.content if b.type == "text")
    usage = response.usage
    cost = (usage.input_tokens * PRICE_PER_MTOK["input"] + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1e6
    return {
        "answer": answer,
        "abstained": is_abstained(answer),
        "model": response.model,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "cost_usd": round(cost, 6)},
    }
