"""STEP 3-4 (query): AUGMENT + GENERATE — put retrieved chunks in the prompt, ask the LLM.

This is the "A" and "G" in RAG: the model answers from YOUR documents instead
of only its training data, and cites which chunk each fact came from.
"""
import os

import anthropic

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.
- Cite passages inline like [1] or [2][3] right after the facts they support.
- If the context does not contain the answer, say so plainly instead of guessing.
- Be concise."""


def llm_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def build_prompt(question: str, hits: list) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {h['doc_name']}, chunk {h['chunk_index']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def generate(prompt: str) -> dict:
    if not llm_enabled():
        return {
            "answer": "LLM is off (no ANTHROPIC_API_KEY set), so only retrieval ran. "
                      "The retrieved chunks below are what would be sent to the model.",
            "model": None, "usage": None,
        }

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium"},
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
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
    return {
        "answer": answer,
        "model": response.model,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }
