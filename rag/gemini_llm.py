"""GEMINI GENERATE — same Augment + Generate step as Claude, but routed to
Gemini Flash-Lite through OpenRouter (https://openrouter.ai) instead of a
direct Anthropic/Google call. Cheap: $0.10/$0.40 per MTok in/out.

Uses the same system prompt and context format as the Claude generator, so
the engines are directly comparable. `chat()` is the low-level call reused
by the summarizer, which needs a different system prompt.
"""
import os

import requests

from . import generator

MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
PRICE_PER_MTOK = {"input": 0.10, "output": 0.40}
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def gemini_enabled() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def chat(system_prompt: str, user_content: str, max_tokens: int = 1024, timeout: int = 60,
         frequency_penalty: float = 0.0) -> dict:
    """One request/response. Returns {answer, model, usage} or {answer, error: True}.

    max_tokens is a hard cap, not a default to leave unset: without one, a
    reduce-style call with no natural stopping point can run to the model's
    max output and degenerate into repeating itself for tens of thousands of
    tokens before hitting the ceiling (seen in practice — cost $0.035 of
    repeated garbage on one call). Pass a larger value for calls that need it.

    frequency_penalty defaults to 0 (matches temperature=0's determinism, fine
    for short citation-style answers). Long list-extraction calls should pass
    ~0.4-0.6: temperature=0 is pure greedy decoding, and once the model runs
    out of genuinely new content it has no incentive to stop rather than loop
    on repeating the same lines verbatim until max_tokens cuts it off (seen in
    practice, twice, independently — this is what actually causes it, not the
    output cap itself).
    """
    if not gemini_enabled():
        return {"answer": "Gemini is disabled: add OPENROUTER_API_KEY to .env.",
                "model": None, "usage": None, "error": True}

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={
                "model": MODEL,
                "temperature": 0,
                "frequency_penalty": frequency_penalty,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return {"answer": f"OpenRouter error: {e}", "model": None, "usage": None, "error": True}

    if "error" in data:
        return {"answer": f"OpenRouter error: {data['error'].get('message', data['error'])}",
                "model": None, "usage": None, "error": True}

    answer = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = (input_tokens * PRICE_PER_MTOK["input"] + output_tokens * PRICE_PER_MTOK["output"]) / 1e6
    return {
        "answer": answer,
        "model": data.get("model", MODEL),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": round(cost, 6)},
        "error": False,
    }


def generate(prompt: str) -> dict:
    result = chat(generator.SYSTEM_PROMPT, prompt)
    if result["error"]:
        return {"answer": result["answer"], "abstained": None, "model": None, "usage": None}
    return {
        "answer": result["answer"],
        "abstained": result["answer"].startswith(generator.ABSTAIN_PREFIX),
        "model": result["model"],
        "usage": result["usage"],
    }
