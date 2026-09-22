"""LOCAL GENERATE — same Augment + Generate step as Claude, but with an open
model running on your own GPU through Ollama (https://ollama.com).

Free and private: nothing leaves your machine. Uses the exact same system
prompt and context format as the Claude generator, so the two are comparable.
"""
import os

import ollama

from . import generator

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
_client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))


def local_status() -> dict:
    try:
        names = {m.model for m in _client.list().models}
    except (ollama.RequestError, ollama.ResponseError, ConnectionError):
        return {"enabled": False, "reason": "Ollama is not running"}
    if MODEL not in names:
        return {"enabled": False, "reason": f"model {MODEL} not pulled (ollama pull {MODEL})"}
    return {"enabled": True, "reason": None}


def generate(prompt: str) -> dict:
    status = local_status()
    if not status["enabled"]:
        return {"answer": f"Local LLM is off: {status['reason']}.", "abstained": None, "model": None, "usage": None}
    try:
        response = _client.chat(
            model=MODEL,
            messages=[{"role": "system", "content": generator.SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            think=False,                     # Qwen3 reasons by default; off = fast, direct answers
            options={"temperature": 0, "num_ctx": 8192},
            keep_alive="30m",                # keep the model loaded in VRAM between questions
        )
    except (ollama.RequestError, ollama.ResponseError) as e:
        return {"answer": f"Ollama error: {e}", "abstained": None, "model": None, "usage": None}

    answer = response.message.content.strip()
    return {
        "answer": answer,
        "abstained": answer.startswith(generator.ABSTAIN_PREFIX),
        "model": MODEL,
        "usage": {
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
            "cost_usd": 0.0,
            "load_ms": round((response.load_duration or 0) / 1e6, 1),  # >0 means a cold start
        },
    }
