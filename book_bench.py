"""Small hand-written Q&A benchmark against whatever document is currently
ingested (unlike bench.py, which downloads SQuAD 2.0). The question sets in
book_questions/ were written against robida_il_20_secolo.pdf (Albert Robida's
"Il 20. secolo") — ingest that document first, or write your own set in the
same {question, keywords, answerable} shape for a different book.

  just book-bench local en      # or: uv run python book_bench.py --engine local --lang en
  just book-bench jev it

Grading is a simple case-insensitive substring check: an answerable question
is correct if the answer contains one of its keywords and didn't abstain; an
unanswerable one is correct if the engine abstained. This is far looser than
human judgement (a wrong-but-keyword-matching answer scores as correct) but
catches the two things that matter most for a RAG demo: did retrieval find
the fact, and did the model refuse instead of guessing when it wasn't there.
"""
import argparse
import json
import time
from pathlib import Path

import app  # noqa: F401  (loads .env before the SDK clients are created)
from rag import pipeline

ROOT = Path(__file__).resolve().parent


def grade(item: dict, answer: dict) -> bool:
    text = (answer.get("answer") or "").strip().lower()
    if item["answerable"]:
        return not answer.get("abstained") and any(kw.lower() in text for kw in item["kw"])
    return bool(answer.get("abstained")) or text.startswith("not in the documents")


def run(engine: str, questions: list, top_k: int) -> None:
    correct = 0
    total_ms = 0.0
    for i, item in enumerate(questions, 1):
        hits = pipeline.retrieve(item["q"], top_k, pipeline.Trace())
        t0 = time.perf_counter()
        answer = pipeline.ENGINES[engine](item["q"], hits, pipeline.Trace())
        ms = answer.get("ms") or (time.perf_counter() - t0) * 1000
        total_ms += ms
        ok = grade(item, answer)
        correct += ok
        excerpt = (answer.get("answer") or "").replace("\n", " ")
        if len(excerpt) > 90:
            excerpt = excerpt[:90] + "…"
        print(f"{i:2d} {'OK ' if ok else 'X  '} {ms:>7.0f}ms  {excerpt}")
    print(f"-- {engine}: {correct}/{len(questions)} correct, avg {total_ms / len(questions):.0f}ms --")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="jev", choices=["claude", "local", "jev", "gemini"])
    ap.add_argument("--lang", default="en", choices=["en", "it"])
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    questions = json.loads((ROOT / "book_questions" / f"{args.lang}.json").read_text(encoding="utf-8"))
    if not pipeline.store.chunks:
        raise SystemExit("No documents ingested yet — upload robida_il_20_secolo.pdf (or your own set's book) first.")
    run(args.engine, questions, args.top_k)


if __name__ == "__main__":
    main()
