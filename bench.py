"""Benchmark: Claude (generate) vs local LLM (generate) vs Jev (select) on SQuAD 2.0.

  .venv\\Scripts\\python bench.py --dry-run     # build dataset + ingest + retrieval check (no API calls)
  .venv\\Scripts\\python bench.py               # full run, writes data/bench_results.json
  then open http://localhost:8000/bench

All engines get exactly the same retrieved chunks for each question, so the
comparison isolates the final step: writing an answer vs selecting a sentence.
By default every enabled engine runs; pick a subset with e.g. --engines local,jev.
Questions run one at a time (not in parallel) so latencies don't interfere.

Grading follows SQuAD's convention:
  answerable question   -> correct if a gold answer string appears in the reply
                           (for Jev: in the selected sentence) and it didn't abstain
  unanswerable question -> correct if the engine abstained ("Not in the documents.")
"""
import argparse
import json
import random
import re
import statistics
import string
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import app  # noqa: F401  (loads .env before the SDK clients are created)
from rag import generator, gemini_llm, jev, local_llm, pipeline

ROOT = Path(__file__).resolve().parent
SQUAD_FILE = ROOT / "datasets" / "squad-dev-v2.0.json"
DOCS_DIR = ROOT / "sample_docs" / "squad"
QUESTIONS_FILE = ROOT / "datasets" / "squad_questions.json"
RESULTS_FILE = ROOT / "data" / "bench_results.json"
ARTICLES = ["Normans", "Steam_engine", "Oxygen", "Amazon_rainforest", "Black_Death"]
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def normalize(text: str) -> str:
    """SQuAD's official answer normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_gold(reply: str, golds: list) -> bool:
    reply = normalize(reply)
    return any(normalize(g) and normalize(g) in reply for g in golds)


def prepare(per_article_answerable: int, per_article_unanswerable: int, seed: int) -> list:
    if not SQUAD_FILE.exists():
        print(f"Downloading SQuAD 2.0 dev set (~4.4 MB) from {SQUAD_URL}")
        SQUAD_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, SQUAD_FILE)
    data = {a["title"]: a for a in json.loads(SQUAD_FILE.read_text(encoding="utf-8"))["data"]}
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    questions = []
    for title in ARTICLES:
        article = data[title]
        (DOCS_DIR / f"{title}.txt").write_text(
            "\n\n".join(p["context"] for p in article["paragraphs"]), encoding="utf-8")
        qas = [q for p in article["paragraphs"] for q in p["qas"]]
        answerable = [q for q in qas if not q["is_impossible"]]
        unanswerable = [q for q in qas if q["is_impossible"]]
        for q in rng.sample(answerable, per_article_answerable) + rng.sample(unanswerable, per_article_unanswerable):
            questions.append({
                "id": q["id"], "article": title, "question": q["question"],
                "answerable": not q["is_impossible"],
                "gold": sorted({a["text"] for a in q["answers"]}),
            })
    rng.shuffle(questions)
    QUESTIONS_FILE.write_text(json.dumps(questions, indent=1), encoding="utf-8")
    return questions


def ensure_ingested():
    have = {d["name"] for d in pipeline.store.documents()}
    for path in sorted(DOCS_DIR.glob("*.txt")):
        if path.name not in have:
            r = pipeline.ingest(path.name, path.read_bytes(), 800, 150)
            print(f"  ingested {path.name}: {r['trace'][1]['count']} chunks")


def percentile(values: list, p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def summarize(rows: list, engines: list) -> dict:
    summary = {}
    for e in engines:
        done = [r for r in rows if e in r]
        if not done:
            continue
        ms = [r[e]["ms"] for r in done]
        ans = [r for r in done if r["answerable"]]
        unans = [r for r in done if not r["answerable"]]
        tokens_in = sum((r[e].get("usage") or {}).get("input_tokens") or 0 for r in done)
        tokens_out = sum((r[e].get("usage") or {}).get("output_tokens") or 0 for r in done)
        summary[e] = {
            "n": len(done),
            "median_ms": round(statistics.median(ms), 1),
            "p95_ms": round(percentile(ms, 95), 1),
            "mean_ms": round(statistics.mean(ms), 1),
            "accuracy": round(sum(r[e]["correct"] for r in done) / len(done), 4),
            "answerable_accuracy": round(sum(r[e]["correct"] for r in ans) / len(ans), 4) if ans else None,
            "abstain_accuracy": round(sum(r[e]["correct"] for r in unans) / len(unans), 4) if unans else None,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": round(sum((r[e].get("usage") or {}).get("cost_usd") or 0 for r in done), 4)
                        if e in ("claude", "local") else None,
        }
    retrievable = [r for r in rows if r["answerable"]]
    summary["retrieval_hit_rate"] = (round(sum(r["retrieval_hit"] for r in retrievable) / len(retrievable), 4)
                                     if retrievable else None)
    return summary


def run_engine(engine: str, question: str, hits: list) -> dict:
    return pipeline.ENGINES[engine](question, hits, pipeline.Trace())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answerable", type=int, default=8, help="answerable questions per article")
    ap.add_argument("--unanswerable", type=int, default=4, help="unanswerable questions per article")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--engines", default="", help="comma list of claude,local,jev (default: all enabled)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="no API calls: dataset, ingest and retrieval only")
    args = ap.parse_args()
    enabled = {"claude": generator.llm_enabled(), "gemini": gemini_llm.gemini_enabled(),
               "local": local_llm.local_status()["enabled"], "jev": jev.jev_enabled()}
    engines = [e for e in args.engines.split(",") if e] or [e for e, on in enabled.items() if on]
    if not engines and not args.dry_run:
        raise SystemExit("No engine is enabled. Start Ollama, add TYPESAFE_API_KEY, or enable Claude in .env.")

    questions = prepare(args.answerable, args.unanswerable, args.seed)
    print(f"Dataset: {len(questions)} questions from {len(ARTICLES)} SQuAD 2.0 articles")
    ensure_ingested()

    if not args.dry_run:
        # Warm-up doubles as a preflight: the first call pays for connection setup
        # (or loading the local model into VRAM), so it isn't counted; if an engine
        # fails here (missing key, no credits, Ollama down) stop before wasting a run.
        warm_hits = pipeline.retrieve(questions[0]["question"], args.top_k, pipeline.Trace())
        for e in engines:
            r = run_engine(e, questions[0]["question"], warm_hits)
            if r.get("usage") is None:
                raise SystemExit(f"{e} is not working: {r['answer']} Fix it, or leave it out with --engines.")
            print(f"  warm-up {e}: ok ({r['ms']:.0f} ms)")

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "SQuAD 2.0 dev (CC BY-SA 4.0)", "articles": ARTICLES,
        "top_k": args.top_k, "chunk_size": 800, "overlap": 150,
        "claude_model": generator.MODEL, "local_model": local_llm.MODEL,
        "jev_threshold": jev.ANSWERABLE_THRESHOLD, "engines": [] if args.dry_run else engines,
        "dry_run": args.dry_run, "total": len(questions),
    }
    rows = []
    for i, q in enumerate(questions, 1):
        t = time.perf_counter()
        hits = pipeline.retrieve(q["question"], args.top_k, pipeline.Trace())
        row = {**q, "retrieval_ms": round((time.perf_counter() - t) * 1000, 1),
               "retrieval_hit": q["answerable"] and contains_gold(" ".join(h["text"] for h in hits), q["gold"])}
        if not args.dry_run:
            for e in engines:
                r = run_engine(e, q["question"], hits)
                if q["answerable"]:
                    r["correct"] = not r["abstained"] and contains_gold(r["answer"], q["gold"])
                else:
                    r["correct"] = bool(r["abstained"])
                row[e] = r
        rows.append(row)
        marks = "  ".join(f"{e}:{'✓' if row[e]['correct'] else '✗'} {row[e]['ms']:>7.0f}ms" for e in engines if e in row)
        print(f"[{i:>2}/{len(questions)}] {'A' if q['answerable'] else 'U'}  {marks}  {q['question'][:60]}")
        RESULTS_FILE.parent.mkdir(exist_ok=True)
        RESULTS_FILE.write_text(json.dumps({"meta": meta, "summary": summarize(rows, engines), "rows": rows},
                                           indent=1), encoding="utf-8")

    summary = summarize(rows, engines)
    print(f"\nRetrieval hit rate (answer text in top-{args.top_k} chunks): {summary['retrieval_hit_rate']:.0%}")
    for e in engines:
        if e in summary:
            s = summary[e]
            print(f"{e:>7}: median {s['median_ms']:.0f} ms · p95 {s['p95_ms']:.0f} ms · accuracy {s['accuracy']:.0%} "
                  f"(answerable {s['answerable_accuracy']:.0%}, abstain {s['abstain_accuracy']:.0%})"
                  + (f" · ${s['cost_usd']:.3f}" if s["cost_usd"] is not None else ""))
    print(f"\nResults: {RESULTS_FILE}  ·  view at http://localhost:8000/bench")


if __name__ == "__main__":
    main()
