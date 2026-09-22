# Tiny RAG + Jev

A small, readable Retrieval-Augmented Generation (RAG) web app. You upload documents, ask
questions, and get answers with citations. The right-hand panel shows each pipeline step: its
timing, inputs and outputs.

It compares three ways to produce the final answer:

| Engine | What it does | Runs on | Cost |
|---|---|---|---|
| **Local LLM** (default: `qwen3:14b`) | Writes an answer with `[n]` citations | Your GPU, via [Ollama](https://ollama.com) | Free |
| **Jev** ([TypeSafe](https://docs.typesafe.ai)) | *Selects* the sentence that answers the question, or says "not in the documents" | TypeSafe API | Per token |
| **Claude** (optional, off by default) | Writes an answer with `[n]` citations | Anthropic API | Per token, prepaid credits |

**Stack:** FastAPI · fastembed (local ONNX embeddings) · a numpy vector store · Ollama · TypeSafe SDK · Anthropic SDK · vanilla HTML/JS. Requires **Python 3.10+**.

## The RAG workflow

RAG has two pipelines. **Ingest** runs once per document. **Query** runs on every question.

```
 INGEST (per document)
 ┌────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐
 │  Load  │──▶│  Chunk  │──▶│  Embed   │──▶│    Store     │
 │ pdf/md │   │ 800 chr │   │ 384-dim  │   │ vectors.npy  │
 │ → text │   │ +overlap│   │ vectors  │   │ chunks.json  │
 └────────┘   └─────────┘   └──────────┘   └──────┬───────┘
                                                  │ similarity search
 QUERY (per question)                             ▼
 "question?" ─▶ Embed ─▶ Retrieve top-k ─┬─▶ Augment ─▶ Generate   Local LLM or Claude writes
                                         │                          an answer citing [1]..[k]
                                         └─▶ Select                 Jev picks a sentence
```

| Step | File | What happens | Knob to experiment with |
|---|---|---|---|
| Load | `rag/loader.py` | PDF/TXT/MD → plain text | – |
| Chunk | `rag/chunker.py` | Split into overlapping pieces, cut at paragraph or sentence breaks | chunk size, overlap |
| Embed | `rag/embedder.py` | Text → normalized 384-dim vector (`bge-small-en-v1.5`, CPU) | embedding model |
| Store | `rag/store.py` | Append vectors + metadata, persist to `data/` | – |
| Retrieve | `rag/store.py` | `scores = vectors @ query_vec`, take top-k | top-k |
| Augment | `rag/generator.py` | Number the chunks `[1]..[k]` and wrap them in `<context>` | system prompt |
| Generate | `rag/local_llm.py`, `rag/generator.py` | The LLM answers only from that context and cites `[n]` | model |
| Select | `rag/jev.py` | Code splits chunks into sentences; in **one** request Jev judges "is the answer here?" (Noul) and "which sentence?" (Choice) | confidence threshold |

`rag/pipeline.py` connects the steps and records the trace. `app.py` is the HTTP API.

## Setup

```bash
git clone https://github.com/HyeranPark99/tiny-rag-jev.git
cd tiny-rag-jev
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Local LLM:** install [Ollama](https://ollama.com), then pull the model (~9 GB; fits a 16 GB GPU):

```bash
ollama pull qwen3:14b
```

On a smaller GPU, use `qwen3:8b` (~5 GB) and set `OLLAMA_MODEL=qwen3:8b` in `.env`.

**Keys:** create `.env` in the project folder:

```
# Jev (from https://console.typesafe.ai)
TYPESAFE_API_KEY=...

# Optional: a different local model (default qwen3:14b)
# OLLAMA_MODEL=qwen3:8b

# Optional: Claude (needs both lines)
# ANTHROPIC_API_KEY=sk-ant-...
# CLAUDE_ENABLED=true
```

Claude is off by default. The Anthropic API is billed separately from a Claude.ai subscription
and needs prepaid credits (https://console.anthropic.com → Billing).

## Run it

```bash
.venv\Scripts\python app.py
```

Open http://localhost:8000.

- The header badges show which engines are on. Hover an "off" badge to see why.
- Upload a `.txt` / `.md` / `.pdf`, or use the SQuAD articles the benchmark loads (see below).
- Under the chat box, toggle any combination of **Local LLM**, **Jev** and **Claude**. With two or more selected, answers appear side by side with timing bars.

The first local answer after startup includes loading the model into VRAM (~20–30 s). After that it stays loaded for 30 minutes.

## Benchmark: generate vs select on SQuAD 2.0

[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) is a public set of Wikipedia questions with gold answers (CC BY-SA 4.0). It includes *unanswerable* questions written to sound answerable, which tests whether an engine will say "not in the documents" instead of guessing.

```bash
.venv\Scripts\python bench.py --dry-run
```
Downloads SQuAD, ingests 5 articles and checks retrieval, with no API calls.

```bash
.venv\Scripts\python bench.py
```
Runs 60 questions (40 answerable, 20 unanswerable) through every enabled engine. Then open http://localhost:8000/bench.

Every engine gets the same retrieved chunks, so only the answer step is compared. To pick engines, use `--engines local,jev`. For a different question sample, use `--seed 11`.

### Results (qwen3:14b on an RTX 5070 Ti vs Jev, 60 questions)

| | Local LLM (qwen3:14b) | Jev |
|---|---|---|
| Median answer time | 530 ms | **255 ms (2.1× faster)** |
| Slowest 5% (p95) | 1,208 ms | **354 ms (3.4× faster)** |
| Answerable questions correct | 82% | **88%** |
| Unanswerable, correctly abstained | 45% | 45% |
| Overall | 70% | 73% |
| Cost | $0 | ~102k input + 48k output tokens |

The answer text was in the top-4 retrieved chunks for 95% of answerable questions, which is the ceiling for both engines.

**What we learned:**
- **Jev was faster on 59 of 60 questions**, network round trip included, and its timing barely varies. A local LLM slows down as answers get longer.
- **The local LLM sometimes answers from memory, not the documents.** It said "Constantinople is in Turkey" and "Edgar married Margaret", neither of which is in the context. Jev can't do this: it can only pick a sentence that's actually in your documents.
- **Jev's weak spot is near-miss sentences.** For example, it picked a sentence about Arthur Woolf's invention for "Who patented the Woolf cooling cylinder?" (unanswerable).
- **Jev's confidence threshold is a free dial.** Re-scoring the saved results without new API calls, a cutoff of 0.8 instead of 0.5 raised correct abstentions from 45% to 70% and cost 6 points on answerable questions (78% overall). That cutoff was chosen on these same 60 questions, so validate it on a fresh `--seed` before relying on it.

**Caveats:** Jev returns the evidence sentence, not a rewritten answer, and a sentence containing the gold answer counts as correct, which favors Jev slightly. 60 questions shows the 2× speed gap reliably, but not the 3-point accuracy gap.

## Experiments that teach you RAG

1. **Chunk size.** Re-upload the same file at 200 and at 2000 chars. Ask the same question and compare the retrieval scores and how focused each hit is.
2. **Top-k.** Set it to 1, then 8. Too low and the answer can be missed. Too high and irrelevant context is added (more tokens, more noise).
3. **Unanswerable questions.** Ask something the docs don't cover and compare how each engine handles it.
4. **Paraphrases.** Ask "Can I work from home?" when the doc says "remote work". Embeddings match meaning, not keywords.
5. **Jev's threshold.** Change `ANSWERABLE_THRESHOLD` in `rag/jev.py` and re-run the benchmark.

## Where to go next

- **Hybrid search:** combine BM25 keyword scores with vector scores (helps with names, codes and IDs).
- **Reranking:** retrieve 20 chunks, then have Jev score each one for relevance and keep the best 4.
- **Citation checks:** have Jev verify each `[n]` in a generated answer against its chunk.
- **Real vector DB:** replace `store.py` with Chroma, FAISS or pgvector once you have 100k+ chunks.
- **Streaming answers:** stream the local LLM's tokens to the UI so the answer appears as it's written.
