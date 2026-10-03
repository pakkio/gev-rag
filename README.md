# Tiny RAG + Jev

A small, readable Retrieval-Augmented Generation (RAG) web app. You upload documents, ask
questions, and get answers with citations. The right-hand panel shows each pipeline step: its
timing, inputs and outputs.

It compares four ways to produce the final answer:

| Engine | What it does | Runs on | Cost |
|---|---|---|---|
| **Local LLM** (default: `qwen3:14b`) | Writes an answer with `[n]` citations | Your GPU, via [Ollama](https://ollama.com) | Free |
| **Jev** ([TypeSafe](https://docs.typesafe.ai)) | *Selects* the sentence that answers the question, or says "not in the documents" | TypeSafe API | Per token |
| **Flash** (`OPENROUTER_MODEL`: default `google/gemini-2.5-flash-lite`, here `deepseek-v4-flash-0731`; via [OpenRouter](https://openrouter.ai)) | Writes an answer with `[n]` citations | OpenRouter API | Per token |
| **Claude** (optional, off by default) | Writes an answer with `[n]` citations | Anthropic API | Per token, prepaid credits |

**Stack:** FastAPI · fastembed (local ONNX embeddings, on the GPU via CUDA when available) · a numpy vector store · Ollama · TypeSafe SDK · Anthropic SDK · OpenRouter (Flash) · vanilla HTML/JS. Requires **Python 3.10+**. A [`justfile`](justfile) wraps the common commands (see below).

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
| Load | `rag/loader.py` | PDF/TXT/MD → plain text; drops the second copy of pages some PDFs draw twice | – |
| Chunk | `rag/chunker.py` | Split into overlapping pieces, cut at paragraph or sentence breaks | chunk size, overlap |
| Embed | `rag/embedder.py` | Text → normalized 768-dim vector (`paraphrase-multilingual-mpnet-base-v2`, multilingual; GPU if CUDA loads, else CPU) | embedding model |
| Store | `rag/store.py` | Append vectors + metadata, persist to `data/` | – |
| Retrieve | `rag/store.py` | `scores = vectors @ query_vec`, take top-k | top-k |
| Augment | `rag/generator.py` | Number the chunks `[1]..[k]` and wrap them in `<context>` | system prompt |
| Generate | `rag/local_llm.py`, `rag/gemini_llm.py`, `rag/generator.py` | The LLM answers only from that context and cites `[n]` | model |
| Select | `rag/jev.py` | Code splits chunks into sentences; in **one** request Jev judges "is the answer here?" (Noul) and "which sentence?" (Choice) | confidence threshold |

`rag/pipeline.py` connects the steps and records the trace. `app.py` is the HTTP API.
Each retrieved source also names the **chapter** its chunk belongs to (see below).

### Beyond top-k: book cards, chapters and library-wide questions

`/api/ask` only ever sees the top-k retrieved chunks, so "what is this book about?" or
"which novels end with the hero's death?" has no single matching chunk. For those, each
book gets a **card**, built once and cached in `data/summaries.json` (`rag/summarizer.py`,
`rag/summary_store.py`):

- **Card** — an Italian reference card (work, setting, main characters and their fate, plot
  *including the ending*, themes), written by the Flash model (`OPENROUTER_MODEL`) from the whole book: one
  call when it fits the 1M-token context, otherwise map-reduce with length-capped
  intermediate levels. About 5 cents for an average book.
- **Chapters** (`rag/chapters.py`) — pattern matching finds lines that look like headings
  ("CAPITOLO XII", "CANTO V", "Cap. 3", "III.") with their exact chunk; one cheap LLM call then
  names the false positives (table of contents, notes) to drop. Bare numerals get their
  part as prefix ("PARTE PRIMA · III"). Retrieval uses this to label every chunk.
- **Library-wide questions** (`POST /api/ask-all`) — all cards (~30K tokens) plus the question,
  in one call via OpenRouter: `inclusionai/ling-3.0-flash` (~0.05 cents a question), falling
  back to `deepseek/deepseek-v4-flash` if it errors or answers empty. The cards come first so
  repeat questions hit the prompt cache. `engine: "local"` uses Ollama instead — which needs a
  ~40K context that doesn't fit an 8 GB GPU, so there it runs on the CPU and takes minutes.
- **Ask one whole book** (`POST /api/documents/{doc_id}/ask`) — one question against the
  book's full text, not cached: ling-3.0-flash when the book fits its 262K context,
  deepseek-v4-flash (1M) for long novels, ~3 cents a question. For minor details pass
  `model: "xiaomi/mimo-v2.5"` (~6 cents, ~1 min): it was the only one to get a minor
  character right in testing.

Model choice comes from two benchmarks of cheap OpenRouter models (13 library-wide questions
including "not in the library" traps, 5 whole-book questions); the response says which model
answered and which ones failed before it.

## Setup

Uses [`just`](https://github.com/casey/just) + [`uv`](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:pakkio/gev-rag.git
cd jev-rag
just setup
```

Without `just`/`uv`, the equivalent is a plain venv:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip install -r requirements.txt
```

**Local LLM:** install [Ollama](https://ollama.com), then pull the model (~9 GB; fits a 16 GB GPU):

```bash
just pull-model          # or: ollama pull qwen3:14b
```

On a smaller GPU, use `just pull-model qwen3:8b` (~5 GB) and set `OLLAMA_MODEL=qwen3:8b` in `.env`.

**Keys:** create `.env` in the project folder:

```
# Jev (from https://console.typesafe.ai)
TYPESAFE_API_KEY=...
# Optional hard-off switch for Jev even with a key set (0/false/no/off) —
# e.g. save tokens while keeping the key. Read once at startup, not per query.
# JEV_ENABLED=1

# OpenRouter (from https://openrouter.ai/keys) — the Flash engine, book cards,
# chapters, whole-book Q&A and library-wide questions
OPENROUTER_API_KEY=...
# Optional: model for the Flash engine, cards and chapters (default google/gemini-2.5-flash-lite)
# OPENROUTER_MODEL=...
# Optional: models tried in order for library-wide and whole-book questions
# (default for both: inclusionai/ling-3.0-flash,deepseek/deepseek-v4-flash)
# ASK_ALL_ENGINE=cloud
# ASK_ALL_MODELS=inclusionai/ling-3.0-flash,deepseek/deepseek-v4-flash
# ASK_DOC_MODELS=deepseek/deepseek-v4-flash

# Optional: a different local model (default qwen3:14b)
# OLLAMA_MODEL=qwen3:8b

# Optional: Claude (needs both lines)
# ANTHROPIC_API_KEY=sk-ant-...
# CLAUDE_ENABLED=true
```

Claude is off by default. The Anthropic API is billed separately from a Claude.ai subscription
and needs prepaid credits (https://console.anthropic.com → Billing).

**GPU embeddings:** `requirements.txt` installs `fastembed-gpu` with the CUDA/cuDNN runtime
from pip, pinned to the CUDA 13.0 series: on WSL2 the 13.4 runtime segfaults in
`cudaSetDevice` against a driver reporting CUDA 13.3. Without a usable GPU, embedding falls
back to the CPU (~13 chunks/s instead of ~1000).

## Run it

```bash
just run          # or: .venv/bin/python app.py
```

Open http://localhost:8000.

- The header badges show which engines are on. Hover an "off" badge to see why.
- Upload a `.txt` / `.md` / `.pdf`, or use the SQuAD articles the benchmark loads (see below).
- Under the chat box, toggle any combination of **Local LLM**, **Jev**, **Flash** and **Claude**. With two or more selected, answers appear side by side with timing bars.

The first local answer after startup includes loading the model into VRAM (~20–30 s). After that it stays loaded for 30 minutes.

### CLI (justfile)

`just --list` shows all recipes. The main ones, against a running `just run` server:

| Recipe | What it does |
|---|---|
| `just setup` | Create/refresh the venv and install deps (`uv`) |
| `just pull-model [model]` | `ollama pull` the local model (default `qwen3:14b`) |
| `just run` | Start the FastAPI app |
| `just status` | Engine status + ingested documents |
| `just list-docs [regex]` | List ingested documents, filtered by a regex on the file name (title/author) |
| `just ingest <files...>` | Upload one or more documents |
| `just ask "<question>" [engines]` | Ask a question (`engines` default: `jev,local`) |
| `just ask-all "<question>"` | Question about the whole library, answered from the book cards |
| `just ask-doc <doc_id> "<question>"` | Question against one whole book (not top-k retrieval) |
| `just summarize-all [force]` | Build the card + chapters for every book lacking one (`force=1`: all) |
| `just summarize <doc_id>` | Build (or rebuild) one book's card + chapters |
| `just doc-summary <doc_id>` | Read a book's card and chapters |
| `just delete-doc <doc_id>` | Remove an ingested document |
| `just bench-dry` | Dry-run benchmark (no API calls) |
| `just book-bench [engine] [lang]` | 20-question benchmark against the ingested book (`book_questions/`, `en`/`it`) |

## Benchmark: generate vs select on SQuAD 2.0

[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) is a public set of Wikipedia questions with gold answers (CC BY-SA 4.0). It includes *unanswerable* questions written to sound answerable, which tests whether an engine will say "not in the documents" instead of guessing.

```bash
just bench-dry          # or: uv run python bench.py --dry-run
```
Downloads SQuAD, ingests 5 articles and checks retrieval, with no API calls.

```bash
uv run python bench.py
```
Runs 60 questions (40 answerable, 20 unanswerable) through every enabled engine (Claude, local,
Jev, Flash — all auto-detected from the same env the app uses). Then open
http://localhost:8000/bench.

Every engine gets the same retrieved chunks, so only the answer step is compared. To pick engines, use `--engines local,jev` (or include `gemini`, the CLI id of the Flash engine). For a different question sample, use `--seed 11`.

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
- **Jev's pick confidence is a second dial.** When the "answerable" judge says yes but no sentence is a confident match, the pick is a near-miss. `MIN_PICK_CONFIDENCE` (default 0.6, `rag/jev.py`) abstains instead of forcing the weak sentence.

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
