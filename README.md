# Tiny RAG

A small, readable Retrieval-Augmented Generation (RAG) web app. You upload documents, ask
questions, and the app answers with citations. The right-hand panel shows each pipeline step:
its timing, inputs and outputs.

**Stack:** FastAPI · fastembed (local ONNX embeddings, no GPU or PyTorch) · a numpy vector store · Claude for generation · Jev (TypeSafe) for fast answer selection · vanilla HTML/JS. Requires Python 3.10+.

## The RAG workflow

RAG has two pipelines. **Ingest** runs once per document. **Query** runs on every question.

```
 INGEST (offline, per document)
 ┌────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐
 │  Load  │──▶│  Chunk  │──▶│  Embed   │──▶│    Store     │
 │ pdf/md │   │ 800 chr │   │ 384-dim  │   │ vectors.npy  │
 │ → text │   │ +overlap│   │ vectors  │   │ chunks.json  │
 └────────┘   └─────────┘   └──────────┘   └──────┬───────┘
                                                  │
 QUERY (online, per question)                     │ similarity search
 ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──▼───────┐
 │ Generate │◀──│ Augment  │◀──│ Retrieve  │◀──│  Embed   │◀── "question?"
 │  Claude  │   │ prompt = │   │ top-k by  │   │  query   │
 │ + cites  │   │ context+Q│   │ cosine sim│   │          │
 └──────────┘   └──────────┘   └───────────┘   └──────────┘
```

| Step | File | What happens | Knob to experiment with |
|---|---|---|---|
| Load | `rag/loader.py` | PDF/TXT/MD → plain text | – |
| Chunk | `rag/chunker.py` | Split into overlapping pieces, cut at paragraph or sentence breaks | chunk size, overlap |
| Embed | `rag/embedder.py` | Text → normalized 384-dim vector (`bge-small-en-v1.5`) | embedding model |
| Store | `rag/store.py` | Append vectors + metadata, persist to `data/` | – |
| Retrieve | `rag/store.py` | `scores = vectors @ query_vec`, take top-k | top-k |
| Augment | `rag/generator.py` | Number the chunks `[1]..[k]` and wrap them in `<context>` | system prompt |
| Generate | `rag/generator.py` | Claude answers only from that context and cites `[n]` | model, effort |

`rag/pipeline.py` connects the steps and records the trace. `app.py` is the HTTP API.

## Run it

```bash
cd rag-app
.venv\Scripts\python app.py
```

Open http://localhost:8000 and drop in `sample_docs/acme_handbook.md`.

The first upload downloads the embedding model (~70 MB, cached in `models/`).

**Turning on the LLM:** without a key the app runs in *retrieval-only* mode, which is still
useful for learning: you see which chunks would be sent to the model. To get real answers, create
`rag-app/.env` containing:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then restart the server. You can get a key at https://console.anthropic.com.

Fresh setup on another machine:

```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```

## Experiment: Claude (generate) vs Jev (select)

[Jev](https://docs.typesafe.ai) is TypeSafe's System One model. It returns typed judgments (yes/no probabilities, picks from a list), not text. `rag/jev.py` offers a second way to finish the query pipeline:

```
Retrieve ─┬─▶ Augment ─▶ Generate      Claude writes an answer with [n] citations
          └─▶ Select                   code splits chunks into sentences; in ONE request Jev judges
                                        "is the answer here?" (Noul) + "which sentence?" (Choice)
```

Jev's answer is copied verbatim from your documents, so it can't be invented. In the chat, switch between **Claude**, **Jev** and **Compare both** to see the answers side by side with timing bars.

**Benchmark on public data:** SQuAD 2.0 (Wikipedia questions with gold answers, including unanswerable ones):

```bash
.venv\Scripts\python bench.py --dry-run
```
The dry run downloads SQuAD, ingests 5 articles and checks retrieval, with no API calls.

```bash
.venv\Scripts\python bench.py
```
The full run sends 60 questions to both engines. Then open http://localhost:8000/bench.

Needs both keys in `.env`: `ANTHROPIC_API_KEY=...` and `TYPESAFE_API_KEY=...` (from https://console.typesafe.ai).

## Experiments that teach you RAG

1. **Chunk size.** Re-upload the same file at 200 and at 2000 chars. Ask the same question and compare the retrieval scores and how focused each hit is.
2. **Top-k.** Set it to 1, then 8. Too low and the answer can be missed. Too high and irrelevant context is added (more tokens, more noise).
3. **Unanswerable questions.** Ask something the doc doesn't cover ("What's the dress code?"). Check the scores: they drop. The prompt tells the model to say it doesn't know.
4. **Paraphrases.** Ask "Can I work from home?" The doc says "remote work". Embeddings match meaning, not keywords.

## Where to go next

- **Hybrid search:** combine BM25 keyword scores with vector scores (helps with names, codes and IDs).
- **Reranking:** retrieve 20 chunks, then rerank them with a cross-encoder and keep the best 4.
- **Real vector DB:** replace `store.py` with Chroma, FAISS or pgvector once you have 100k+ chunks.
- **Streaming answers:** stream tokens to the UI with `client.messages.stream(...)`.
- **Evaluation:** build a small set of question → expected-source pairs and measure retrieval hit-rate as you change the knobs.
