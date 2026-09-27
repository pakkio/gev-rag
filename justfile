host := "127.0.0.1"
port := "8000"
base_url := "http://" + host + ":" + port

# Show available recipes
default:
    @just --list

# Create/refresh the venv and install deps
setup:
    uv venv
    uv pip install -r requirements.txt

# Pull the default local LLM used by the "local" engine
pull-model model="qwen3:14b":
    ollama pull {{model}}

# Run the FastAPI app (open http://127.0.0.1:8000)
run:
    uv run python app.py

# Upload one or more documents to the running server
# usage: just ingest uploads/robida_il_20_secolo.pdf
ingest *files:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{files}}" ]; then
        echo "usage: just ingest path/to/file.pdf [more files...]" >&2
        exit 1
    fi
    args=()
    for f in {{files}}; do args+=(-F "files=@$f"); done
    curl -s -X POST "{{base_url}}/api/upload" "${args[@]}" | python3 -m json.tool

# Ask a question against ingested documents (engines: comma-separated, e.g. jev,local,claude)
# usage: just ask "what year is the story set in?" jev
ask question engines="jev":
    #!/usr/bin/env bash
    set -euo pipefail
    python3 -c "import json, urllib.request; body = json.dumps({'question': '''{{question}}''', 'engines': '''{{engines}}'''.split(',')}).encode(); req = urllib.request.Request('{{base_url}}/api/ask', data=body, headers={'Content-Type': 'application/json'}); print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))"

# Check server + engine status
status:
    curl -s {{base_url}}/api/status | python3 -m json.tool

# Remove an ingested document by doc_id (see `just status` for ids)
delete-doc doc_id:
    curl -s -X DELETE {{base_url}}/api/documents/{{doc_id}} | python3 -m json.tool

# Dry-run benchmark (no API calls, downloads SQuAD sample)
bench-dry:
    uv run python bench.py --dry-run

# Build (or rebuild) a whole-document summary via map-reduce over every chunk
# usage: just summarize a21a8af1
summarize doc_id:
    curl -s -X POST {{base_url}}/api/documents/{{doc_id}}/summary | python3 -m json.tool

# Read a previously built summary
doc-summary doc_id:
    curl -s {{base_url}}/api/documents/{{doc_id}}/summary | python3 -m json.tool

# Ask a question against the WHOLE document in one call (not top-k retrieval like `ask`) —
# use for "what is this about", "does it mention X anywhere", things no single chunk answers
# usage: just ask-doc a21a8af1 "parla di macchine volanti?"
ask-doc doc_id question:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 -c "import json, urllib.request; body = json.dumps({'question': '''{{question}}'''}).encode(); req = urllib.request.Request('{{base_url}}/api/documents/{{doc_id}}/ask', data=body, headers={'Content-Type': 'application/json'}); print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))"
