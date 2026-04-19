# RAG Playground — Claude Code Context

## Project Goal
Interactive, educational RAG demo that visualizes every step of the pipeline via a Streamlit UI.

## Architecture
- **Chunk**: Split input text into overlapping chunks (`src/chunker.py`)
- **Embed**: Generate 384-dim L2-normalized vectors via `all-MiniLM-L6-v2` (`src/embedder.py`)
- **Index**: Store vectors in an in-memory FAISS `IndexFlatIP` (`src/embedder.py`)
- **Retrieve**: Cosine similarity search at query time (`src/retriever.py`)
- **Generate**: Build augmented prompt and stream response from local Ollama LLM (`src/generator.py`)

## Key Files
- `app.py` — Streamlit entry point; 3-tab UI (Text Splitting / Vector Embedding / Response Generation)
- `src/chunker.py` — 6 chunking strategies: Fixed Character, Recursive Character, Parent-Child, Token (ST), Markdown Header, Python Code
- `src/embedder.py` — sentence-transformers embedding + FAISS index builder
- `src/retriever.py` — FAISS cosine similarity search
- `src/generator.py` — Ollama streaming inference (check_ollama, build_prompt_messages, stream_response)
- `src/logger.py` — centralised logging; writes to `logs/DD-MM-YYYY/dd-mm-yyyy_hh-mm-ss.log`

## Decisions (finalised)
- **Vector store**: FAISS (`faiss-cpu`, `IndexFlatIP` — exact cosine via inner product)
- **Embedding model**: `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, CPU, ~90 MB)
- **LLM**: Ollama (local) — default `tinyllama:latest`; no cloud API key required
- **Orchestration**: plain Python + Streamlit (no LangChain orchestration layer)
- **Text splitting**: LangChain text splitters (`langchain-text-splitters`)

## Testing
- **Unit tests** (66): `python -m pytest` → `tests/report.html`
  - Covers: chunker (all 6 strategies), embedder, retriever, generator (mocked Ollama), logger
- **UI tests** (32): `python -m pytest -c pytest-ui.ini` → `tests/ui_report.html`
  - Covers: page load, sidebar, tab navigation, Tab 1 interactions, Tab 2 & Tab 3 element presence
  - Requires Playwright Chromium: `python -m playwright install chromium`
  - Starts Streamlit on port 8502 as a subprocess; Ollama not required for UI tests

## Environment
- Python 3.x
- Ollama must be running locally (`ollama serve`) with at least one model pulled
- No external API keys required
