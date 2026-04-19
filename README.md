# RAG Playground

An interactive, educational RAG demo that visualizes every step of the pipeline.

## What it demonstrates

| Tab | RAG Step | What you see |
|-----|----------|--------------|
| ① Text Splitting | Pre-processing | 6 chunking strategies; color-coded chunk boundaries |
| ② Vector Embedding | **R**etrieval | 2D PCA scatter of embeddings + FAISS similarity search |
| ③ Response Generation | **A**ugment + **G**enerate | Prompt preview with context · Streamed LLM response |

## Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Chunking | LangChain text splitters (Fixed / Recursive / Parent-Child / Token / Markdown / Python) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (50MB, CPU) |
| Vector DB | FAISS (`IndexFlatIP` — cosine similarity) |
| LLM | Ollama (local) — `tinyllama:latest` default |
| Visualization | Plotly + scikit-learn PCA |

All open-source. No paid APIs. Runs fully offline after setup.

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

```bash
# Install from https://ollama.com
ollama serve
ollama pull tinyllama      # ~637MB — default model used by the app
# or
ollama pull llama3.2:3b   # ~2GB — better quality
```

### 3. Run the app

```bash
streamlit run app.py
# if the above doesn't work
python -m streamlit run app.py
```

## Usage

1. **Tab 1 — Text Splitting**: Choose a chunking strategy and chunk size. Click **Apply** to see the source document color-highlighted with chunk boundaries, and the generated chunks on the right.

2. **Tab 2 — Vector Embedding**: Click **Build FAISS Index** to embed all chunks. Enter a query to see which chunks are retrieved (highlighted in the 2D scatter plot by rank).

3. **Tab 3 — Response Generation**: Enter a question. The left panel shows the full prompt (system message + retrieved context chunks). Click **Generate Response** to stream the LLM's answer.

## Testing

### Unit tests (66 tests — no server required)

```bash
python -m pytest
# report → tests/report.html
```

### UI tests (32 tests — requires Streamlit + Chromium)

```bash
# First-time setup: install Chromium for Playwright
python -m playwright install chromium

python -m pytest -c pytest-ui.ini
# report → tests/ui_report.html
```

## Project Structure

```
thirdProject-Rag/
├── app.py              # Streamlit app (3 tabs)
├── requirements.txt
├── pytest.ini          # Unit test config
├── pytest-ui.ini       # UI test config
├── data/
│   └── sample_docs.txt # XYZ Enterprises synthetic dataset
├── src/
│   ├── chunker.py      # 6 strategies: Fixed / Recursive / Parent-Child / Token / Markdown / Python
│   ├── embedder.py     # sentence-transformers + FAISS index builder
│   ├── retriever.py    # FAISS cosine similarity search
│   ├── generator.py    # Ollama streaming inference
│   └── logger.py       # Centralised logging (writes to logs/)
└── tests/
    ├── report.html         # Unit test execution report
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_retriever.py
    ├── test_generator.py
    ├── test_logger.py
    └── ui/
        ├── ui_report.html  # UI test execution report
        ├── conftest.py     # Streamlit server + Playwright fixtures
        └── test_ui.py      # 32 UI tests (Playwright + headless Chromium)
```
