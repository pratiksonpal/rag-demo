# RAG Playground — Project Documentation

## What Is This?

RAG Playground is an **interactive, fully local** Retrieval-Augmented Generation (RAG) demo
built with Streamlit. It lets you load a document, chunk it, embed the chunks into a vector
store, and query it using a local LLM — all without any API keys or internet connection after
the initial model download.

The app is designed to be educational: each step of the RAG pipeline is exposed as a separate
tab with visualizations and explanations.

---

## RAG Pipeline — High Level

```
Document Text
     │
     ▼
 ┌─────────┐      LangChain text splitters
 │ CHUNKER │  ──────────────────────────────►  List of text chunks
 └─────────┘

     │
     ▼
 ┌─────────┐      sentence-transformers (all-MiniLM-L6-v2)
 │ EMBEDDER│  ──────────────────────────────►  384-dim float32 vectors
 └─────────┘

     │
     ▼
 ┌──────────────┐  FAISS IndexFlatIP
 │ VECTOR INDEX │  ──────────────────►  In-memory similarity index
 └──────────────┘

     │  (at query time)
     ▼
 ┌──────────┐     FAISS .search()
 │ RETRIEVER│  ──────────────────────────────►  Top-K chunk indices + scores
 └──────────┘

     │
     ▼
 ┌──────────┐     Ollama local LLM (llama3.2 / tinyllama)
 │ GENERATOR│  ──────────────────────────────►  Streamed text response
 └──────────┘
```

---

## File Structure

```
thirdProject-Rag/
├── app.py                  # Streamlit UI — orchestrates all pipeline steps
├── requirements.txt        # Python dependencies
├── data/
│   └── sample_docs.txt     # Default document loaded on startup
├── src/
│   ├── __init__.py
│   ├── chunker.py          # Text splitting logic (3 strategies)
│   ├── embedder.py         # Embedding model + FAISS index builder
│   ├── retriever.py        # Vector similarity search
│   └── generator.py        # Ollama LLM prompt builder + streamer
└── docs/
    └── project_overview.md  # This file
```

---

## Technology Stack

| Layer | Tool / Library | Why |
|---|---|---|
| UI | **Streamlit** | Rapid interactive web app with Python only |
| Text Splitting | **LangChain `langchain-text-splitters`** | Battle-tested chunking strategies |
| Embedding Model | **`sentence-transformers` — all-MiniLM-L6-v2** | Free, local, fast 384-dim embeddings |
| Vector Store | **FAISS (`faiss-cpu`) — `IndexFlatIP`** | In-memory exact cosine similarity search |
| LLM Inference | **Ollama** | Run quantized LLMs locally, no API key |
| Visualization | **Plotly** | Interactive scatter plot for embedding space |
| Dimensionality Reduction | **scikit-learn `PCA`** | Project 384D vectors to 2D for display |
| Numerical Ops | **NumPy** | Array manipulation for embeddings |

---

## Functionality — Tab by Tab

### Tab 1 — Text Splitting (`src/chunker.py`)

**What it does:** Divides the raw document into smaller, overlapping chunks that fit
within the embedding model's context window and carry focused meaning.

**How it works:**
- The user picks a strategy, chunk size (chars), and overlap (chars).
- `get_chunks()` dispatches to one of three private functions.
- Returns a flat `List[str]` of chunk texts.

**Three strategies:**

#### 1. Fixed Character (`_fixed`)
- Uses `langchain_text_splitters.CharacterTextSplitter` with `separator=""`.
- Splits purely by character count — no awareness of words or sentences.
- Produces uniform-size chunks. Can cut mid-word.
- Best for: structured/tabular data, logs.

#### 2. Recursive Character (`_recursive`)
- Uses `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- Tries separators in priority order: `\n\n` → `\n` → `. ` → ` ` → `""`.
- Respects paragraph → sentence → word boundaries before resorting to raw character split.
- Produces semantically coherent chunks. **Recommended for natural language.**

#### 3. Parent-Child (`_parent_child`)
- Two-pass split using `RecursiveCharacterTextSplitter` twice.
- **Parent pass:** `chunk_size × 2` — broad context windows.
- **Child pass:** `chunk_size` — precise, focused chunks (split from each parent).
- Returns the child chunks for indexing while preserving the parent's context.
- Used in advanced RAG to improve retrieval precision without losing context.

**UI features (in `app.py`):**
- Live metrics: total chunks, avg/min/max size, overlap.
- Editable source text area.
- Color-coded chunk cards.
- Chunk highlight map: overlays chunk boundaries on the source text.

---

### Tab 2 — Vector Embedding (`src/embedder.py` + `src/retriever.py`)

**What it does:** Converts every chunk into a dense numerical vector and stores them in
FAISS. Lets you run semantic queries to see which chunks are closest to a query.

#### Embedding (`src/embedder.py`)

**`get_model()`**
- Loads `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace (cached locally after first download).
- 22M parameter model, 6-layer MiniLM distilled from BERT.
- Output: 384-dimensional L2-normalized float32 vectors.

**`embed_texts(model, texts)`**
- Calls `model.encode()` with `normalize_embeddings=True` — outputs are L2-normalized.
- L2 normalization means cosine similarity = dot product (inner product), enabling use of FAISS `IndexFlatIP`.
- Returns a `(N, 384)` float32 NumPy array.

**`build_faiss_index(model, texts)`**
- Embeds all texts, creates a `faiss.IndexFlatIP` (exact brute-force inner product search).
- Adds all embeddings to the index.
- Returns both the raw embedding matrix and the FAISS index object.

#### Retrieval (`src/retriever.py`)

**`retrieve_chunks(index, query_embedding, top_k)`**
- Calls `index.search()` — FAISS returns the `top_k` nearest vectors by inner product score.
- Since embeddings are L2-normalized, inner product = cosine similarity ∈ [−1, 1].
- Returns `(indices, scores)` — indices into the original chunk list and their similarity scores.

**UI features (in `app.py`):**
- PCA projection: 384D vectors → 2D scatter plot via `sklearn.decomposition.PCA`.
- Query point shown as a red star; retrieved chunks highlighted with rank labels.
- Dashed lines connect query to retrieved chunks.
- Cosine similarity score shown per retrieved chunk with a visual bar.

---

### Tab 3 — Response Generation (`src/generator.py`)

**What it does:** Takes the retrieved chunks, injects them into a structured prompt,
and streams a grounded answer from a local Ollama LLM.

**`check_ollama(model_name)`**
- Calls `ollama.list()` to check if Ollama is running and if the selected model is installed.
- Returns `(True, "")` on success or `(False, error_message)` on failure.

**`build_prompt_messages(system_message, context_chunks, query)`**
- Constructs the OpenAI-style message list `[{role, content}, ...]`.
- System message: user-configurable instruction (e.g. "Answer only from the context").
- User message: numbered excerpts from retrieved chunks + the user's question.
- This is the **Augmentation** step — the LLM never sees the full document, only the relevant pieces.

**`stream_response(...)`**
- Calls `ollama.chat()` with `stream=True`.
- Yields individual token strings as they arrive — Streamlit renders them progressively.
- Passes `temperature` and `num_predict` (max tokens) via Ollama options.

**Ollama models supported:**
- `llama3.2:latest` (~2GB) — capable, good for complex questions.
- `tinyllama:latest` (~637MB) — fast, low memory, simpler answers.
- Any other model installed via `ollama pull <model>`.

**UI features (in `app.py`):**
- Live prompt preview: shows system message, retrieved chunks (color-coded), and user question.
- Real-time token streaming into the response panel.
- Temperature and max-tokens sliders.

---

## Data Flow — Step by Step

```
1. User pastes/uploads document text
        │
        ▼
2. app.py → chunker.get_chunks(text, strategy, size, overlap)
        │   Returns List[str] → stored in st.session_state.chunks
        ▼
3. app.py → embedder.build_faiss_index(model, chunks)
        │   Returns (np.ndarray, faiss.Index) → stored in session state
        ▼
4. User types a query
        │
        ▼
5. app.py → embedder.embed_texts(model, [query])
        │   Returns (1, 384) float32 array
        ▼
6. app.py → retriever.retrieve_chunks(index, query_emb, top_k)
        │   Returns (List[int], List[float]) — chunk indices + scores
        ▼
7. app.py → generator.stream_response(model, system_msg, chunks[indices], query, ...)
        │   Streams tokens via Ollama API
        ▼
8. Streamlit renders streamed response in real time
```

---

## State Management

Streamlit reruns the entire script on every user interaction. Expensive objects are
preserved across reruns using:

- **`st.session_state`** — stores chunks, embeddings matrix, FAISS index, last response.
- **`@st.cache_resource`** — caches the sentence-transformer model (loaded once per session).
- **`@st.cache_data`** — caches the sample document file read.

The `_init_state()` function in `app.py` sets defaults for all session state keys on first load.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (if not already running)
ollama serve
ollama pull llama3.2:latest   # or tinyllama:latest

# 3. Launch the app
streamlit run app.py
```

Opens at: http://localhost:8501

No API keys required. Fully local.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 | Free, local, fast, good quality for semantic search |
| Vector store | FAISS IndexFlatIP | In-memory, exact search, no server needed |
| LLM | Ollama | Fully local, no API key, supports multiple models |
| Chunking library | LangChain text splitters | Provides all three strategies out of the box |
| Similarity metric | Cosine (via normalized inner product) | Standard for semantic similarity; enabled by L2 normalization |
| UI | Streamlit | Python-only, fast to build, handles session state and streaming |
