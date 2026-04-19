# RAG Demo — Claude Code Context

## Project Goal
Build a working RAG (Retrieval-Augmented Generation) demo using Claude as the LLM.

## Architecture
- **Ingest**: Load and chunk documents from `data/`
- **Embed**: Generate embeddings for chunks
- **Store**: Persist embeddings in a vector store
- **Retrieve**: Semantic search at query time
- **Generate**: Claude API call with retrieved context injected into prompt

## Key Files
- `src/ingest.py` — document loading & chunking
- `src/embed.py` — embedding generation
- `src/retrieve.py` — vector search
- `src/generate.py` — Claude LLM integration
- `main.py` — orchestration entry point

## Decisions (TBD)
- Vector store choice: ChromaDB (local, easy setup) vs FAISS vs Pinecone
- Embedding model: sentence-transformers (free/local) vs OpenAI embeddings
- Orchestration: plain Python vs LangChain

## Environment
- Python 3.x
- `ANTHROPIC_API_KEY` required
