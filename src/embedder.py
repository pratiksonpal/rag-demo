"""
embedder.py — Embedding Generation & FAISS Index

Responsibility: convert text strings into dense numerical vectors (embeddings)
and build a FAISS index for fast similarity search.

Tools used:
  - sentence-transformers (all-MiniLM-L6-v2):
      Produces 384-dimensional L2-normalized float32 vectors.
  - FAISS (faiss-cpu) — IndexFlatIP:
      Exact brute-force inner product search.
      Because embeddings are L2-normalized, inner product == cosine similarity.
"""

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.logger import get_logger, separator

log = get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    """
    Load and return the sentence-transformer embedding model.

    Downloaded from HuggingFace on first call and cached locally (~90 MB).
    In app.py wrapped with @st.cache_resource to load once per session.
    """
    separator(log)
    log.info("[EMBED MODEL LOAD] Loading sentence-transformer model '%s'", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("[EMBED MODEL LOAD DONE] Model ready | max_seq_length=%s tokens",
             getattr(model, "max_seq_length", "?"))
    return model


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalized embedding vectors.

    Returns:
        NumPy array of shape (len(texts), 384), dtype float32.
    """
    log.debug("embedding %d text(s)", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    result = embeddings.astype(np.float32)
    log.debug("embeddings shape: %s", result.shape)
    return result


def build_faiss_index(
    model: SentenceTransformer, texts: List[str]
) -> Tuple[np.ndarray, faiss.Index]:
    """
    Embed all texts and build a FAISS IndexFlatIP for cosine similarity search.

    Returns:
        Tuple of (embeddings array, FAISS index).
    """
    separator(log)
    log.info("[FAISS BUILD START] Encoding %d chunks into embedding vectors", len(texts))
    embeddings = embed_texts(model, texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("[FAISS BUILD DONE] Index ready | vectors=%d | dimensions=%d | index_type=IndexFlatIP (cosine via inner product)",
             index.ntotal, dim)
    return embeddings, index
