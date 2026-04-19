"""
retriever.py — Vector Similarity Search (the R in RAG)

Responsibility: given a query embedding and a FAISS index of chunk embeddings,
return the indices and similarity scores of the top-K most relevant chunks.

FAISS IndexFlatIP.search() performs exact brute-force inner product search.
Since embeddings are L2-normalized, inner product == cosine similarity ∈ [-1, 1].
"""

from typing import List, Tuple
import numpy as np
import faiss

from src.logger import get_logger, separator

log = get_logger(__name__)


def retrieve_chunks(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    """
    Search the FAISS index for the top-K chunks most similar to the query.

    Args:
        index:           FAISS IndexFlatIP built by embedder.build_faiss_index().
        query_embedding: Shape (1, 384) float32 array — the embedded user query.
        top_k:           Number of nearest neighbours to return.

    Returns:
        Tuple of (indices, scores) ordered by descending cosine similarity.
    """
    separator(log)
    log.info("[RETRIEVE START] Searching FAISS index (size=%d) for top-%d most similar chunks",
             index.ntotal, top_k)
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    idx_list = indices[0].tolist()
    score_list = scores[0].tolist()
    for rank, (idx, score) in enumerate(zip(idx_list, score_list), start=1):
        log.info("  Rank #%d → chunk[%d]  cosine_similarity=%.4f", rank, idx, score)
    log.info("[RETRIEVE DONE] Returned %d chunk(s)", len(idx_list))
    return idx_list, score_list
