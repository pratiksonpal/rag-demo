"""Tests for src/retriever.py — FAISS similarity search."""

import pytest
import numpy as np
import faiss
from src.retriever import retrieve_chunks
from src.embedder import get_model, embed_texts, build_faiss_index

TEXTS = [
    "XYZ Enterprises focuses on cloud infrastructure.",
    "The HR department manages employee benefits.",
    "AI analytics is a key product offering.",
    "Finance team handles budgeting and forecasting.",
    "IT Operations manages servers and network.",
]


@pytest.fixture(scope="module")
def model():
    return get_model()


@pytest.fixture(scope="module")
def index_and_embeddings(model):
    embeddings, index = build_faiss_index(model, TEXTS)
    return embeddings, index


class TestRetrieveChunks:
    def test_returns_correct_count(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["cloud services"])
        indices, scores = retrieve_chunks(index, query_emb, top_k=3)
        assert len(indices) == 3
        assert len(scores) == 3

    def test_scores_descending(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["AI machine learning"])
        _, scores = retrieve_chunks(index, query_emb, top_k=3)
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_cosine_similarity(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["cloud infrastructure"])
        _, scores = retrieve_chunks(index, query_emb, top_k=5)
        for score in scores:
            assert -1.0 <= score <= 1.0

    def test_indices_are_valid(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["HR employees"])
        indices, _ = retrieve_chunks(index, query_emb, top_k=3)
        for idx in indices:
            assert 0 <= idx < len(TEXTS)

    def test_top_k_1_returns_single_result(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["finance budget"])
        indices, scores = retrieve_chunks(index, query_emb, top_k=1)
        assert len(indices) == 1
        assert len(scores) == 1

    def test_semantically_relevant_result(self, model, index_and_embeddings):
        _, index = index_and_embeddings
        query_emb = embed_texts(model, ["cloud infrastructure"])
        indices, _ = retrieve_chunks(index, query_emb, top_k=1)
        assert indices[0] == 0
