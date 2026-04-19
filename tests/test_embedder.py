"""Tests for src/embedder.py — embedding generation and FAISS index."""

import pytest
import numpy as np
from src.embedder import get_model, embed_texts, build_faiss_index

TEXTS = [
    "XYZ Enterprises is a global technology firm.",
    "The company has offices in India, Germany, and USA.",
    "Engineering and Data AI are core departments.",
    "Cloud infrastructure is a primary focus area.",
]


@pytest.fixture(scope="module")
def model():
    return get_model()


class TestGetModel:
    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_encode(self, model):
        assert hasattr(model, "encode")


class TestEmbedTexts:
    def test_output_shape(self, model):
        embeddings = embed_texts(model, TEXTS)
        assert embeddings.shape == (len(TEXTS), 384)

    def test_output_dtype(self, model):
        embeddings = embed_texts(model, TEXTS)
        assert embeddings.dtype == np.float32

    def test_single_text(self, model):
        embeddings = embed_texts(model, ["single sentence"])
        assert embeddings.shape == (1, 384)

    def test_l2_normalized(self, model):
        embeddings = embed_texts(model, TEXTS)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(TEXTS)), atol=1e-5)

    def test_different_texts_different_vectors(self, model):
        e1 = embed_texts(model, ["cat"])
        e2 = embed_texts(model, ["database"])
        assert not np.allclose(e1, e2)


class TestBuildFaissIndex:
    def test_index_size(self, model):
        _, index = build_faiss_index(model, TEXTS)
        assert index.ntotal == len(TEXTS)

    def test_returns_embeddings_and_index(self, model):
        embeddings, index = build_faiss_index(model, TEXTS)
        assert embeddings is not None
        assert index is not None

    def test_embeddings_shape(self, model):
        embeddings, _ = build_faiss_index(model, TEXTS)
        assert embeddings.shape == (len(TEXTS), 384)

    def test_index_dimension(self, model):
        _, index = build_faiss_index(model, TEXTS)
        assert index.d == 384
