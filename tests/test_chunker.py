"""Tests for src/chunker.py — all six chunking strategies."""

import pytest
from src.chunker import get_chunks, count_tokens, STRATEGIES

SAMPLE_TEXT = (
    "XYZ Enterprises is a global technology firm.\n\n"
    "It focuses on cloud infrastructure and AI analytics.\n\n"
    "The company has offices in India, Germany, and USA.\n\n"
    "Employees number around 3,500 across five departments.\n\n"
    "Engineering, Data & AI, HR, Finance, and IT Operations."
)

PYTHON_CODE = (
    "def greet(name):\n"
    "    return f'Hello, {name}'\n\n"
    "class Greeter:\n"
    "    def __init__(self, prefix):\n"
    "        self.prefix = prefix\n\n"
    "    def greet(self, name):\n"
    "        return f'{self.prefix} {name}'\n"
)

MARKDOWN_TEXT = (
    "# Introduction\n\nThis is the intro section.\n\n"
    "## Background\n\nThis is background content.\n\n"
    "### Details\n\nDetailed information goes here.\n"
)


class TestStrategiesList:
    def test_six_strategies_defined(self):
        assert len(STRATEGIES) == 6

    def test_strategy_names(self):
        expected = {
            "Fixed Character", "Recursive Character", "Parent-Child",
            "Token (Sentence-Transformers)", "Markdown Header", "Python Code",
        }
        assert set(STRATEGIES) == expected


class TestFixedCharacter:
    def test_returns_chunks(self):
        chunks, meta = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=100, overlap=10)
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=50, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_returns_strings(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=100, overlap=10)
        assert all(isinstance(c, str) for c in chunks)

    def test_meta_is_dict(self):
        _, meta = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=100, overlap=10)
        assert isinstance(meta, dict)

    def test_overlap_increases_chunk_count(self):
        chunks_no_overlap, _ = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=80, overlap=0)
        chunks_with_overlap, _ = get_chunks(SAMPLE_TEXT, "Fixed Character", chunk_size=80, overlap=20)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestRecursiveCharacter:
    def test_returns_chunks(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Recursive Character", chunk_size=100, overlap=10)
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Recursive Character", chunk_size=80, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= 80

    def test_prefers_paragraph_boundaries(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Recursive Character", chunk_size=200, overlap=0)
        assert len(chunks) > 0
        assert all(c.strip() for c in chunks)


class TestParentChild:
    def test_returns_child_chunks(self):
        chunks, meta = get_chunks(SAMPLE_TEXT, "Parent-Child", chunk_size=80, overlap=10)
        assert len(chunks) > 0

    def test_meta_contains_parent_info(self):
        _, meta = get_chunks(SAMPLE_TEXT, "Parent-Child", chunk_size=80, overlap=10)
        assert "parent_count" in meta
        assert "parent_size" in meta
        assert "child_to_parent" in meta

    def test_parent_size_is_double_chunk_size(self):
        chunk_size = 80
        _, meta = get_chunks(SAMPLE_TEXT, "Parent-Child", chunk_size=chunk_size, overlap=10)
        assert meta["parent_size"] == chunk_size * 2

    def test_child_to_parent_mapping_length_matches_chunks(self):
        chunks, meta = get_chunks(SAMPLE_TEXT, "Parent-Child", chunk_size=80, overlap=10)
        assert len(meta["child_to_parent"]) == len(chunks)

    def test_parent_indices_are_valid(self):
        chunks, meta = get_chunks(SAMPLE_TEXT, "Parent-Child", chunk_size=80, overlap=10)
        parent_count = meta["parent_count"]
        for idx in meta["child_to_parent"]:
            assert 1 <= idx <= parent_count


class TestTokenSentenceTransformers:
    def test_returns_chunks(self):
        chunks, _ = get_chunks(SAMPLE_TEXT, "Token (Sentence-Transformers)", chunk_size=50, overlap=5)
        assert len(chunks) > 0

    def test_clamps_to_256_tokens(self):
        _, meta = get_chunks(SAMPLE_TEXT, "Token (Sentence-Transformers)", chunk_size=500, overlap=0)
        assert meta["tokens_per_chunk"] == 256

    def test_within_limit_not_clamped(self):
        _, meta = get_chunks(SAMPLE_TEXT, "Token (Sentence-Transformers)", chunk_size=100, overlap=0)
        assert meta["tokens_per_chunk"] == 100

    def test_meta_has_token_fields(self):
        _, meta = get_chunks(SAMPLE_TEXT, "Token (Sentence-Transformers)", chunk_size=50, overlap=5)
        assert "tokens_per_chunk" in meta
        assert "token_overlap" in meta


class TestMarkdownHeader:
    def test_splits_at_headers(self):
        chunks, meta = get_chunks(MARKDOWN_TEXT, "Markdown Header", chunk_size=200, overlap=0)
        assert len(chunks) >= 3

    def test_meta_contains_header_counts(self):
        _, meta = get_chunks(MARKDOWN_TEXT, "Markdown Header", chunk_size=200, overlap=0)
        assert "h1_count" in meta
        assert "h2_count" in meta
        assert "h3_count" in meta

    def test_h1_detected(self):
        _, meta = get_chunks(MARKDOWN_TEXT, "Markdown Header", chunk_size=200, overlap=0)
        assert meta["h1_count"] >= 1

    def test_no_headers_returns_full_text(self):
        plain = "This is plain text with no markdown headers at all."
        chunks, _ = get_chunks(plain, "Markdown Header", chunk_size=200, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == plain

    def test_section_headers_in_meta(self):
        _, meta = get_chunks(MARKDOWN_TEXT, "Markdown Header", chunk_size=200, overlap=0)
        assert "section_headers" in meta
        assert isinstance(meta["section_headers"], list)


class TestPythonCode:
    def test_splits_python_code(self):
        chunks, _ = get_chunks(PYTHON_CODE, "Python Code", chunk_size=80, overlap=0)
        assert len(chunks) > 0

    def test_returns_strings(self):
        chunks, _ = get_chunks(PYTHON_CODE, "Python Code", chunk_size=80, overlap=0)
        assert all(isinstance(c, str) for c in chunks)


class TestUnknownStrategy:
    def test_unknown_returns_empty(self):
        chunks, meta = get_chunks(SAMPLE_TEXT, "Nonexistent Strategy", chunk_size=100, overlap=10)
        assert chunks == []
        assert meta == {}


class TestCountTokens:
    def test_returns_integer(self):
        result = count_tokens("Hello world")
        assert isinstance(result, int)

    def test_empty_string(self):
        result = count_tokens("")
        assert isinstance(result, int)

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello world this is a longer sentence with more words")
        assert long > short
