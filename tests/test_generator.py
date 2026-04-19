"""Tests for src/generator.py — prompt building and Ollama integration (mocked)."""

import pytest
from unittest.mock import patch, MagicMock
from src.generator import build_prompt_messages, check_ollama, stream_response

SYSTEM_MSG = "You are a helpful assistant. Answer only from the provided context."
CONTEXT_CHUNKS = [
    "XYZ Enterprises has 3,500 employees.",
    "The company operates in India, Germany, and USA.",
]
QUERY = "How many employees does XYZ Enterprises have?"


class TestBuildPromptMessages:
    def test_returns_two_messages(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        assert len(messages) == 2

    def test_first_message_is_system(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_MSG

    def test_second_message_is_user(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        assert messages[1]["role"] == "user"

    def test_user_message_contains_query(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        assert QUERY in messages[1]["content"]

    def test_user_message_contains_all_chunks(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        user_content = messages[1]["content"]
        for chunk in CONTEXT_CHUNKS:
            assert chunk.strip() in user_content

    def test_excerpts_are_numbered(self):
        messages = build_prompt_messages(SYSTEM_MSG, CONTEXT_CHUNKS, QUERY)
        user_content = messages[1]["content"]
        assert "Excerpt 1" in user_content
        assert "Excerpt 2" in user_content

    def test_single_chunk(self):
        messages = build_prompt_messages(SYSTEM_MSG, ["Only one context chunk."], QUERY)
        assert len(messages) == 2
        assert "Excerpt 1" in messages[1]["content"]

    def test_empty_chunks(self):
        messages = build_prompt_messages(SYSTEM_MSG, [], QUERY)
        assert len(messages) == 2
        assert QUERY in messages[1]["content"]


class TestCheckOllama:
    def test_returns_false_when_ollama_unreachable(self):
        with patch("src.generator.ollama.list", side_effect=Exception("Connection refused")):
            ok, msg = check_ollama("tinyllama:latest")
        assert ok is False
        assert "Connection refused" in msg or "Cannot reach" in msg

    def test_returns_false_when_model_not_pulled(self):
        mock_model = MagicMock()
        mock_model.model = "llama3.2:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("src.generator.ollama.list", return_value=mock_response):
            ok, msg = check_ollama("tinyllama:latest")
        assert ok is False
        assert "tinyllama:latest" in msg

    def test_returns_true_when_model_available(self):
        mock_model = MagicMock()
        mock_model.model = "tinyllama:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("src.generator.ollama.list", return_value=mock_response):
            ok, msg = check_ollama("tinyllama:latest")
        assert ok is True
        assert msg == ""


class TestStreamResponse:
    def test_yields_tokens(self):
        mock_chunk = MagicMock()
        mock_chunk.message.content = "Hello"
        with patch("src.generator.ollama.chat", return_value=[mock_chunk, mock_chunk]):
            tokens = list(stream_response(
                "tinyllama:latest", SYSTEM_MSG, CONTEXT_CHUNKS, QUERY,
                temperature=0.7, max_tokens=100,
            ))
        assert tokens == ["Hello", "Hello"]

    def test_skips_empty_content(self):
        chunk_with_content = MagicMock()
        chunk_with_content.message.content = "word"
        chunk_empty = MagicMock()
        chunk_empty.message.content = ""
        with patch("src.generator.ollama.chat", return_value=[chunk_with_content, chunk_empty]):
            tokens = list(stream_response(
                "tinyllama:latest", SYSTEM_MSG, CONTEXT_CHUNKS, QUERY,
                temperature=0.7, max_tokens=100,
            ))
        assert tokens == ["word"]
