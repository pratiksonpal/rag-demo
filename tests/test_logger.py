"""Tests for src/logger.py — logging setup."""

import logging
import pytest
from src.logger import get_logger, separator


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefixed(self):
        logger = get_logger("mymodule")
        assert logger.name == "rag.mymodule"

    def test_same_name_returns_same_logger(self):
        l1 = get_logger("shared")
        l2 = get_logger("shared")
        assert l1 is l2

    def test_different_names_return_different_loggers(self):
        l1 = get_logger("module_a")
        l2 = get_logger("module_b")
        assert l1 is not l2

    def test_logger_is_enabled(self):
        logger = get_logger("enabled_check")
        assert logger.isEnabledFor(logging.DEBUG)


class TestSeparator:
    def test_separator_does_not_raise(self):
        logger = get_logger("separator_test")
        try:
            separator(logger)
        except Exception as e:
            pytest.fail(f"separator() raised an exception: {e}")
