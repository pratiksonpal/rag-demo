"""
logger.py — Centralised logging setup for the RAG demo.

Log directory layout:
    logs/
      DD-MM-YYYY/
        dd-mm-yyyy_hh-mm-ss.log   ← one file per application run

All src modules call get_logger(__name__) to obtain a child logger that
shares the same file handler, so every step of the pipeline lands in one file.
"""

import logging
from datetime import datetime
from pathlib import Path

_initialized = False
_log_file: Path | None = None


def _init() -> None:
    global _initialized, _log_file
    if _initialized:
        return

    now = datetime.now()
    day_folder = now.strftime("%d-%m-%Y")
    file_name = now.strftime("%d-%m-%Y_%H-%M-%S") + ".log"

    log_dir = Path(__file__).parent.parent / "logs" / day_folder
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = log_dir / file_name

    root = logging.getLogger("rag")
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    fh = logging.FileHandler(_log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger under the shared 'rag' hierarchy."""
    _init()
    return logging.getLogger(f"rag.{name}")


def separator(logger: logging.Logger) -> None:
    """Log a visual separator line before the start of a new event/action."""
    logger.info("-------------------------")
