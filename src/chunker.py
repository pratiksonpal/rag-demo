"""
chunker.py — Text Splitting

Strategies implemented (all via langchain-text-splitters):
  1. Fixed Character              — CharacterTextSplitter (separator="")
  2. Recursive Character          — RecursiveCharacterTextSplitter (hierarchical separators)
  3. Parent-Child                 — two-pass recursive split (parent 2× size → child)
  4. Token (Sentence-Transformers)— SentenceTransformersTokenTextSplitter (token count)
  5. Markdown Header              — MarkdownHeaderTextSplitter (split at # / ## / ###)
  6. Python Code                  — RecursiveCharacterTextSplitter.from_language(PYTHON)
"""

from typing import Dict, List, Tuple, Any
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

from src.logger import get_logger, separator

log = get_logger(__name__)

STRATEGIES = [
    "Fixed Character",
    "Recursive Character",
    "Parent-Child",
    "Token (Sentence-Transformers)",
    "Markdown Header",
    "Python Code",
]

# Module-level cache so the tokeniser is loaded only once
_token_splitter: SentenceTransformersTokenTextSplitter | None = None


def _get_token_splitter() -> SentenceTransformersTokenTextSplitter:
    global _token_splitter
    if _token_splitter is None:
        _token_splitter = SentenceTransformersTokenTextSplitter(model_name="all-MiniLM-L6-v2")
    return _token_splitter


def count_tokens(text: str) -> int:
    """Count all-MiniLM-L6-v2 tokens in text (tokenisation only, no model forward pass)."""
    return _get_token_splitter().count_tokens(text=text)


def get_chunks(
    text: str, strategy: str, chunk_size: int, overlap: int
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Entry point. Dispatch to the correct splitter based on strategy name.

    Returns:
        (chunks, meta) where meta is a strategy-specific dict of extra stats
        (e.g. parent_count, h1_count, tokens_per_chunk …).
    """
    separator(log)
    log.info("[TEXT SPLIT START] strategy='%s' | input=%d chars | chunk_size=%d | overlap=%d",
             strategy, len(text), chunk_size, overlap)

    dispatch = {
        "Fixed Character": _fixed,
        "Recursive Character": _recursive,
        "Parent-Child": _parent_child,
        "Token (Sentence-Transformers)": _token,
        "Markdown Header": _markdown_header,
        "Python Code": _python_code,
    }

    fn = dispatch.get(strategy)
    if fn is None:
        log.warning("unknown strategy '%s', returning empty list", strategy)
        return [], {}

    chunks, meta = fn(text, chunk_size, overlap)

    if chunks:
        sizes = [len(c) for c in chunks]
        log.info("[TEXT SPLIT DONE] strategy='%s' | chunks=%d | avg=%d chars | min=%d chars | max=%d chars",
                 strategy, len(chunks), int(sum(sizes) / len(sizes)), min(sizes), max(sizes))
    else:
        log.warning("[TEXT SPLIT DONE] strategy='%s' produced 0 chunks", strategy)

    return chunks, meta


# ── Strategy implementations ──────────────────────────────────────────────────

def _fixed(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    log.info(
        "Splitting %d chars into fixed blocks of %d chars each, "
        "with %d chars repeated from the previous chunk to preserve boundary context. "
        "No sentence or paragraph awareness — may cut mid-word.",
        len(text), chunk_size, overlap,
    )
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    log.debug("_fixed produced %d chunks", len(chunks))
    return chunks, {}


def _recursive(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    log.info(
        "Splitting %d chars hierarchically using separators in priority order: "
        "[paragraph (\\\\n\\\\n) → line (\\\\n) → sentence ('. ') → word (' ') → character ('')]. "
        "Each separator is tried first; if a chunk is still >%d chars the next separator is used. "
        "Overlap of %d chars is repeated at each boundary to avoid losing context.",
        len(text), chunk_size, overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    log.debug("_recursive produced %d chunks", len(chunks))
    return chunks, {}


def _parent_child(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    parent_size = chunk_size * 2
    child_overlap = max(0, overlap // 2)
    log.info(
        "Two-pass split on %d chars | "
        "Pass 1 (parent): target=%d chars — creates large segments to preserve broad context. "
        "Pass 2 (child): target=%d chars with %d chars overlap — splits each parent into focused retrieval units. "
        "Only child chunks are indexed.",
        len(text), parent_size, chunk_size, child_overlap,
    )
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=child_overlap)

    parents = parent_splitter.split_text(text)
    parent_avg = int(sum(len(p) for p in parents) / len(parents)) if parents else 0
    log.info("Pass 1 complete: %d parent chunks created (avg %d chars each)", len(parents), parent_avg)

    children: list = []
    child_to_parent: list = []          # child_to_parent[i] = 1-based parent index
    parent_sizes = [len(p) for p in parents]

    for pi, parent in enumerate(parents, 1):
        parent_children = child_splitter.split_text(parent)
        for child in parent_children:
            children.append(child)
            child_to_parent.append(pi)

    log.info("Pass 2 complete: %d parent chunks → %d child chunks (avg %.1f children/parent)",
             len(parents), len(children), len(children) / len(parents) if parents else 0)

    return children, {
        "parent_count": len(parents),
        "parent_size": parent_size,
        "child_overlap": child_overlap,
        "parent_sizes": parent_sizes,
        "child_to_parent": child_to_parent,
    }


def _token(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    tokens_per_chunk = min(chunk_size, 256)
    token_overlap = min(overlap, max(0, tokens_per_chunk - 1))
    clamped_note = (
        f"requested chunk_size={chunk_size} was clamped to model max of 256"
        if chunk_size > 256 else
        f"chunk_size={chunk_size} is within model max of 256"
    )
    log.info(
        "Splitting %d chars by token count using all-MiniLM-L6-v2 tokeniser | "
        "tokens_per_chunk=%d (%s) | token_overlap=%d tokens. "
        "Chunk size is measured in tokens (not characters) — guarantees every chunk fits "
        "within the embedding model's 256-token context window without truncation.",
        len(text), tokens_per_chunk, clamped_note, token_overlap,
    )
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=token_overlap,
    )
    chunks = splitter.split_text(text)
    log.debug("_token produced %d chunks", len(chunks))
    return chunks, {"tokens_per_chunk": tokens_per_chunk, "token_overlap": token_overlap}


def _markdown_header(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    log.info(
        "Splitting %d chars at markdown heading boundaries: # (H1), ## (H2), ### (H3). "
        "Each section from one header to the next becomes a separate chunk. "
        "chunk_size=%d and overlap=%d are not applied — split points determined by header presence.",
        len(text), chunk_size, overlap,
    )
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    docs = splitter.split_text(text)
    chunks = [d.page_content for d in docs if d.page_content.strip()]

    h1 = sum(1 for d in docs if d.metadata.get("H1"))
    h2 = sum(1 for d in docs if d.metadata.get("H2"))
    h3 = sum(1 for d in docs if d.metadata.get("H3"))

    # Build per-section header info for the log table
    section_headers: list = []
    for d in docs:
        if d.page_content.strip():
            if d.metadata.get("H3"):
                level, text_h = "H3", d.metadata["H3"]
            elif d.metadata.get("H2"):
                level, text_h = "H2", d.metadata["H2"]
            elif d.metadata.get("H1"):
                level, text_h = "H1", d.metadata["H1"]
            else:
                level, text_h = "—", "—"
            section_headers.append({"level": level, "text": text_h})

    if not chunks:
        log.info("No markdown headers (# / ## / ###) found — returning full text as a single chunk.")
        chunks = [text]
        h1 = h2 = h3 = 0
        section_headers = [{"level": "—", "text": "(no headers)"}]
    else:
        log.info("Found %d section(s): %d×H1, %d×H2, %d×H3", len(chunks), h1, h2, h3)

    return chunks, {
        "h1_count": h1,
        "h2_count": h2,
        "h3_count": h3,
        "section_headers": section_headers,
    }


def _python_code(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], Dict]:
    log.info(
        "Splitting %d chars of Python source using language-aware separators in priority order: "
        "[class definitions → function (def) boundaries → blank lines → newlines → statements]. "
        "chunk_size=%d chars | overlap=%d chars.",
        len(text), chunk_size, overlap,
    )
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    log.debug("_python_code produced %d chunks", len(chunks))
    return chunks, {}
