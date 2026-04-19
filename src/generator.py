"""
generator.py — LLM Response Generation (the A + G in RAG)

Responsibility: build an augmented prompt from retrieved context chunks and
stream a grounded answer from a local Ollama LLM.

Tool used: Ollama (ollama Python SDK)
  Runs quantized LLMs locally at localhost:11434.
  Default model: tinyllama:latest (~637 MB)
"""

from typing import List, Tuple, Generator
import ollama

from src.logger import get_logger, separator

log = get_logger(__name__)


def check_ollama(model_name: str) -> Tuple[bool, str]:
    """
    Verify that Ollama is running and the requested model is available locally.

    Returns:
        Tuple of (ok: bool, error_message: str).
    """
    separator(log)
    log.info("[OLLAMA CHECK] Verifying Ollama is reachable and model '%s' is pulled", model_name)
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        pulled = any(model_name in m for m in available)
        if not pulled:
            log.warning("model '%s' not found | available=%s", model_name, available)
            return False, f"Model '{model_name}' not found. Run: ollama pull {model_name}"
        log.info("ollama ok | model='%s' is available", model_name)
        return True, ""
    except Exception as e:
        log.error("cannot reach ollama: %s", e)
        return False, f"Cannot reach Ollama. Is it running? ({e})"


def build_prompt_messages(
    system_message: str,
    context_chunks: List[str],
    query: str,
) -> List[dict]:
    """
    Construct the augmented prompt in OpenAI-compatible message format.

    [system]  User-configurable instruction.
    [user]    Numbered excerpts from retrieved chunks + the user's question.
    """
    excerpts = "\n\n".join(
        f"--- Excerpt {i + 1} ---\n{chunk.strip()}" for i, chunk in enumerate(context_chunks)
    )
    user_content = (
        f"Here are relevant excerpts from the document:\n\n"
        f"{excerpts}\n\n"
        f"Using only the excerpts above, answer the following question clearly and concisely:\n"
        f"{query}"
    )
    log.debug("prompt built | context_chunks=%d  query_len=%d  prompt_len=%d",
              len(context_chunks), len(query), len(user_content))
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def stream_response(
    model_name: str,
    system_message: str,
    context_chunks: List[str],
    query: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """
    Stream token-by-token LLM response via Ollama.

    Builds the augmented prompt, calls ollama.chat() with stream=True, and
    yields each content token as it arrives.
    """
    separator(log)
    log.info("[GENERATE START] model='%s' | context_chunks=%d | temperature=%.2f | max_tokens=%d | query='%.80s'",
             model_name, len(context_chunks), temperature, max_tokens, query)
    messages = build_prompt_messages(system_message, context_chunks, query)
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    total_tokens = 0
    for chunk in stream:
        content = chunk.message.content
        if content:
            total_tokens += 1
            yield content
    log.info("generation done | model=%s  token_chunks_yielded=%d", model_name, total_tokens)
