"""Phase 4a — The LLM backend (swappable).

The whole point of this file: the rest of SonaRAG calls ONE function,
`generate(prompt, system) -> str`, and never knows or cares which engine
answers. Today it's Groq (free hosted). Switch `config.LLM_BACKEND` to
"claude" or "ollama" later and nothing else in the project changes.

We keep generation TEMPERATURE low (config.TEMPERATURE) because we want
grounded, factual answers from the retrieved passages — not creative writing.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

import config

# Load GROQ_API_KEY (etc.) from sona-rag/.env regardless of the current working
# directory (Streamlit, pytest, and the CLI may all launch from different places).
load_dotenv(config.PROJECT_ROOT / ".env")


class LLMError(RuntimeError):
    """Raised when the LLM backend can't run (missing key, unknown backend...)."""


# --- Groq (free hosted, OpenAI-compatible) ---------------------------------
@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "(no credit card), then add it to a .env file in sona-rag/:\n"
            "    GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=key)


def _generate_groq(prompt: str, system: str) -> str:
    response = _groq_client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_ANSWER_TOKENS,
    )
    return response.choices[0].message.content.strip()


# --- Public interface ------------------------------------------------------
def generate(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Generate a completion using whichever backend config.LLM_BACKEND names."""
    backend = config.LLM_BACKEND
    if backend == "groq":
        return _generate_groq(prompt, system)
    # Future backends slot in here without touching the rest of the project:
    #   if backend == "claude":  return _generate_claude(prompt, system)
    #   if backend == "ollama":  return _generate_ollama(prompt, system)
    raise LLMError(f"Unsupported LLM_BACKEND: {backend!r}")


if __name__ == "__main__":
    # Smoke test — confirms your key + model work end to end.
    try:
        out = generate("Reply with exactly: SonaRAG is wired up.",
                       system="You follow instructions precisely.")
        print("LLM responded:", out)
    except LLMError as e:
        print("LLM not ready:\n", e)
