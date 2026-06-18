"""Groq / Llama-70B LLM config for AutoGen 0.2, with 429 retry/backoff."""
import os
import time
import logging
import functools
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from shared.config import LLM_MAX_RETRIES, LLM_BACKOFF_BASE_S

logger = logging.getLogger(__name__)

# Verify this at https://console.groq.com — model IDs change occasionally.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Standard AutoGen 0.2 llm_config dict.
# Groq is OpenAI-compatible; base_url routes calls to the Groq endpoint.
llm_config = {
    "config_list": [{
        "model":    GROQ_MODEL,
        "api_key":  os.environ.get("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        "api_type": "openai",
    }],
    "temperature": 0.2,   # low = more deterministic structured output
    "timeout":     60,
    "cache_seed":  None,  # disable AutoGen response caching (we need fresh calls)
}


def with_retry(fn):
    """Decorator: retry *fn* on rate-limit errors (HTTP 429) with
    exponential backoff.  Other exceptions propagate immediately.

    Usage:
        @with_retry
        def call_llm(...): ...

    Or wrap at call-site:
        result = with_retry(client.chat.completions.create)(**kwargs)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "ratelimit" in msg:
                    wait = LLM_BACKOFF_BASE_S * (2 ** attempt)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Retrying in %.1fs.",
                        attempt + 1, LLM_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise
        # Final attempt — let it propagate naturally.
        raise last_exc  # type: ignore[misc]
    return wrapper


def make_groq_client():
    """Return an openai.OpenAI client pointed at Groq, for direct calls
    outside AutoGen (e.g. seed_dkb, metrics helpers).  The client's
    chat.completions.create is wrapped with with_retry automatically."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package is required — pip install openai") from exc

    client = openai.OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )
    # Patch create with retry so all direct calls benefit automatically.
    client.chat.completions.create = with_retry(client.chat.completions.create)
    return client
