"""Model client for the evaluation harness.

Mirrors the LiteLLM call pattern used by the generation scripts
(``realistic_benchmark.llm_utils`` / ``generate_qa.py``): if ``LITELLM_BASE_URL``
is set, route through an OpenAI-compatible proxy; otherwise call the provider
directly via ``litellm.completion``.

``litellm``/``openai``/``dotenv`` are imported lazily inside ``complete`` so that
the mock path (and ``import toolsense_eval.*``) works with only the standard library —
no heavy dependencies required for ``--dry-run`` runs or for the unit tests.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Best-effort .env load; no-op if python-dotenv is not installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional for the eval path
        pass


def complete(model: str, prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
    """Return the model's text completion for a single user prompt.

    Uses the proxy (OpenAI client) when ``LITELLM_BASE_URL`` is set, else calls
    ``litellm.completion`` directly. Retries with exponential backoff.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    proxy_base_url = os.environ.get("LITELLM_BASE_URL", "").strip() or None
    proxy_api_key = os.environ.get("LITELLM_API_KEY", "").strip() or None

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            if proxy_base_url:
                from openai import OpenAI

                client = OpenAI(base_url=proxy_base_url, api_key=proxy_api_key or "unused")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            else:
                import litellm

                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            return response.choices[0].message.content or ""
        except ImportError:
            # A missing optional dependency (litellm/openai) will never succeed on
            # retry — surface it immediately instead of sleeping through backoffs.
            raise
        except Exception as e:  # noqa: BLE001 - surface after retries
            last_exc = e
            if attempt < max_retries - 1:
                logger.warning("Model call attempt %d failed: %s. Retrying...", attempt + 1, e)
                time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]
