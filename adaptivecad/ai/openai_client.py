from __future__ import annotations

import os
from typing import TYPE_CHECKING

try:
    from openai import OpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from openai import OpenAI as _OpenAI  # noqa: F401

_client: "OpenAI | None" = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client configured from the environment."""
    global _client
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run 'pip install openai' to enable the AI copilot."
        ) from _OPENAI_IMPORT_ERROR

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        _client = OpenAI(api_key=api_key)
    return _client
