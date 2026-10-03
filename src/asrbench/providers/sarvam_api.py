"""Sarvam AI platform ASR provider (network mode).

Requires ``SARVAM_API_KEY`` (see ``.env.example``) and ``pip install requests``.

.. warning::
   The endpoint path, form field names and response shape below are written
   from the public Sarvam API surface and are marked ``[verify]``. Check them
   against https://api.sarvam.ai/docs before trusting real numbers — they are
   overridable via ``SARVAM_API_BASE`` / ``SARVAM_ASR_PATH`` without code
   changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..manifest import Utterance
from .base import Provider, ProviderError

# [verify] exact endpoint + field names against the live API docs.
DEFAULT_BASE_URL = "https://api.sarvam.ai"
DEFAULT_ASR_PATH = "/speech-to-text"
DEFAULT_MODEL = "saarika:v2"

# [verify] language tag format (bare "hi" vs "hi-IN").
_LANGUAGE_TAGS = {"hi": "hi-IN", "mr": "mr-IN", "en": "en-IN", "ta": "ta-IN"}


def _extract_transcript(payload: Any) -> str:
    """Defensively pull transcript text out of a JSON response."""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        raise ProviderError(f"unexpected ASR response type: {type(payload).__name__}")
    for key in ("transcript", "text", "output", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # One level of nesting, e.g. {"results": {"transcript": "..."}}
    for value in payload.values():
        if isinstance(value, dict):
            try:
                return _extract_transcript(value)
            except ProviderError:
                continue
    raise ProviderError(f"no transcript field found in response: {payload!r}")


class SarvamASRProvider(Provider):
    name = "sarvam"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        asr_path: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "").strip()
        if not self.api_key:
            raise ProviderError(
                "SARVAM_API_KEY is not set. Copy .env.example to .env and fill it "
                "in, or run with --provider mock for an offline pipeline check."
            )
        try:
            import requests  # noqa: F401 - optional dependency
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ProviderError(
                "the 'requests' package is required for --provider sarvam: "
                "pip install requests"
            ) from exc

        import requests

        self._http = requests
        self.base_url = (base_url or os.environ.get("SARVAM_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        self.asr_path = asr_path or os.environ.get("SARVAM_ASR_PATH") or DEFAULT_ASR_PATH
        self.model = model or os.environ.get("SARVAM_ASR_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    def transcribe(self, utterance: Utterance) -> str:
        audio = Path(utterance.audio_path)
        if not audio.exists():
            raise ProviderError(f"audio file not found: {audio}")

        language = utterance.language
        data = {"model": self.model}
        if language and language != "unknown":
            data["language_code"] = _LANGUAGE_TAGS.get(language, language)

        url = f"{self.base_url}{self.asr_path}"
        headers = {"api-subscription-key": self.api_key}
        try:
            with audio.open("rb") as handle:
                response = self._http.post(
                    url,
                    headers=headers,
                    files={"file": (audio.name, handle, "audio/wav")},
                    data=data,
                    timeout=self.timeout,
                )
        except self._http.RequestException as exc:  # pragma: no cover - network
            raise ProviderError(f"ASR request failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"ASR request returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        return _extract_transcript(response.json())
