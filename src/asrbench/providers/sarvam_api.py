"""Sarvam AI platform ASR provider (network mode).

Requires ``SARVAM_API_KEY`` — read from the environment or from a local
``.env`` (see ``.env.example``) — and ``pip install requests``.

.. note::
   Live-checked against https://api.sarvam.ai on 2026-10-03: endpoint path,
   ``api-subscription-key`` auth, the ``model``/``language_code`` form fields
   and the response's ``transcript`` key were all exercised for real (the API
   itself rejected the retired ``saarika:v2`` naming and named ``saaras:v3``).
   Language tags other than ``hi-IN`` remain ``[verify]``. Everything stays
   overridable via ``SARVAM_API_BASE`` / ``SARVAM_ASR_PATH`` without code
   changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..manifest import Utterance
from .base import Provider, ProviderError

# Live-verified 2026-10-03 (path, auth header, form fields, model id).
DEFAULT_BASE_URL = "https://api.sarvam.ai"
DEFAULT_ASR_PATH = "/speech-to-text"
DEFAULT_MODEL = "saaras:v3"

# [verify] language tag format (bare "hi" vs "hi-IN").
_LANGUAGE_TAGS = {"hi": "hi-IN", "mr": "mr-IN", "en": "en-IN", "ta": "ta-IN"}


def _load_dotenv(path: Path) -> None:
    """Fill *unset* ``os.environ`` entries from a local ``KEY=VALUE`` file.

    The repo ships ``.env.example`` and tells people to ``cp`` it to ``.env``,
    so that file has to actually be read — stdlib only, because a zero
    dependency core is a stated requirement. Variables already present in the
    environment keep their value, so an explicit ``export`` still wins.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def _extract_transcript(payload: Any) -> str:
    """Defensively pull transcript text out of a JSON response."""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        raise ProviderError(f"unexpected ASR response type: {type(payload).__name__}")
    seen_empty = False
    for key in ("transcript", "text", "output", "result"):
        value = payload.get(key)
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            seen_empty = True
    if seen_empty:
        # A present-but-empty transcript is a real answer: the model heard no
        # speech (silence, a pure tone). That is an empty hypothesis for WER
        # to score, not a malformed response. Live-confirmed 2026-10-03.
        return ""
    # One level of nesting, e.g. {"results": {"transcript": "..."}}
    for value in payload.values():
        if isinstance(value, dict):
            try:
                return _extract_transcript(value)
            except ProviderError:
                continue
    # Truncated: a failed response can be a whole HTML error page, and an
    # exception message is not the place to dump it.
    raise ProviderError(f"no transcript field found in response: {repr(payload)[:300]}")


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
        _load_dotenv(Path(".env"))
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "").strip()
        if not self.api_key:
            raise ProviderError(
                "SARVAM_API_KEY is not set. Copy .env.example to .env and fill it "
                "in, or run with --provider mock for an offline pipeline check."
            )
        try:
            import requests  # optional dependency: only this provider needs it
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ProviderError(
                "the 'requests' package is required for --provider sarvam: "
                "pip install requests"
            ) from exc

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
        try:
            payload = response.json()
        except ValueError as exc:
            # 200 with a non-JSON body (proxy/gateway page) — the same class
            # of failure _extract_transcript already handles, so it should
            # surface as a ProviderError, not a traceback.
            raise ProviderError(
                f"ASR response was not valid JSON: {response.text[:300]}"
            ) from exc
        return _extract_transcript(payload)
