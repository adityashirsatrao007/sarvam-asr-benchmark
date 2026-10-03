"""Local open-source ASR baseline via faster-whisper (optional dependency).

Useful as the comparison arm of the benchmark (Sarvam vs Whisper). Guarded
import: the core package must work with nothing installed but Python.
"""

from __future__ import annotations

from ..manifest import Utterance
from .base import Provider, ProviderError

# whisper language codes are the same bare tags used in the manifest.
SUPPORTED_LANGUAGES = {"hi", "mr", "en", "bn", "ta", "te", "kn", "ml", "gu", "ur"}


class WhisperProvider(Provider):
    name = "whisper"

    def __init__(self, *, model: str = "small", device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ProviderError(
                "the 'faster-whisper' package is required for --provider whisper: "
                "pip install faster-whisper"
            ) from exc
        self._model = WhisperModel(model, device=device, compute_type="int8")

    def transcribe(self, utterance: Utterance) -> str:
        language = utterance.language
        kwargs = {}
        if language in SUPPORTED_LANGUAGES:
            kwargs["language"] = language
        try:
            segments, _info = self._model.transcribe(utterance.audio_path, **kwargs)
        except Exception as exc:  # pragma: no cover - model/env dependent
            raise ProviderError(f"whisper failed on {utterance.utt_id}: {exc}") from exc
        return " ".join(segment.text.strip() for segment in segments).strip()
