"""Provider interface: anything that can turn one utterance into text."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..manifest import Utterance


class ProviderError(RuntimeError):
    """Raised for provider misconfiguration or a failed transcription call."""


class Provider(ABC):
    """A transcription backend (API, local model, or offline simulator)."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, utterance: Utterance) -> str:
        """Return the hypothesis text for *utterance*.

        Implementations raise ``ProviderError`` when the text cannot be
        produced (bad key, missing file, API failure) — never a partial or
        invented transcript, which would be scored as if it were real.
        """

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release any resources (HTTP sessions, model handles)."""
