"""Provider registry."""

from __future__ import annotations

from .base import Provider, ProviderError
from .mock import MockProvider

__all__ = ["Provider", "ProviderError", "create_provider"]


def create_provider(name: str, *, model: str | None = None, seed: int = 13) -> Provider:
    """Instantiate a provider by CLI name."""
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "sarvam":
        from .sarvam_api import SarvamASRProvider

        return SarvamASRProvider(model=model)
    if name == "whisper":
        from .whisper_local import WhisperProvider

        return WhisperProvider(model=model or "small")
    raise ProviderError(
        f"unknown provider '{name}' (expected one of: mock, sarvam, whisper)"
    )
