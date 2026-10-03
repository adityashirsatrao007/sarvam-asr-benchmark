"""Deterministic offline provider for wiring checks — NOT a benchmark.

MockProvider never opens an audio file and never talks to a network. It
produces a *seeded, reproducible corruption* of each reference transcript so
the whole pipeline (manifest -> transcription -> WER/CER -> report) can be
exercised on machines with no API key and no GPU.

Its numbers are synthetic by construction. Never quote them as ASR results.
"""

from __future__ import annotations

import random

from ..manifest import Utterance
from .base import Provider

_DEVANAGARI = "कखगघङचछजझञटठडढणतथदधन"
_LATIN = "abcdefghijklmnopqrstuvwxyz"

# Tokens the mock may hallucinate, so insertions show up in the WER.
_FILLERS = ("है", "एक", "the", "um")


def _corrupt_token(token: str, rng: random.Random) -> str:
    """Delete or substitute one character inside *token*."""
    if len(token) < 2:
        return token
    position = rng.randrange(len(token))
    if rng.random() < 0.5:
        return token[:position] + token[position + 1 :]
    alphabet = (
        _DEVANAGARI
        if any("\u0900" <= char <= "\u097f" for char in token)
        else _LATIN
    )
    replacement = rng.choice(alphabet)
    return token[:position] + replacement + token[position + 1 :]


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        *,
        drop_rate: float = 0.08,
        sub_rate: float = 0.12,
        insert_rate: float = 0.06,
        seed: int = 13,
    ) -> None:
        self.drop_rate = drop_rate
        self.sub_rate = sub_rate
        self.insert_rate = insert_rate
        self.seed = seed

    def transcribe(self, utterance: Utterance) -> str:
        # Reseed per utterance, not per run: the hypothesis then depends only
        # on (seed, utt_id), so adding, dropping or reordering manifest rows
        # (--languages, --limit) never changes any *other* row's output.
        rng = random.Random(f"{self.seed}:{utterance.utt_id}")
        tokens: list[str] = []
        for token in utterance.reference.split():
            draw = rng.random()
            if draw < self.drop_rate:
                continue  # substitution-free deletion
            if draw < self.drop_rate + self.sub_rate:
                tokens.append(_corrupt_token(token, rng))
            else:
                tokens.append(token)
        if tokens and rng.random() < self.insert_rate:
            tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(_FILLERS))
        return " ".join(tokens)
