"""Word/character error rate metrics, dependency-free.

WER is computed the standard way: ``edit_distance(ref, hyp) / len(ref)``
over normalised tokens. It can exceed 1.0 when the hypothesis contains
insertions — that is correct behaviour, not a bug.
"""

from __future__ import annotations

from collections.abc import Sequence

from .textnorm import char_tokens, word_tokens


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences (word-level ops)."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_token != hyp_token)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def _error_rate(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    # No reference tokens means no denominator: the rate is undefined. Cap it
    # at 1.0 rather than raising or returning inf, so one blank gold
    # transcript cannot poison an average. Corpus numbers never come from
    # this function (report.aggregate pools edit counts), and load_manifest()
    # rejects references that normalise to nothing before they can be scored.
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Single-utterance WER over normalised word tokens.

    Corpus numbers are *not* the mean of these values: see ``report.aggregate``
    for the micro-average that is actually reported.
    """
    return _error_rate(word_tokens(reference), word_tokens(hypothesis))


def char_error_rate(reference: str, hypothesis: str) -> float:
    """CER for a single utterance (spaces excluded)."""
    return _error_rate(char_tokens(reference), char_tokens(hypothesis))
