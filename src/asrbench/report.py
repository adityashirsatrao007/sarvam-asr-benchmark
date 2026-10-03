"""Scoring, aggregation and report rendering.

We aggregate with the *micro* average (total edits / total reference tokens),
which is the standard way to report corpus WER; a per-language/domain row and
an overall row are both emitted. It is deliberately not the mean of the
per-utterance WERs: that would give a 4-word clip the same vote as a
400-word one, so a handful of very short utterances could swing the headline
number. Per-utterance results are therefore stored as *edit counts*, and only
aggregation turns them into rates.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .manifest import Utterance
from .metrics import edit_distance
from .textnorm import char_tokens, word_tokens

CSV_FIELDS = [
    "utt_id",
    "language",
    "domain",
    "dataset",
    "provider",
    "reference",
    "hypothesis",
    "ref_words",
    "ref_chars",
    "word_edits",
    "char_edits",
]


@dataclass(frozen=True)
class UtteranceResult:
    utt_id: str
    language: str
    domain: str
    dataset: str
    provider: str
    reference: str
    hypothesis: str
    ref_words: int
    ref_chars: int
    word_edits: int
    char_edits: int


def score(utterance: Utterance, hypothesis: str, provider: str) -> UtteranceResult:
    """Turn one (utterance, hypothesis) pair into an auditable result row.

    Stores the raw strings plus edit *counts* — never a per-utterance rate:
    the denominator only exists once utterances are pooled, and counts let
    anyone recompute the published WER from ``results.csv`` by hand.
    """
    ref_words = word_tokens(utterance.reference)
    ref_chars = char_tokens(utterance.reference)
    hyp_words = word_tokens(hypothesis)
    hyp_chars = char_tokens(hypothesis)
    return UtteranceResult(
        utt_id=utterance.utt_id,
        language=utterance.language,
        domain=utterance.domain,
        dataset=utterance.dataset,
        provider=provider,
        reference=utterance.reference,
        hypothesis=hypothesis,
        ref_words=len(ref_words),
        ref_chars=len(ref_chars),
        word_edits=edit_distance(ref_words, hyp_words),
        char_edits=edit_distance(ref_chars, hyp_chars),
    )


@dataclass(frozen=True)
class AggregateRow:
    language: str
    domain: str
    provider: str
    utterances: int
    words: int
    wer: float
    cer: float


def aggregate(results: list[UtteranceResult]) -> list[AggregateRow]:
    """Micro-averaged rows grouped by (language, domain, provider) + overall."""
    groups: dict[tuple[str, str, str], list[UtteranceResult]] = {}
    for result in results:
        key = (result.language, result.domain, result.provider)
        groups.setdefault(key, []).append(result)

    rows = [
        _group_to_row(language, domain, provider, items)
        for (language, domain, provider), items in sorted(groups.items())
    ]
    if results:
        # A run produces exactly one provider, but a results.csv that was
        # concatenated by hand can hold several: label the overall row
        # "mixed" rather than quietly crediting the first row's provider.
        providers = {result.provider for result in results}
        overall_provider = providers.pop() if len(providers) == 1 else "mixed"
        rows.append(_group_to_row("all", "all", overall_provider, results))
    return rows


def _group_to_row(
    language: str, domain: str, provider: str, items: list[UtteranceResult]
) -> AggregateRow:
    words = sum(item.ref_words for item in items)
    chars = sum(item.ref_chars for item in items)
    word_edits = sum(item.word_edits for item in items)
    char_edits = sum(item.char_edits for item in items)
    # A group with zero reference tokens has no denominator, so the rate is
    # undefined rather than zero; 0.0 keeps the column numeric and the Words
    # column shows the group has no scorable text. Normal loading makes this
    # unreachable — see the reference check in manifest.load_manifest().
    return AggregateRow(
        language=language,
        domain=domain,
        provider=provider,
        utterances=len(items),
        words=words,
        wer=word_edits / words if words else 0.0,
        cer=char_edits / chars if chars else 0.0,
    )


def render_markdown(rows: list[AggregateRow], *, title: str = "ASR benchmark") -> str:
    lines = [
        f"## {title}",
        "",
        "| Language | Domain | Provider | Utterances | Words | WER | CER |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.language} | {row.domain} | {row.provider} | "
            f"{row.utterances} | {row.words} | {row.wer:.3f} | {row.cer:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_results_csv(results: list[UtteranceResult], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return destination


class ResultsError(ValueError):
    """Raised when a saved results CSV is missing or malformed."""


def read_results_csv(path: str | Path) -> list[UtteranceResult]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"results file not found: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in CSV_FIELDS if column not in header]
        if missing:
            raise ResultsError(
                f"{source} is missing column(s): {', '.join(missing)} "
                "(expected an asrbench results file, e.g. results/results.csv)"
            )
        try:
            return [
                UtteranceResult(
                    utt_id=row["utt_id"],
                    language=row["language"],
                    domain=row["domain"],
                    dataset=row["dataset"],
                    provider=row["provider"],
                    reference=row["reference"],
                    hypothesis=row["hypothesis"],
                    ref_words=int(row["ref_words"]),
                    ref_chars=int(row["ref_chars"]),
                    word_edits=int(row["word_edits"]),
                    char_edits=int(row["char_edits"]),
                )
                for row in reader
            ]
        except (TypeError, ValueError) as exc:
            # Short row (None cells) or a count column that is not an int —
            # report it as a data problem, not a traceback.
            raise ResultsError(f"{source} has a malformed row: {exc}") from exc


def write_report(rows: list[AggregateRow], path: str | Path, *, title: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(rows, title=title), encoding="utf-8")
    return destination


# Re-exported for callers that only import ``report``.
__all__ = [
    "AggregateRow",
    "ResultsError",
    "UtteranceResult",
    "aggregate",
    "read_results_csv",
    "render_markdown",
    "score",
    "write_report",
    "write_results_csv",
]
