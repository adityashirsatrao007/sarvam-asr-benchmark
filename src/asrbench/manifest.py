"""Manifest loading: one CSV row == one evaluation utterance.

Required columns: ``utt_id,audio_path,reference``
Optional columns: ``language,domain,dataset`` (default to ``unknown``).

Relative ``audio_path`` values are resolved against the manifest's own
directory so a manifest stays portable inside its dataset folder.

Loading is strict on purpose: a row that cannot be scored (missing field,
reference with no text behind it) aborts the run with the offending line
number, because a dataset that loads "successfully" and then reports a
silently wrong WER is worse than one that refuses to start.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .textnorm import normalize

REQUIRED_COLUMNS = ("utt_id", "audio_path", "reference")
OPTIONAL_DEFAULTS = {"language": "unknown", "domain": "unknown", "dataset": "unknown"}


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: str
    reference: str
    language: str = "unknown"
    domain: str = "unknown"
    dataset: str = "unknown"


class ManifestError(ValueError):
    """Raised when a manifest is missing or malformed."""


def load_manifest(path: str | Path) -> list[Utterance]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")

    base_dir = manifest_path.parent
    utterances: list[Utterance] = []
    # utf-8-sig, not utf-8: a manifest exported from a spreadsheet arrives
    # with a BOM, and the BOM would otherwise glue itself onto "utt_id" and
    # make every column check fail on a perfectly good file.
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ManifestError(
                f"manifest {manifest_path} is missing column(s): {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}
            if not all(values.values()):
                raise ManifestError(
                    f"{manifest_path}:{line_number} has an empty required field"
                )
            # "..." or a lone danda is non-empty but has nothing to align
            # against: its reference-word count is 0, so any errors it
            # produces would be divided by zero downstream and reported as a
            # perfect score. Reject it here, where we still know the row.
            if not normalize(values["reference"]):
                raise ManifestError(
                    f"{manifest_path}:{line_number} reference "
                    f"{values['reference']!r} contains no scoreable text"
                )
            audio_path = Path(values["audio_path"])
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path
            utterances.append(
                Utterance(
                    utt_id=values["utt_id"],
                    audio_path=str(audio_path),
                    reference=values["reference"],
                    **{
                        key: (row.get(key) or "").strip() or default
                        for key, default in OPTIONAL_DEFAULTS.items()
                    },
                )
            )

    if not utterances:
        raise ManifestError(f"manifest {manifest_path} contains no rows")
    return utterances
