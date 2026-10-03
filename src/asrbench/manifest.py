"""Manifest loading: one CSV row == one evaluation utterance.

Required columns: ``utt_id,audio_path,reference``
Optional columns: ``language,domain,dataset`` (default to ``unknown``).

Relative ``audio_path`` values are resolved against the manifest's own
directory so a manifest stays portable inside its dataset folder.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def audio_exists(self) -> bool:
        return Path(self.audio_path).exists()


class ManifestError(ValueError):
    """Raised when a manifest is missing or malformed."""


def load_manifest(path: str | Path) -> list[Utterance]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")

    base_dir = manifest_path.parent
    utterances: list[Utterance] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
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
