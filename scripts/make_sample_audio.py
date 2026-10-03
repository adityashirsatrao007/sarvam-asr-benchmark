#!/usr/bin/env python3
"""Generate tiny placeholder WAV files for every row in a manifest.

The generated audio is a short low-volume tone, NOT speech. It exists so:

* relative paths in ``data/sample/manifest.csv`` resolve,
* providers that open files fail for the *right* reason (no speech in them).

Real benchmark numbers require real recordings — see the README section
"Getting real data" (Common Voice / FLEURS / your own labelled clips).

Usage:  python3 scripts/make_sample_audio.py [manifest.csv]
"""

from __future__ import annotations

import csv
import math
import struct
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
DURATION_S = 0.75
FREQUENCY_HZ = 330.0
AMPLITUDE = 0.05  # quiet on purpose: these clips are placeholders


def write_tone(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_samples = int(SAMPLE_RATE * DURATION_S)
    frames = bytearray()
    for index in range(total_samples):
        value = AMPLITUDE * math.sin(2.0 * math.pi * FREQUENCY_HZ * index / SAMPLE_RATE)
        frames += struct.pack("<h", int(value * 32767))
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(frames))


def main(argv: list[str]) -> int:
    manifest = Path(argv[1]) if len(argv) > 1 else Path("data/sample/manifest.csv")
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 1

    written = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            write_tone(audio_path)
            written += 1
            print(f"wrote {audio_path}")
    print(f"{written} placeholder clip(s) generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
