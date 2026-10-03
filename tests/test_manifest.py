"""Manifest loading: path resolution, defaults and validation messages."""

import re
import tempfile
import unittest
from pathlib import Path

from asrbench.manifest import ManifestError, load_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def _write(self, body: str) -> Path:
        path = self.directory / "manifest.csv"
        path.write_text(body, encoding="utf-8")
        return path

    def test_relative_audio_path_resolves_against_the_manifest(self) -> None:
        path = self._write("utt_id,audio_path,reference\nu1,clips/u1.wav,hello world\n")
        [utterance] = load_manifest(path)
        self.assertEqual(utterance.audio_path, str(self.directory / "clips" / "u1.wav"))
        # Columns the manifest does not carry fall back to "unknown".
        self.assertEqual(utterance.language, "unknown")
        self.assertEqual(utterance.domain, "unknown")
        self.assertEqual(utterance.dataset, "unknown")

    def test_optional_columns_are_read_when_present(self) -> None:
        path = self._write(
            "utt_id,audio_path,language,domain,dataset,reference\n"
            "u1,/abs/u1.wav,hi,telephony,sample,नमस्ते दुनिया\n"
        )
        [utterance] = load_manifest(path)
        self.assertEqual(utterance.language, "hi")
        self.assertEqual(utterance.domain, "telephony")
        self.assertEqual(utterance.dataset, "sample")
        self.assertEqual(utterance.audio_path, "/abs/u1.wav")  # absolute: untouched

    def test_malformed_manifests_name_the_problem(self) -> None:
        # Each case pins the exact diagnostic: a user must be told which file,
        # which line and which field is wrong, without reading the source.
        empty_field = "utt_id,audio_path,reference\nu1,,hello world\n"
        cases = [
            ("utt_id,audio_path\nu1,a.wav\n", "missing column"),
            (empty_field, ":2 has an empty required field"),
            ("utt_id,audio_path,reference\n", "contains no rows"),
        ]
        for body, expected in cases:
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(ManifestError, re.escape(expected)),
            ):
                load_manifest(self._write(body))

    def test_reference_with_no_scoreable_text_is_rejected(self) -> None:
        # "..." is a non-empty string but yields zero reference words: WER
        # would have no denominator and would report 0.000 for real errors.
        path = self._write("utt_id,audio_path,reference\nu1,a.wav, । \n")
        with self.assertRaisesRegex(ManifestError, "no scoreable text"):
            load_manifest(path)

    def test_manifest_exported_with_a_bom_still_loads(self) -> None:
        path = self.directory / "manifest.csv"
        path.write_bytes(
            "\ufeffutt_id,audio_path,reference\nu1,a.wav,hello world\n".encode("utf-8")
        )
        [utterance] = load_manifest(path)
        self.assertEqual(utterance.utt_id, "u1")


if __name__ == "__main__":
    unittest.main()
