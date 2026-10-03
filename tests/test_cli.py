import csv
import tempfile
import unittest
from pathlib import Path

from asrbench.cli import main
from asrbench.manifest import load_manifest
from asrbench.report import aggregate, read_results_csv, render_markdown, score
from asrbench.manifest import Utterance

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "sample" / "manifest.csv"


class CliTests(unittest.TestCase):
    def test_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            exit_code = main(
                [
                    "run",
                    "--manifest",
                    str(MANIFEST),
                    "--provider",
                    "mock",
                    "--out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(exit_code, 0)
            results_csv = out / "results.csv"
            report_md = out / "report.md"
            self.assertTrue(results_csv.exists())
            self.assertTrue(report_md.exists())

            with results_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(load_manifest(MANIFEST)))
            self.assertIn("|", report_md.read_text(encoding="utf-8"))

    def test_language_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            exit_code = main(
                [
                    "run",
                    "--manifest",
                    str(MANIFEST),
                    "--languages",
                    "hi",
                    "--out",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(exit_code, 0)
            results = read_results_csv(out / "results.csv")
            self.assertTrue(results)
            self.assertTrue(all(row.language == "hi" for row in results))

    def test_report_command_rerenders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(
                main(
                    [
                        "run",
                        "--manifest",
                        str(MANIFEST),
                        "--out",
                        str(out),
                        "--quiet",
                    ]
                ),
                0,
            )
            exit_code = main(["report", "--results", str(out / "results.csv")])
            self.assertEqual(exit_code, 0)

    def test_missing_manifest_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(
                ["run", "--manifest", f"{tmp}/nope.csv", "--out", tmp, "--quiet"]
            )
        self.assertEqual(exit_code, 1)

    def test_filter_with_no_matches_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(
                [
                    "run",
                    "--manifest",
                    str(MANIFEST),
                    "--languages",
                    "xx",
                    "--out",
                    tmp,
                    "--quiet",
                ]
            )
        self.assertEqual(exit_code, 2)


class ReportTests(unittest.TestCase):
    def test_aggregate_includes_overall_row(self) -> None:
        utterances = load_manifest(MANIFEST)
        results = [score(utt, utt.reference, "mock") for utt in utterances]
        rows = aggregate(results)
        self.assertEqual(rows[-1].language, "all")
        self.assertEqual(rows[-1].utterances, len(utterances))

    def test_perfect_hypotheses_score_zero(self) -> None:
        utterance = Utterance(
            utt_id="perfect",
            audio_path="/x.wav",
            reference="एक दो तीन",
            language="hi",
            domain="read-speech",
            dataset="unit-test",
        )
        result = score(utterance, utterance.reference, "mock")
        self.assertEqual(result.word_edits, 0)
        self.assertEqual(result.char_edits, 0)
        self.assertEqual(aggregate([result])[-1].wer, 0.0)

    def test_markdown_table_shape(self) -> None:
        utterance = Utterance(
            utt_id="one",
            audio_path="/x.wav",
            reference="a b c d",
            language="en",
            domain="read-speech",
            dataset="unit-test",
        )
        markdown = render_markdown(aggregate([score(utterance, "a b x d", "mock")]))
        self.assertIn("| Language | Domain | Provider |", markdown)
        self.assertIn("| en | read-speech | mock | 1 | 4 | 0.250 |", markdown)


if __name__ == "__main__":
    unittest.main()
