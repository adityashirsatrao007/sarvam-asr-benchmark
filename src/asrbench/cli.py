"""Command line interface: ``python3 -m asrbench {run,report}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manifest import ManifestError, load_manifest
from .providers import ProviderError, create_provider
from .report import (
    ResultsError,
    aggregate,
    read_results_csv,
    render_markdown,
    score,
    write_report,
    write_results_csv,
)

DEFAULT_MANIFEST = Path("data/sample/manifest.csv")
DEFAULT_OUT = Path("results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asrbench",
        description="Reproducible ASR evaluation for Indian-language speech.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="transcribe a manifest and score it")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument(
        "--provider",
        choices=("mock", "sarvam", "whisper"),
        default="mock",
        help="mock = offline pipeline check (default); sarvam/whisper = real ASR",
    )
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run.add_argument(
        "--languages",
        default="",
        help="comma-separated filter, e.g. --languages hi,mr",
    )
    run.add_argument("--limit", type=int, default=0, help="0 = no limit")
    run.add_argument("--model", default=None, help="provider model override")
    run.add_argument("--seed", type=int, default=13, help="mock provider seed")
    run.add_argument("--quiet", action="store_true")

    report = subparsers.add_parser(
        "report", help="re-render the markdown report from saved results"
    )
    report.add_argument("--results", type=Path, default=DEFAULT_OUT / "results.csv")
    report.add_argument("--out", type=Path, default=DEFAULT_OUT / "report.md")
    report.add_argument("--title", default="ASR benchmark")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    utterances = load_manifest(args.manifest)

    wanted = {code.strip() for code in args.languages.split(",") if code.strip()}
    if wanted:
        utterances = [utt for utt in utterances if utt.language in wanted]
    if args.limit > 0:
        utterances = utterances[: args.limit]
    if not utterances:
        print("no utterances matched the given filters", file=sys.stderr)
        return 2

    provider = create_provider(args.provider, model=args.model, seed=args.seed)
    results = []
    for index, utterance in enumerate(utterances, start=1):
        hypothesis = provider.transcribe(utterance)
        results.append(score(utterance, hypothesis, provider.name))
        if not args.quiet:
            print(
                f"[{index}/{len(utterances)}] {utterance.utt_id} "
                f"({utterance.language}) ok",
                file=sys.stderr,
            )
    provider.close()

    csv_path = write_results_csv(results, args.out / "results.csv")
    rows = aggregate(results)
    title = f"ASR benchmark — provider: {provider.name}"
    report_path = write_report(rows, args.out / "report.md", title=title)

    print(render_markdown(rows, title=title))
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    if provider.name == "mock":
        print(
            "\nNOTE: 'mock' numbers are SYNTHETIC (a wiring check, not ASR "
            "results). Re-run with --provider sarvam or --provider whisper "
            "for real measurements.",
            file=sys.stderr,
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results = read_results_csv(args.results)
    if not results:
        print(f"{args.results} contains no rows", file=sys.stderr)
        return 2
    rows = aggregate(results)
    report_path = write_report(rows, args.out, title=args.title)
    print(render_markdown(rows, title=args.title))
    print(f"wrote {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        return _cmd_report(args)
    except (ManifestError, ResultsError, ProviderError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
