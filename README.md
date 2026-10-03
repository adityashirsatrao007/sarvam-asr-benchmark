# sarvam-asr-benchmark

A small, reproducible **ASR evaluation harness for Indian-language speech** —
micro-averaged **WER / CER broken down by language × domain**, with pluggable
transcription providers (offline mock, Sarvam AI platform API, local
Whisper) and zero required dependencies.

Built because Sarvam AI published *"Evaluating Indian Language ASR"* and
hires for evaluation work: this repo is a working answer to the question
*"how would you measure whether an ASR model is good for Marathi
code-mixed speech?"* — with the harness, the metric definitions and the
reporting already in place.

---

## Quickstart (no install, no network, no API key)

```bash
python3 scripts/make_sample_audio.py      # placeholder clips for the sample set
PYTHONPATH=src python3 -m unittest discover -s tests -v   # 39 tests
PYTHONPATH=src python3 -m asrbench run --out results --quiet
```

That last command prints (real output, captured from this repo):

```text
## ASR benchmark — provider: mock

| Language | Domain | Provider | Utterances | Words | WER | CER |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| en | broadcast | mock | 1 | 8 | 0.375 | 0.383 |
| en | code-mixed | mock | 1 | 8 | 0.375 | 0.312 |
| en | conversational | mock | 1 | 9 | 0.222 | 0.179 |
| en | read-speech | mock | 1 | 10 | 0.100 | 0.082 |
| hi | broadcast | mock | 1 | 8 | 0.375 | 0.250 |
| hi | code-mixed | mock | 1 | 10 | 0.100 | 0.025 |
| hi | conversational | mock | 1 | 10 | 0.200 | 0.121 |
| hi | read-speech | mock | 2 | 22 | 0.182 | 0.057 |
| hi | telephony | mock | 1 | 8 | 0.250 | 0.219 |
| mr | code-mixed | mock | 1 | 9 | 0.111 | 0.054 |
| mr | conversational | mock | 1 | 7 | 0.429 | 0.275 |
| mr | read-speech | mock | 2 | 14 | 0.429 | 0.172 |
| all | all | mock | 14 | 123 | 0.252 | 0.162 |

wrote results/results.csv
wrote results/report.md
```

> **These numbers are synthetic.** `mock` never opens an audio file — it
> applies a seeded corruption to each reference transcript to exercise the
> pipeline end-to-end without keys or GPUs. Real measurements come from the
> `sarvam` and `whisper` providers below. The table above exists to prove the
> harness runs, not to claim anything about model quality.

---

## Providers

| `--provider` | What it does | Requirements |
| --- | --- | --- |
| `mock` (default) | Seeded, deterministic transcript corruption — pipeline check | nothing (stdlib only) |
| `sarvam` | Sarvam AI platform ASR (`saarika` model) | `SARVAM_API_KEY` + `pip install requests` |
| `whisper` | Local `faster-whisper` baseline for comparison | `pip install faster-whisper` |

```bash
# real run against the Sarvam platform
cp .env.example .env      # fill in SARVAM_API_KEY  (never committed)
PYTHONPATH=src python3 -m asrbench run --provider sarvam --languages hi,mr --out results/sarvam

# comparison arm
PYTHONPATH=src python3 -m asrbench run --provider whisper --model small --out results/whisper
```

The Sarvam endpoint path, form fields and language-tag format are marked
`[verify]` in `src/asrbench/providers/sarvam_api.py` and are overridable via
`SARVAM_API_BASE` / `SARVAM_ASR_PATH` — check them against
`https://api.sarvam.ai/docs` before trusting a real number. **No key was used
while building this repo.**

## Getting real data

The bundled 14-utterance sample set (Hindi / Marathi / English across
read-speech, conversational, code-mixed, broadcast, telephony domains) is
*illustrative* — its audio files are placeholder tones. For a real benchmark:

1. Download labelled speech: [Common Voice](https://commonvoice.mozilla.org/)
   (`hi`, `mr`) or [FLEURS](https://huggingface.co/datasets/google/fleurs).
2. Export a manifest in the same shape — one row per clip:
   `utt_id,audio_path,language,domain,dataset,reference`
3. Run the same command with `--manifest /path/to/manifest.csv`.

Nothing else changes: metrics, aggregation and reporting are
dataset-agnostic.

## Metric definitions

- **WER** = `edit_distance(ref_words, hyp_words) / len(ref_words)` — Levenshtein
  at word level; **can exceed 1.0** when the hypothesis contains insertions.
  Corpus numbers are **micro-averaged** (Σedits / Σreference tokens), the
  standard for ASR reporting.
- **CER** = same, at character level with spaces removed (comparable across
  languages that don't always space-separate words).
- **Normalisation** is Unicode-category based, deliberately: Python's `\w`
  regex excludes combining marks, so the usual `[^\w\s]` "strip punctuation"
  trick silently deletes Devanagari matras and viramas
  (`नमस्ते` → `नमस त`), corrupting every Hindi/Marathi score. A regression
  test (`test_devanagari_matras_and_viramas_preserved`) pins this down.

## Repository layout

```text
data/sample/manifest.csv     14 labelled utterances (hi/mr/en, 5 domains)
scripts/make_sample_audio.py placeholder clip generator
src/asrbench/
  textnorm.py                category-based normalisation
  metrics.py                 WER / CER / edit distance (stdlib)
  manifest.py                CSV manifest loading + validation
  report.py                  micro-aggregation, CSV + markdown reports
  cli.py                     `python3 -m asrbench {run,report}`
  providers/                 mock · sarvam API · local whisper
tests/                       39 unittest tests (offline)
results/                     generated (gitignored)
```

## Limits, honestly

- Sample set is 14 utterances of placeholder audio — it proves the *harness*,
  not model quality.
- No real API numbers yet (no key was available while building).
- `report` groups by language × domain only; speaker/gender stratification
  would need manifest columns.
- The Sarvam API request shape is `[verify]`-flagged against live docs.

## Roadmap

- [ ] Real Common Voice / FLEURS manifests + first genuine Sarvam-vs-Whisper table
- [ ] Confidence intervals (bootstrap) on WER
- [ ] Code-mixed slice metrics (word-level script detection: Devanagari vs Latin)
- [ ] Sign-error / latency columns for the platform API

## License

MIT — see [LICENSE](LICENSE).
