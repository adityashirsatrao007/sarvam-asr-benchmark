# data/real — bundled real-speech evaluation clips

16 clips fetched **2026-10-03** from the Hugging Face datasets-server and
converted to 16 kHz mono 16-bit WAV with ffmpeg (`-ar 16000 -ac 1
-c:a pcm_s16le`). Audio is committed so the README's live numbers reproduce
without re-downloading anything; `manifest.csv` is authoritative.

| Rows | Source | Config / split | Offsets | Reference field |
| --- | --- | --- | --- | --- |
| `fleurs-hi-00` … `fleurs-hi-05` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | `hi_in` / `validation` | 0–5 | `transcription` |
| `libri-en-00` … `libri-en-09` | [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) (LibriSpeech `dev-clean`) | `clean` / `validation` | 0–9 | `text` |

Exact fetch endpoints:

```text
https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs&config=hi_in&split=validation&offset=0&length=6
https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing%2Flibrispeech_asr_dummy&config=clean&split=validation&offset=0&length=10
```

Each row's `audio[0].src` (a datasets-server cached-asset URL) was
downloaded and normalised as above.

**Licence / attribution:** FLEURS and LibriSpeech are distributed under
CC-BY 4.0 — cite the dataset pages above when reusing these clips or the
numbers derived from them. This is a convenience sample of 16 read-speech
clips chosen by fixed row offsets, not a random or representative draw;
treat resulting metrics as integration evidence, not a benchmark ranking.

**Not included:** Marathi. FLEURS `mr_in` row groups exceed the
datasets-server 300 MB scan limit and Common Voice downloads are gated, so
no honestly-sourced Marathi eval audio is bundled yet.
