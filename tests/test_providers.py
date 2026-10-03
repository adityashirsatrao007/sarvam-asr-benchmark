"""Provider registry and Sarvam response handling — offline, no key.

The HTTP layer is swapped for a stub, so request shape, status handling and
payload parsing are all covered without a socket or an API key.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrbench.manifest import Utterance
from asrbench.providers import ProviderError, create_provider
from asrbench.providers.sarvam_api import (
    SarvamASRProvider,
    _extract_transcript,
    _load_dotenv,
)

HAVE_REQUESTS = importlib.util.find_spec("requests") is not None
ENV_KEYS = ("SARVAM_API_KEY", "SARVAM_API_BASE", "SARVAM_ASR_PATH", "SARVAM_ASR_MODEL")


class ParsingTests(unittest.TestCase):
    def test_unknown_provider_name_is_rejected(self) -> None:
        # The CLI's argparse choices hide this branch; library callers do not.
        with self.assertRaisesRegex(ProviderError, "mock, sarvam, whisper"):
            create_provider("deepgram")

    def test_extract_transcript_accepts_known_shapes(self) -> None:
        cases = [
            ("  bare string  ", "bare string"),
            ({"transcript": "  hi there "}, "hi there"),
            ({"text": "ok"}, "ok"),
            ({"result": {"output": "nested"}}, "nested"),
            ({"transcript": None, "text": "fallback"}, "fallback"),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(_extract_transcript(payload), expected)

    def test_extract_transcript_rejects_unusable_payloads(self) -> None:
        cases = [
            (None, "unexpected ASR response type"),
            (["a", "list"], "unexpected ASR response type"),
            ({"error": "quota exceeded"}, "no transcript field"),
            ({"error_code": 429, "message": "slow down"}, "no transcript field"),
        ]
        for payload, expected in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(ProviderError, re.escape(expected)),
            ):
                _extract_transcript(payload)

        # An error payload can be a whole HTML page; the message stays small.
        with self.assertRaises(ProviderError) as raised:
            _extract_transcript({"error": "x" * 10_000})
        self.assertLess(len(str(raised.exception)), 400)

    def test_load_dotenv_fills_unset_variables_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# a comment\n\nSARVAM_API_KEY=from-file\nSARVAM_ASR_PATH='/stt'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SARVAM_API_KEY": "from-export"}):
                _load_dotenv(env_file)
                # An explicit export must beat the file, or CI/CD secrets
                # would silently lose to a stray .env in the checkout.
                self.assertEqual(os.environ["SARVAM_API_KEY"], "from-export")
                self.assertEqual(os.environ["SARVAM_ASR_PATH"], "/stt")


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: object = None,
        text: str = "",
        *,
        json_raises: bool = False,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.json_raises = json_raises

    def json(self) -> object:
        if self.json_raises:
            raise ValueError("No JSON object could be decoded")
        return self.payload


class _FakeHttp:
    """Stands in for the ``requests`` module: records the call, opens no socket."""

    class RequestException(Exception):  # the exception type transcribe() catches
        pass

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return self.response


@unittest.skipUnless(HAVE_REQUESTS, "requests is not installed")
class SarvamProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        self._cwd = os.getcwd()
        self._env = dict(os.environ)
        # Empty working directory and no SARVAM_* variables: a developer's
        # real .env or exported key must never influence these assertions.
        os.chdir(self.directory)
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        os.chdir(self._cwd)
        os.environ.clear()
        os.environ.update(self._env)

    def _provider(self, **kwargs: object) -> SarvamASRProvider:
        return SarvamASRProvider(api_key="unit-test-key", **kwargs)

    def _utterance(self) -> Utterance:
        audio = self.directory / "clip.wav"
        audio.write_bytes(b"RIFF....WAVE")
        return Utterance(
            utt_id="u1",
            audio_path=str(audio),
            reference="नमस्ते दुनिया",
            language="hi",
            domain="read-speech",
            dataset="unit-test",
        )

    def test_missing_api_key_fails_with_guidance(self) -> None:
        with self.assertRaisesRegex(ProviderError, "SARVAM_API_KEY"):
            SarvamASRProvider()

    def test_dotenv_supplies_key_and_model(self) -> None:
        (self.directory / ".env").write_text(
            "# copy of .env.example\nSARVAM_API_KEY='file-key'\n"
            "SARVAM_ASR_MODEL=saarika:v3\n",
            encoding="utf-8",
        )
        provider = SarvamASRProvider()
        self.assertEqual(provider.api_key, "file-key")
        self.assertEqual(provider.model, "saarika:v3")

    def test_transcribe_posts_the_expected_request(self) -> None:
        provider = self._provider()
        fake = _FakeHttp(_FakeResponse(payload={"transcript": "  नमस्ते दुनिया  "}))
        provider._http = fake

        text = provider.transcribe(self._utterance())

        self.assertEqual(text, "नमस्ते दुनिया")
        url, kwargs = fake.calls[0]
        self.assertTrue(url.endswith("/speech-to-text"))
        self.assertEqual(kwargs["headers"], {"api-subscription-key": "unit-test-key"})
        self.assertEqual(kwargs["files"]["file"][0], "clip.wav")
        # Pins the [verify]-flagged request shape: manifest "hi" -> "hi-IN".
        self.assertEqual(
            kwargs["data"], {"model": "saaras:v3", "language_code": "hi-IN"}
        )

    def test_transcribe_reports_http_error_status(self) -> None:
        provider = self._provider()
        provider._http = _FakeHttp(_FakeResponse(status_code=429, text="rate limited"))
        with self.assertRaisesRegex(ProviderError, "HTTP 429"):
            provider.transcribe(self._utterance())

    def test_transcribe_reports_a_non_json_body(self) -> None:
        provider = self._provider()
        provider._http = _FakeHttp(
            _FakeResponse(text="<html>gateway timeout</html>", json_raises=True)
        )
        with self.assertRaisesRegex(ProviderError, "not valid JSON"):
            provider.transcribe(self._utterance())


if __name__ == "__main__":
    unittest.main()
