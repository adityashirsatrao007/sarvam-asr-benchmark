import unittest

from asrbench.manifest import Utterance
from asrbench.providers.mock import MockProvider

HINDI = "भारत में कृत्रिम बुद्धिमत्ता के उपयोग तेज़ी से बढ़ रहे हैं।"
ENGLISH = "the model reports word error rate for each language"


def _utt(utt_id: str, reference: str, language: str = "hi") -> Utterance:
    return Utterance(
        utt_id=utt_id,
        audio_path="/nonexistent/audio.wav",
        reference=reference,
        language=language,
        domain="read-speech",
        dataset="unit-test",
    )


class MockProviderTests(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        provider = MockProvider(seed=13)
        utterance = _utt("samp-hi-01", HINDI)
        self.assertEqual(provider.transcribe(utterance), provider.transcribe(utterance))

    def test_seed_changes_output(self) -> None:
        utterance = _utt("samp-en-01", ENGLISH, language="en")
        first = MockProvider(seed=1).transcribe(utterance)
        second = MockProvider(seed=2).transcribe(utterance)
        self.assertNotEqual(first, second)

    def test_produces_non_empty_text(self) -> None:
        provider = MockProvider()
        for index, reference in enumerate([HINDI, ENGLISH], start=1):
            hypothesis = provider.transcribe(_utt(f"utt-{index}", reference))
            self.assertTrue(hypothesis.strip())

    def test_never_reads_audio_file(self) -> None:
        # The path deliberately does not exist; mock must not care.
        provider = MockProvider()
        utterance = _utt("missing-audio", ENGLISH, language="en")
        self.assertTrue(provider.transcribe(utterance))

    def test_corrupts_long_references(self) -> None:
        # With default noise rates a 10+ word reference virtually always
        # differs from its hypothesis; the assertion is about the contract
        # (noise is applied), not about a specific word count.
        provider = MockProvider(drop_rate=0.3, sub_rate=0.3, insert_rate=0.3)
        utterance = _utt("samp-en-02", ENGLISH, language="en")
        self.assertNotEqual(provider.transcribe(utterance), utterance.reference)

    def test_reference_is_not_returned_verbatim_when_noisy(self) -> None:
        provider = MockProvider(drop_rate=1.0)  # drop every token
        utterance = _utt("samp-en-03", ENGLISH, language="en")
        self.assertEqual(provider.transcribe(utterance), "")


if __name__ == "__main__":
    unittest.main()
