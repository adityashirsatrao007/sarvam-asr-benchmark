import unittest

from asrbench.metrics import char_error_rate, edit_distance, word_error_rate


class EditDistanceTests(unittest.TestCase):
    def test_identical_sequences(self) -> None:
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "b", "c"]), 0)

    def test_single_substitution(self) -> None:
        self.assertEqual(edit_distance(["a", "b", "c", "d"], ["a", "x", "c", "d"]), 1)

    def test_deletion(self) -> None:
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "c"]), 1)

    def test_insertion(self) -> None:
        self.assertEqual(edit_distance(["a", "c"], ["a", "b", "c"]), 1)

    def test_empty_reference(self) -> None:
        self.assertEqual(edit_distance([], ["a", "b"]), 2)

    def test_empty_hypothesis(self) -> None:
        self.assertEqual(edit_distance(["a", "b"], []), 2)


class ErrorRateTests(unittest.TestCase):
    def test_perfect_transcript(self) -> None:
        self.assertEqual(word_error_rate("the cat sat", "the cat sat"), 0.0)

    def test_one_substitution_in_four_words(self) -> None:
        self.assertAlmostEqual(
            word_error_rate("one two three four", "one two X four"), 0.25
        )

    def test_insertions_can_exceed_one(self) -> None:
        self.assertAlmostEqual(word_error_rate("a b", "a x y z b"), 1.5)

    def test_empty_reference_with_hypothesis(self) -> None:
        self.assertEqual(word_error_rate("", "hello"), 1.0)

    def test_empty_reference_and_hypothesis(self) -> None:
        self.assertEqual(word_error_rate("", ""), 0.0)

    def test_devanagari_perfect_match(self) -> None:
        text = "भारत में कृत्रिम बुद्धिमत्ता के उपयोग बढ़ रहे हैं"
        self.assertEqual(word_error_rate(text, text), 0.0)

    def test_devanagari_one_word_error(self) -> None:
        reference = "सोलापूर शहरातील विद्यार्थ्यांसाठी नवीन तंत्रज्ञान आहे"
        hypothesis = "सोलापूर शहरातील विद्यार्थ्यांसाठी नवीन तंत्रज्ञान नाही"
        self.assertAlmostEqual(word_error_rate(reference, hypothesis), 1 / 6)

    def test_punctuation_and_case_are_normalised(self) -> None:
        self.assertEqual(word_error_rate("The Cat Sat.", "the cat sat"), 0.0)

    def test_char_error_rate_counts_characters(self) -> None:
        self.assertAlmostEqual(char_error_rate("abcd", "abxd"), 0.25)


if __name__ == "__main__":
    unittest.main()
