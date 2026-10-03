import unittest

from asrbench.textnorm import char_tokens, normalize, word_tokens


class NormalizeTests(unittest.TestCase):
    def test_casefold(self) -> None:
        self.assertEqual(normalize("The Model"), "the model")

    def test_punctuation_removed(self) -> None:
        self.assertEqual(normalize("hello, world!"), "hello world")

    def test_devanagari_danda_removed(self) -> None:
        self.assertEqual(normalize("नमस्ते।"), "नमस्ते")

    def test_devanagari_matras_and_viramas_preserved(self) -> None:
        # Regression: a [^\w\s] punctuation regex strips category-Mn marks
        # and silently corrupts Hindi/Marathi tokens.
        for word in ("नमस्ते", "विद्यार्थ्यांसाठी", "भारतीय"):
            self.assertEqual(normalize(word), word)

    def test_whitespace_collapsed(self) -> None:
        self.assertEqual(normalize("a \t  b\n c"), "a b c")

    def test_fullwidth_form_nfkc(self) -> None:
        self.assertEqual(normalize("１２３"), "123")

    def test_none_is_empty(self) -> None:
        # Providers and CSV cells can hand us None; scoring must not explode.
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize(""), "")

    def test_word_tokens(self) -> None:
        self.assertEqual(word_tokens("one  two-three"), ["one", "two", "three"])

    def test_word_tokens_empty(self) -> None:
        self.assertEqual(word_tokens("!!!"), [])

    def test_char_tokens_drop_spaces(self) -> None:
        self.assertEqual(char_tokens("a b c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
