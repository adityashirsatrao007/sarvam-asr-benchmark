"""Text normalisation shared by every metric.

Normalisation must be *language-neutral*: it has to work for Devanagari
(Hindi, Marathi), Latin script and code-mixed strings without a language
detector, so we only do things that are safe for all of them:

* Unicode NFKC (full-width -> ASCII, composed -> decomposed where relevant)
* casefold (no-op for Devanagari, useful for English)
* punctuation/symbols -> space, apostrophes deleted inside words
* whitespace collapse

Crucially we classify by Unicode *category* rather than using a ``\\w``
regex: Python's ``\\w`` excludes combining marks (category ``Mn``), so a
regex like ``[^\\w\\s]`` silently destroys Devanagari matras and viramas
(``नमस्ते`` -> ``नमस त``), which would corrupt every Hindi/Marathi score.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+", re.UNICODE)

# Deleted outright (English contractions: don't -> dont, not "don t").
_JOINED_APOSTROPHES = "'\u2019"


def _replace(character: str) -> str:
    """Apostrophes vanish; other punctuation/symbols become a space."""
    if character in _JOINED_APOSTROPHES:
        return ""
    return " " if unicodedata.category(character)[0] in {"P", "S"} else character


def normalize(text: str) -> str:
    """Return the canonical form of *text* used for scoring."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(_replace(character) for character in text)
    return _WS.sub(" ", text).strip()


def word_tokens(text: str) -> list[str]:
    """Whitespace word tokens of the normalised string."""
    normalised = normalize(text)
    return normalised.split() if normalised else []


def char_tokens(text: str) -> list[str]:
    """Character tokens for CER, with spaces removed (industry practice)."""
    return list(normalize(text).replace(" ", ""))
