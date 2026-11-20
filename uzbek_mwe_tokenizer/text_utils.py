"""Text normalization and cross-script utilities for UzbekMWETokenizer."""

from __future__ import annotations

import re
from typing import List


APOSTROPHE_MAP = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "ʼ": "'",
    "ʻ": "'",
}


def normalize_apostrophes(text: str) -> str:
    """Standardize common apostrophe variants in Uzbek Latin text to ASCII quote `'`.

    Parameters
    ----------
    text:
        Raw input string in Latin or Cyrillic script.

    Returns
    -------
    str
        Normalized string with uniform apostrophes and collapsed whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text
    for old, new in APOSTROPHE_MAP.items():
        normalized = normalized.replace(old, new)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def split_words(text: str) -> List[str]:
    """Split normalized text into whitespace-delimited orthographic words."""
    normalized = normalize_apostrophes(text)
    if not normalized:
        return []
    return normalized.split()
