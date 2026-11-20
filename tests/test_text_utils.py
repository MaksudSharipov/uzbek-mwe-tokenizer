"""Text normalization tests for UzbekMWETokenizer."""

import pytest
from uzbek_mwe_tokenizer.text_utils import normalize_apostrophes, split_words


def test_apostrophe_normalization():
    raw = "o‘quvchilarning qo’ltigʻidan g`oʼya"
    assert normalize_apostrophes(raw) == "o'quvchilarning qo'ltig'idan g'o'ya"


def test_whitespace_collapse():
    raw = "  tarvuzi   qo‘ltig‘idan \n  tushdi  "
    assert normalize_apostrophes(raw) == "tarvuzi qo'ltig'idan tushdi"


def test_split_words():
    assert split_words("tarvuzi qo‘ltig‘idan tushdi") == [
        "tarvuzi",
        "qo'ltig'idan",
        "tushdi",
    ]


def test_non_string_raises_type_error():
    with pytest.raises(TypeError):
        normalize_apostrophes(123)  # type: ignore[arg-type]
