"""Unit tests for UzbekMWETokenizer (PyPI v0.1.0 API)."""

import pytest
from UzTransliterator import UzTransliterator
import uzbek_mwe_tokenizer
from uzbek_mwe_tokenizer import UzbekMWETokenizer


def test_package_imports():
    assert hasattr(uzbek_mwe_tokenizer, "UzbekMWETokenizer")


def test_prepare_text_apostrophe_normalization_and_transliteration():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tokenizer.mode = "lot"
    tokenizer.transliterator = UzTransliterator.UzTransliterator()

    prepared = tokenizer._prepare_text("tarvuzi qo‘ltig’idan tushdi")
    assert "’" not in prepared
    assert "‘" not in prepared


def test_format_output_text_cyr_and_lot():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tokenizer.transliterator = UzTransliterator.UzTransliterator()

    tokenizer.mode = "cyr"
    assert tokenizer._format_output_text("тарвузи қўлтиғидан тушди") == "тарвузи қўлтиғидан тушди"

    tokenizer.mode = "lot"
    restored = tokenizer._format_output_text("тарвузи қўлтиғидан тушди")
    assert "tarvuzi" in restored.lower()


def test_empty_input_returns_empty_list():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    assert tokenizer.extract_mwe("   ") == []
