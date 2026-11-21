"""Comprehensive pytest suite for UzbekMWETokenizer (Algorithm 1 & Listing 1)."""

import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from UzTransliterator import UzTransliterator

import uzbek_mwe_tokenizer
from uzbek_mwe_tokenizer import UzbekMWETokenizer


def _make_lightweight_tokenizer(mode="lot", tag_sequence=None):
    """Construct an UzbekMWETokenizer instance with a lightweight deterministic mock model for fast CI."""
    tok_instance = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tok_instance.mode = mode
    tok_instance.id2label = {0: "O", 1: "B-MWE", 2: "I-MWE"}
    if hasattr(UzTransliterator, "UzTransliterator") and callable(
        getattr(UzTransliterator, "UzTransliterator")
    ):
        tok_instance.transliterator = UzTransliterator.UzTransliterator()
    else:
        tok_instance.transliterator = UzTransliterator()

    class MockBatchEncoding(dict):
        def __init__(self, word_ids_list, seq_len):
            super().__init__(input_ids=torch.zeros((1, seq_len), dtype=torch.long))
            self._word_ids = word_ids_list

        def word_ids(self):
            return self._word_ids

    class MockHFTokenizer:
        def __call__(self, words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt"):
            # Simulate [CLS] + up to (max_length - 2) subwords + [SEP]
            capped_words = words[: max_length - 2] if truncation else words
            wids = [None] + list(range(len(capped_words))) + [None]
            return MockBatchEncoding(wids, len(wids))

    class MockHFModel:
        def __call__(self, **inputs):
            seq_len = inputs["input_ids"].shape[1]
            logits = torch.zeros((1, seq_len, 3), dtype=torch.float32)
            # Default everything to 'O' with high logit
            logits[0, :, 0] = 10.0
            if tag_sequence:
                for word_idx, label_id in tag_sequence.items():
                    sub_pos = word_idx + 1  # offset for [CLS]
                    if sub_pos < seq_len - 1:
                        logits[0, sub_pos, :] = -10.0
                        logits[0, sub_pos, label_id] = 10.0
            return SimpleNamespace(logits=logits)

    tok_instance.tokenizer = MockHFTokenizer()
    tok_instance.model = MockHFModel()
    return tok_instance


def test_version_and_exports():
    assert hasattr(uzbek_mwe_tokenizer, "UzbekMWETokenizer")
    assert uzbek_mwe_tokenizer.__version__ == "0.1.0"


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        UzbekMWETokenizer(mode="eng")


def test_unicode_apostrophe_standardization():
    raw = "Natijani ko‘rib, o’quvchilarning qoʻltig`idan g´oʼya"
    normalized = UzbekMWETokenizer._normalize_apostrophes(raw)
    assert normalized == "Natijani ko'rib, o'quvchilarning qo'ltig'idan g'o'ya"


def test_punctuation_decoupled_tokenization_preserves_internal_apostrophes():
    sample_lat = "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi."
    tokens_lat = UzbekMWETokenizer._tokenize_lexical_and_punct(sample_lat)
    assert tokens_lat == ["Natijani", "ko'rib", ",", "tarvuzi", "qo'ltig'idan", "tushdi", "."]

    sample_cyr = "Қоронғи хонада капалаги учди, гўё ерга кирса."
    tokens_cyr = UzbekMWETokenizer._tokenize_lexical_and_punct(sample_cyr)
    assert tokens_cyr == ["Қоронғи", "хонада", "капалаги", "учди", ",", "гўё", "ерга", "кирса", "."]


def test_empty_and_whitespace_inputs():
    t = _make_lightweight_tokenizer(mode="lot")
    assert t.extract_mwe("") == []
    assert t.extract_mwe("   \n\t  ") == []
    assert t.extract_mwe("... !!! ???") == []
    with pytest.raises(TypeError):
        t.extract_mwe(None)


def test_punctuation_adjacent_idioms_and_buffer_flush():
    # Tokens for "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." (after cyr projection):
    # 0: Натижани, 1: кўриб, 2: ',', 3: тарвузи, 4: қўлтиғидан, 5: тушди, 6: '.'
    # Even if punctuation token 2 or 6 were given I-MWE, _is_punct must flush and exclude punctuation!
    tag_map = {1: 1, 2: 2, 3: 1, 4: 2, 5: 2, 6: 2}
    t_lot = _make_lightweight_tokenizer(mode="lot", tag_sequence=tag_map)
    res = t_lot.extract_mwe("Natijani ko‘rib, tarvuzi qo’ltigʻidan tushdi.")
    # Word 1 ("кўриб") alone before ',' has length 1 (<2), so flushed & discarded at ','!
    # Words 3, 4, 5 form "tarvuzi qo'ltig'idan tushdi" without trailing '.'!
    assert len(res) == 1
    assert res[0]["mwe"] == "tarvuzi qo'ltig'idan tushdi"
    assert isinstance(res[0]["confidence"], float)


def test_cyrillic_extraction_and_rounding():
    # "Қоронғи хонада капалаги учди, ерга кирса."
    # 0: Қоронғи, 1: хонада, 2: капалаги (B), 3: учди (I), 4: ',', 5: ерга (B), 6: кирса (I), 7: '.'
    tag_map = {2: 1, 3: 2, 5: 1, 6: 2}
    t_cyr = _make_lightweight_tokenizer(mode="cyr", tag_sequence=tag_map)
    res = t_cyr.extract_mwe("Қоронғи хонада капалаги учди, ерга кирса.")
    assert [item["mwe"] for item in res] == ["капалаги учди", "ерга кирса"]


def test_long_input_exceeding_512_tokens_does_not_crash():
    # Construct a 750-word string (> 512 subword tokens) with an MWE near the start
    tag_map = {2: 1, 3: 2, 4: 2}
    t_lot = _make_lightweight_tokenizer(mode="lot", tag_sequence=tag_map)
    long_text = "Natijani ko'rib tarvuzi qo'ltig'idan tushdi " + ("kitob o'qidi " * 380)
    res = t_lot.extract_mwe(long_text)
    assert len(res) == 1
    assert res[0]["mwe"] == "tarvuzi qo'ltig'idan tushdi"


LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "UzBERT_Cyrillic"


@pytest.mark.skipif(
    not LOCAL_MODEL_DIR.exists() and os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Requires local UzBERT_Cyrillic weights or RUN_MODEL_TESTS=1.",
)
def test_listing1_exact_empirical_outputs():
    model_ref = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT"

    tok_lot = UzbekMWETokenizer(mode="lot", model_name=model_ref)
    text_lat = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
    out_lat = tok_lot.extract_mwe(text_lat)
    assert out_lat == [{"mwe": "tarvuzi qo'ltig'idan tushdi", "confidence": 99.98}]

    tok_cyr = UzbekMWETokenizer(mode="cyr", model_name=model_ref)
    text_cyr = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
    out_cyr = tok_cyr.extract_mwe(text_cyr)
    assert out_cyr == [
        {"mwe": "капалаги учди", "confidence": 99.99},
        {"mwe": "ерга кирса", "confidence": 76.42},
        {"mwe": "қулоғидан тортиб чиқарадиган", "confidence": 98.08},
    ]
