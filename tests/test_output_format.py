"""Output-format tests that do not require model download."""

from uzbek_mwe_tokenizer.tokenizer import UzbekMWETokenizer


def test_to_slash_without_model_initialization():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tagged = [
        {"token": "Natijani", "tag": "O", "confidence": 99.9},
        {"token": "tarvuzi", "tag": "B-MWE", "confidence": 100.0},
        {"token": "qo'ltig'idan", "tag": "I-MWE", "confidence": 100.0},
        {"token": "tushdi.", "tag": "I-MWE", "confidence": 100.0},
    ]
    assert (
        tokenizer.to_slash(tagged)
        == "Natijani/O tarvuzi/B-MWE qo'ltig'idan/I-MWE tushdi./I-MWE"
    )


def test_to_conll_without_model_initialization():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tagged = [
        {"token": "tarvuzi", "tag": "B-MWE", "confidence": 100.0},
        {"token": "qo'ltig'idan", "tag": "I-MWE", "confidence": 100.0},
    ]
    assert tokenizer.to_conll(tagged) == "tarvuzi\tB-MWE\nqo'ltig'idan\tI-MWE"


def test_to_json_without_model_initialization():
    tokenizer = UzbekMWETokenizer.__new__(UzbekMWETokenizer)
    tagged = [{"token": "tarvuzi", "tag": "B-MWE", "confidence": 100.0}]
    assert tokenizer.to_json(tagged) == tagged
