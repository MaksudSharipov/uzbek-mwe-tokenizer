"""Label map tests for UzbekMWETokenizer."""

from uzbek_mwe_tokenizer.label_map import (
    ID2LABEL_DEFAULT,
    LABEL2ID_DEFAULT,
    MWE_LABEL_DESCRIPTIONS,
    MWE_LABELS,
)


def test_mwe_labels_complete():
    assert MWE_LABELS == ["O", "B-MWE", "I-MWE"]
    for label in MWE_LABELS:
        assert label in MWE_LABEL_DESCRIPTIONS
        assert label in LABEL2ID_DEFAULT


def test_id2label_and_label2id_bijection():
    for idx, label in ID2LABEL_DEFAULT.items():
        assert LABEL2ID_DEFAULT[label] == idx
