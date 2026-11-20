"""BIO label definitions for UzbekMWETokenizer."""

MWE_LABELS = [
    "O",
    "B-MWE",
    "I-MWE",
]

MWE_LABEL_DESCRIPTIONS = {
    "O": "Outside any multi-word expression",
    "B-MWE": "Beginning token of a multi-word expression",
    "I-MWE": "Inside (continuation) token of a multi-word expression",
}

ID2LABEL_DEFAULT = {
    0: "O",
    1: "B-MWE",
    2: "I-MWE",
}

LABEL2ID_DEFAULT = {
    "O": 0,
    "B-MWE": 1,
    "I-MWE": 2,
}
