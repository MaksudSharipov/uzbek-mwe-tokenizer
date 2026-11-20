"""UzbekMWETokenizer: Transformer-based Multi-Word Expression identification for Uzbek."""

from .label_map import MWE_LABEL_DESCRIPTIONS, MWE_LABELS
from .tokenizer import DEFAULT_MODEL_NAME, UzbekMWEExtractor, UzbekMWETokenizer

__version__ = "0.1.0"

__all__ = [
    "UzbekMWETokenizer",
    "UzbekMWEExtractor",
    "MWE_LABELS",
    "MWE_LABEL_DESCRIPTIONS",
    "DEFAULT_MODEL_NAME",
]
