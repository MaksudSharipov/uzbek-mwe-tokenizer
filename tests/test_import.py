"""Import tests for UzbekMWETokenizer."""

import uzbek_mwe_tokenizer


def test_package_imports():
    assert hasattr(uzbek_mwe_tokenizer, "UzbekMWETokenizer")
    assert hasattr(uzbek_mwe_tokenizer, "UzbekMWEExtractor")
    assert hasattr(uzbek_mwe_tokenizer, "MWE_LABELS")


def test_version_exists():
    assert isinstance(uzbek_mwe_tokenizer.__version__, str)
    assert len(uzbek_mwe_tokenizer.__version__) > 0
