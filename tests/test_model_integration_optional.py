"""Optional real-model integration test (runs when RUN_MODEL_TESTS=1)."""

import os
import pytest
from uzbek_mwe_tokenizer import UzbekMWETokenizer


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to run full Hugging Face model integration test.",
)
def test_extract_mwe_latin_integration():
    tokenizer = UzbekMWETokenizer(mode="lot", quiet=True)
    text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
    mwes = tokenizer.extract_mwe(text)
    assert len(mwes) >= 1
    assert "tarvuzi" in mwes[0]["mwe"]
