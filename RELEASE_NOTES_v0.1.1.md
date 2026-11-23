## uzbek-mwe-tokenizer v0.1.1

This release provides the *Software Impacts*-ready update (`v0.1.1`) of **`uzbek-mwe-tokenizer`**, an open-source Python library for Transformer-based Multi-Word Expression (MWE) identification in Uzbek texts.

### What's New in v0.1.1 (over v0.1.0)

- **Punctuation-Decoupled Pre-Tokenization (`Algorithm 1`):** Separates boundary commas and periods (`"tushdi."`, `"учди,"`, `"кирса,"`) from extracted MWE words while keeping internal Uzbek apostrophes (`qo'ltig'idan`, `o'quvchilarning`) intact.
- **Standalone Punctuation Buffer Flush:** Automatically flushes and resets the active MWE span buffer whenever standalone punctuation is encountered.
- **Fast Tokenizer & Long-Input Truncation Fix:** Enables `use_fast=True` for `inputs.word_ids()` compatibility across modern `transformers` releases and adds `truncation=True, max_length=512` to safely handle inputs exceeding 512 subwords.
- **Forward Compatibility with `UzTransliterator` 2.x:** Supports both `UzTransliterator < 2.0` and `UzTransliterator >= 2.0` APIs seamlessly.
- **Confidence Rounding:** Rounds phrase-level Softmax confidence scores to two decimal places (`round(avg_score, 2)`).
- **Test & Benchmark Suite:** Includes `tests/test_tokenizer.py` (`pytest` across Python 3.10–3.13) and `benchmark/` scripts.
- **Co-Author & Environment Metadata:** Synchronized 6 co-authors and `python_requires=">=3.8"`.

### Installation

```bash
pip install --upgrade uzbek-mwe-tokenizer
```

### Model

https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT

### License

MIT License
