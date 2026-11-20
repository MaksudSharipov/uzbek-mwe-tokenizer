## uzbek-mwe-tokenizer v0.1.0

This release corresponds to the published PyPI version (`v0.1.0`) of **`uzbek-mwe-tokenizer`**, an open-source Python library for Transformer-based Multi-Word Expression (MWE) identification in Uzbek texts.

### Highlights

- Transformer-based MWE extraction (`B-MWE`, `I-MWE`, `O`) for Uzbek texts.
- Fine-tuned 108.5M-parameter `UzBERT` model integration (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`).
- Python API (`UzbekMWETokenizer` with `extract_mwe`).
- Native cross-script support for both Latin (`mode="lot"`) and Cyrillic (`mode="cyr"`) Uzbek alphabets.
- Automated Uzbek apostrophe normalization and monographemic Cyrillic projection to eliminate subword fragmentation.
- Subword-to-word alignment via `word_ids()` and phrase-level Softmax confidence aggregation.
- Examples, unit tests, documentation, CI workflow, and benchmark scripts included.
- Prepared for submission as an Original Software Publication (OSP) to *Software Impacts* / *SoftwareX*.

### Repository structure

- `uzbek_mwe_tokenizer/` — exact PyPI `v0.1.0` package source code (`__init__.py`, `tokenizer.py`)
- `examples/` — runnable usage examples (`basic_usage.py`, `batch_extraction.py`)
- `tests/` — unit and integration tests (`test_import.py`, `test_model_integration_optional.py`)
- `benchmark/` — evaluation and inference-speed benchmark scripts (`evaluate.py`, `speed_test.py`, `results.csv`)
- `docs/` — installation, usage, and BIO tagset documentation
- `paper/` — companion Original Software Publication manuscript

### Installation

```bash
pip install uzbek-mwe-tokenizer
```

### Model

https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT

### License

MIT License
