## UzbekMWETokenizer (`uzbek-mwe-tokenizer`) v0.1.0

This release provides the *Software Impacts* / *SoftwareX*-ready version of **UzbekMWETokenizer** (`uzbek-mwe-tokenizer`), an open-source Python library for Transformer-based Multi-Word Expression (MWE) identification and BIO sequence tagging in Uzbek texts.

### Highlights

- Transformer-based MWE extraction and word-level BIO tagging (`B-MWE`, `I-MWE`, `O`) for Uzbek texts.
- Fine-tuned 108.5M-parameter `UzBERT-Uncased` model integration (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`).
- Python API and command-line interface (`uzbek-mwe`).
- Native cross-script support for both Latin (`mode="lot"`) and Cyrillic (`mode="cyr"`) Uzbek alphabets.
- Automated Uzbek apostrophe normalization and monographemic Cyrillic projection to eliminate subword fragmentation.
- Subword-to-word alignment via `word_ids()` and phrase-level Softmax confidence aggregation.
- Examples, unit tests, documentation, CI workflow, and benchmark scripts included.
- Prepared for submission as an Original Software Publication (OSP) to *Software Impacts* / *SoftwareX*.

### Repository structure

- `uzbek_mwe_tokenizer/` — core source code (`tokenizer.py`, `text_utils.py`, `label_map.py`, `cli.py`)
- `examples/` — runnable usage examples (`basic_usage.py`, `batch_extraction.py`, `export_conll.py`, `cli_usage.py`)
- `tests/` — unit and integration tests
- `benchmark/` — evaluation and inference-speed benchmark scripts
- `docs/` — installation, usage, and BIO tagset documentation
- `paper/` — companion Original Software Publication manuscript

### Validation

The package was validated with:

- Unit tests passing (`pytest tests -v`) for package imports, BIO label mappings, apostrophe normalization, and output formatters (`mwe`, `slash`, `conll`, `json`)
- Optional Hugging Face model integration test (`RUN_MODEL_TESTS=1 pytest tests/test_model_integration_optional.py`)
- Python API and CLI tests for Latin and Cyrillic inputs
- Inference-speed benchmark (`benchmark/speed_test.py`)

### Reported MWE identification performance

Evaluated on an isomorphic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, 14,285 MWE instances):
- `UzBERT-Uncased` (integrated Cyrillic backbone): Accuracy = `0.9559`, Precision = `0.8528`, Recall = `0.8926`, F1-Score = `0.8722` (recovering `1,342` `B-MWE` and `1,467` `I-MWE` tokens with zero apostrophe subword fragmentation).

### Installation

```bash
pip install uzbek-mwe-tokenizer
```

### Model

https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT

### License

MIT License
