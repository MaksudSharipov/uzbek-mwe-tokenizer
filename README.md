# UzbekMWETokenizer (`uzbek-mwe-tokenizer`)

[![PyPI version](https://badge.fury.io/py/uzbek-mwe-tokenizer.svg)](https://pypi.org/project/uzbek-mwe-tokenizer/)
[![PyPI Downloads](https://static.pepy.tech/badge/uzbek-mwe-tokenizer)](https://pepy.tech/project/uzbek-mwe-tokenizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT)

**UzbekMWETokenizer** (`uzbek-mwe-tokenizer`) is an open-source Python library for Transformer-based Multi-Word Expression (MWE) identification and BIO sequence tagging of the Uzbek language. It is built on a fine-tuned **UzBERT** model and is designed to extract idiomatic phrases, auxiliary verb constructions, compound verbs, and compound nouns in context.

- **Current software version:** `v0.1.0`
- **PyPI package:** [uzbek-mwe-tokenizer](https://pypi.org/project/uzbek-mwe-tokenizer/)
- **GitHub repository:** [MaksudSharipov/uzbek-mwe-tokenizer](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer)
- **Hugging Face model:** [MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT)
- **License:** MIT License

The library addresses key challenges of Uzbek NLP, including agglutinative word formation, syntactic ambiguity of multi-word predicates, apostrophe normalization in Latin script, cross-script (Latin/Cyrillic) projection, and subword-to-word alignment in Transformer-based token classification.

## Main Features

- Transformer-based MWE extraction and BIO sequence tagging (`B-MWE`, `I-MWE`, `O`) for Uzbek texts.
- Fine-tuned 108.5M-parameter `UzBERT-Uncased` model for contextual MWE boundary prediction.
- Native cross-script support for both Latin (`mode="lot"`) and Cyrillic (`mode="cyr"`) Uzbek alphabets.
- Smart apostrophe normalization (`’`, `‘`, ````, `ʻ` $\rightarrow$ `'`) and monographemic Cyrillic projection to eliminate subword fragmentation.
- Subword-to-word alignment via `word_ids()` and phrase-level Softmax confidence aggregation.
- Python API and command-line interface (`uzbek-mwe`) supporting `mwe`, `slash`, `conll`, and `json` outputs.
- Public distribution through PyPI (1,411+ downloads) and Hugging Face Hub (138+ model downloads).

## Installation

Install the package from PyPI:

```bash
pip install uzbek-mwe-tokenizer
```

The package requires Python 3.8 or higher. The main dependencies are `torch`, `transformers`, `UzTransliterator`, and `sentencepiece`.

For development installation from GitHub:

```bash
git clone https://github.com/MaksudSharipov/uzbek-mwe-tokenizer.git
cd uzbek-mwe-tokenizer
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Latin Script (`mode="lot"`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
result = tokenizer.extract_mwe(text)

print(result)
```

Expected output:

```text
[{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
```

### Cyrillic Script (`mode="cyr"`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="cyr")

sentence = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
print(tokenizer.extract_mwe(sentence))
```

Expected output:

```text
[
    {'mwe': 'капалаги учди', 'confidence': 100.0},
    {'mwe': 'ерга кирса', 'confidence': 76.0},
    {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}
]
```

## Command-Line Usage

After installation, the package provides the `uzbek-mwe` command:

```bash
uzbek-mwe "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
```

Slash BIO output:

```bash
uzbek-mwe "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." --format slash
```

CoNLL BIO output:

```bash
uzbek-mwe "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." --format conll
```

## BIO Tagset

`UzbekMWETokenizer` uses three BIO sequence labels:

| Tag | Description | Uzbek explanation |
|---|---|---|
| `O` | Outside any multi-word expression | Oddiy (mustaqil) so‘z |
| `B-MWE` | Beginning token of a multi-word expression | Ko‘p so‘zli birikmaning boshlanishi |
| `I-MWE` | Continuation (Inside) token of a multi-word expression | Ko‘p so‘zli birikmaning davomi |

More information is available in [`docs/tagset.md`](docs/tagset.md).

## Model

The core MWE token classification model is available on Hugging Face Hub:

```text
MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT
```

Model page:

```text
https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT
```

## Model Performance

Evaluated on an isomorphic Latin–Cyrillic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, and 14,285 MWEs):

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (UzbekMWETokenizer)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

Additional benchmark scripts and inference-speed tests are stored in the [`benchmark/`](benchmark/) directory.

## Repository Structure

```text
uzbek-mwe-tokenizer/
├── README.md
├── LICENSE
├── CITATION.cff
├── RELEASE_NOTES_v0.1.0.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── uzbek_mwe_tokenizer/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── text_utils.py
│   ├── label_map.py
│   └── cli.py
├── examples/
│   ├── basic_usage.py
│   ├── batch_extraction.py
│   ├── export_conll.py
│   └── cli_usage.py
├── tests/
│   ├── test_import.py
│   ├── test_label_map.py
│   ├── test_text_utils.py
│   ├── test_output_format.py
│   └── test_model_integration_optional.py
├── benchmark/
│   ├── evaluate.py
│   ├── speed_test.py
│   ├── results.csv
│   └── README.md
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── tagset.md
└── paper/
    └── SOFTWARE_IMPACTS_MANUSCRIPT.md
```

## Documentation

- Installation guide: [`docs/installation.md`](docs/installation.md)
- Usage guide: [`docs/usage.md`](docs/usage.md)
- BIO tagset & MWE categories: [`docs/tagset.md`](docs/tagset.md)
- Benchmark scripts: [`benchmark/README.md`](benchmark/README.md)

## Examples

Run the basic example:

```bash
python examples/basic_usage.py
```

Run batch extraction:

```bash
python examples/batch_extraction.py
```

Export results in slash and CoNLL BIO formats:

```bash
python examples/export_conll.py
```

## Testing

Install development dependencies and run unit tests:

```bash
pip install pytest
pytest tests -v
```

Optional real-model integration test:

```bash
RUN_MODEL_TESTS=1 pytest tests/test_model_integration_optional.py
```

## Benchmarking

Evaluate on a CoNLL-like BIO test file:

```bash
python benchmark/evaluate.py --test-file path/to/test.conll --output benchmark/eval_summary.csv
```

Run inference-speed test:

```bash
python benchmark/speed_test.py
```

## Software Impacts / SoftwareX Article

This repository accompanies the software manuscript:

**UzbekMWETokenizer: A Python library for Transformer-based multi-word expression identification in Uzbek**

The repository provides the source code, usage examples, documentation, unit tests, benchmark scripts, and reproducibility materials required for the Original Software Publication.

## Citation

If you use this library or model in academic research, please cite:

```bibtex
@article{sharipov2026mwe,
  title={Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts},
  author={Sharipov, Maksud S.},
  journal={Bulletin of TUIT: Management and Communication Technologies},
  volume={2},
  number={14},
  pages={103--107},
  year={2026},
  doi={10.61663/262tuitmct14}
}

@article{sharipov2026uzbektaggerbert,
  title={UzbekTaggerBERT: A Python library for contextual part-of-speech tagging of Uzbek},
  author={Sharipov, Maksud},
  journal={Software Impacts},
  volume={30},
  pages={100865},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.simpa.2026.100865}
}
```

## Author

**Maksud Sharipov**

- ORCID: https://orcid.org/0000-0002-2363-6533
- GitHub: https://github.com/MaksudSharipov
- Hugging Face: https://huggingface.co/MaksudSharipov
- PyPI: https://pypi.org/project/uzbek-mwe-tokenizer/

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
