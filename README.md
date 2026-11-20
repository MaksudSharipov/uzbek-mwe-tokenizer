# Uzbek MWE Tokenizer (`uzbek-mwe-tokenizer`)

[![PyPI version](https://badge.fury.io/py/uzbek-mwe-tokenizer.svg)](https://pypi.org/project/uzbek-mwe-tokenizer/)
[![PyPI Downloads](https://static.pepy.tech/badge/uzbek-mwe-tokenizer)](https://pepy.tech/project/uzbek-mwe-tokenizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT)

**`uzbek-mwe-tokenizer`** is an open-source Python library for Transformer-based Multi-Word Expression (MWE) identification in Uzbek language texts. It is built on a fine-tuned **UzBERT** model (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) and extracts idiomatic expressions, auxiliary verb constructions, compound verbs, and compound nouns in context.

- **Current software version:** `v0.1.0`
- **PyPI package:** [uzbek-mwe-tokenizer](https://pypi.org/project/uzbek-mwe-tokenizer/)
- **GitHub repository:** [MaksudSharipov/uzbek-mwe-tokenizer](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer)
- **Hugging Face model:** [MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT)
- **License:** MIT License

The library addresses key challenges of Uzbek NLP, including agglutinative word formation, syntactic ambiguity of multi-word predicates, apostrophe normalization in Latin script, cross-script (Latin/Cyrillic) projection, and subword-to-word alignment in Transformer-based token classification.

## Main Features

- Transformer-based MWE identification (`B-MWE`, `I-MWE`, `O`) for Uzbek texts.
- Fine-tuned 108.5M-parameter `UzBERT` model for contextual MWE boundary prediction.
- Cross-script support for both Latin (`mode="lot"`) and Cyrillic (`mode="cyr"`) Uzbek alphabets.
- Smart apostrophe normalization (`’`, `‘`, ````, `ʻ` $\rightarrow$ `'`) and monographemic Cyrillic projection to prevent subword fragmentation.
- Subword-to-word alignment via `word_ids()` and phrase-level Softmax confidence aggregation.
- Clean Python API (`UzbekMWETokenizer` and `extract_mwe`) for integration into Uzbek NLP pipelines.
- Public distribution through PyPI (1,411+ cumulative downloads; 207 in the last 30 days) and Hugging Face Hub (146+ all-time model downloads; 18 in the last 30 days).

## Installation

Install the package from PyPI:

```bash
pip install uzbek-mwe-tokenizer
```

For development installation from GitHub:

```bash
git clone https://github.com/MaksudSharipov/uzbek-mwe-tokenizer.git
cd uzbek-mwe-tokenizer
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Example in Cyrillic (`mode="cyr"`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Cyrillic mode
tokenizer = UzbekMWETokenizer(mode="cyr")

sentences = [
    "Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.",
    "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.",
    "Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади."
]

for gap in sentences:
    print("Gap:", gap)
    mwes = tokenizer.extract_mwe(gap)
    print("Topilgan MWE'lar:", mwes)
    print("-" * 50)
```

Expected output:

```text
Gap: Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.
Topilgan MWE'lar: [{'mwe': 'тарвузи қўлтиғидан тушди', 'confidence': 100.0}]
--------------------------------------------------
Gap: Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.
Topilgan MWE'lar: [{'mwe': 'капалаги учди', 'confidence': 100.0}, {'mwe': 'ерга кирса', 'confidence': 76.0}, {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}]
--------------------------------------------------
Gap: Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади.
Topilgan MWE'lar: [{'mwe': 'тепа сочимиз тик бўлмади', 'confidence': 100.0}]
--------------------------------------------------
```

### Example in Latin (`mode="lot"`)

The package automatically transliterates Latin text to Cyrillic for neural inference and translates extracted MWEs back to Latin:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Latin mode
tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
mwes = tokenizer.extract_mwe(text)
print(mwes)
```

Expected output:

```text
[{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
```

## BIO Tagset

`UzbekMWETokenizer` uses three BIO sequence labels internally:

| Tag | Description | Uzbek explanation |
|---|---|---|
| `O` | Outside any multi-word expression | Oddiy (mustaqil) so‘z |
| `B-MWE` | Beginning token of a multi-word expression | Ko‘p so‘zli birikmaning boshlanishi |
| `I-MWE` | Continuation (Inside) token of a multi-word expression | Ko‘p so‘zli birikmaning davomi |

More information is available in [`docs/tagset.md`](docs/tagset.md).

## Model Performance

Evaluated on an isomorphic Latin–Cyrillic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, and 14,285 MWEs):

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (`uzbek-mwe-tokenizer`)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

## Repository Structure

```text
uzbek-mwe-tokenizer/
├── README.md
├── LICENSE
├── CITATION.cff
├── RELEASE_NOTES_v0.1.0.md
├── setup.py
├── requirements.txt
├── test_examples.py
├── uzbek_mwe_tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
├── examples/
│   ├── basic_usage.py
│   └── batch_extraction.py
├── tests/
│   ├── test_import.py
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

## Citation

If you use this model or package in your research or project, please cite:

> Sharipov, M. S. (2026). Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. *Bulletin of TUIT: Management and Communication Technologies*, 2(14), 103-107. DOI: 10.61663/262tuitmct14  
> **Link:** [https://uzjurnal.uz/2/2026/2/index?issue=14](https://uzjurnal.uz/2/2026/2/index?issue=14)

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
```

## Author

**Maksud Sharipov**

- ORCID: https://orcid.org/0000-0002-2363-6533
- GitHub: https://github.com/MaksudSharipov
- Hugging Face: https://huggingface.co/MaksudSharipov
- PyPI: https://pypi.org/project/uzbek-mwe-tokenizer/

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
