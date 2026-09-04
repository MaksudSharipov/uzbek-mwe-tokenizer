# Uzbek MWE Tokenizer (uzbek-mwe-tokenizer)

[![PyPI version](https://badge.fury.io/py/uzbek-mwe-tokenizer.svg)](https://pypi.org/project/uzbek-mwe-tokenizer/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Uzbek--MWE--Tokenizer--uzBERT-yellow)](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.61663%2F262tuitmct14-brightgreen)](https://uzjurnal.uz/2/2026/2/index?issue=14)

**uzbek-mwe-tokenizer** is an open-source Python framework designed to automatically identify, extract, and disambiguate **Multi-Word Expressions (MWEs)**—including idioms, auxiliary verb constructions, compound verbs, and compound nouns—in Uzbek natural language texts.

The library is powered by a fine-tuned **UzBERT** transformer backbone (MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT) using a BIO (Beginning, Inside, Outside) sequence classification scheme. It features a bidirectional cross-script normalization layer that solves the orthographic apostrophe fragmentation issue commonly encountered in Uzbek Latin subword tokenization.

---

## 📌 Key Features

- **High Precision Extraction:** Accurately extracts idioms, verbal collocations, and compound predicates with confidence scores.
- **Cross-Script Support (Latin & Cyrillic):** Native support for both official Uzbek orthographies (mode="lot" and mode="cyr").
- **Apostrophe Boundary Preservation:** Seamlessly overcomes subword fragmentation caused by Latin apostrophes (o', g', etc.) via monographemic Cyrillic neural inference.
- **Seamless Hugging Face Integration:** Automatically downloads and caches pre-trained weights directly from the Hugging Face Hub.
- **Ready for NLP Pipelines:** Acts as a drop-in preprocessing component for dependency parsers, machine translation engines, and search indexers.

---

## 🏗️ Architecture & Pipeline

`
[ Raw Input (Latin or Cyrillic) ]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Normalization & Transliteration Layer                    │
│    - Unifies apostrophes (’, ‘, ʻ -> ')             │
│    - Converts Latin input to monographemic Cyrillic         │
│    - Preserves word boundaries                              │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. UzBERT Transformer Inference Engine                      │
│    - Subword tokenization via UzBERT-Uncased                │
│    - Subword-to-word index alignment (word_ids)             │
│    - BIO sequence classification (B-MWE, I-MWE, O)          │
│    - Softmax confidence estimation                          │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Assembly & Script Reconstruction Layer                   │
│    - Filters valid expressions (length >= 2 words)          │
│    - Reconstructs complete words from subwords              │
│    - Translates results back to user's script mode          │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
[ Structured Output: MWE text + Confidence Score (%) ]
`

---

## 📊 Benchmark Results

Evaluated on an isomorphic dataset of **11,936 sentences, 108,200 tokens, and 14,285 MWE instances**:

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (This Package)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

*Note: While Latin models slightly lead in raw test F1 score, confusion matrix token-level analysis confirms that the monographemic Cyrillic pipeline identifies **1,342 B-MWE** and **1,467 I-MWE** tokens without boundary truncation, fully solving apostrophe fragmentation.*

---

## 🚀 Installation

Install via PyPI:
`ash
pip install uzbek-mwe-tokenizer
`

Or install from source:
`ash
git clone https://github.com/MaksudSharipov/uzbek-mwe-tokenizer.git
cd uzbek-mwe-tokenizer
pip install -r requirements.txt
pip install -e .
`

---

## 💡 Quick Start & Usage

### 1. Latin Script Example (mode="lot")

`python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Latin mode (default)
tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
results = tokenizer.extract_mwe(text)

print("Detected MWEs:")
for item in results:
    print(f" -> [{item['mwe']}] (Confidence: {item['confidence']:.1f}%)")
`

**Output:**
`	ext
Detected MWEs:
 -> [tarvuzi qo'ltig'idan tushdi] (Confidence: 100.0%)
`

### 2. Cyrillic Script Example (mode="cyr")

`python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Cyrillic mode
tokenizer = UzbekMWETokenizer(mode="cyr")

sentence = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
results = tokenizer.extract_mwe(sentence)

print("Topilgan MWE'lar:")
for item in results:
    print(f" -> [{item['mwe']}] (Ishonch: {item['confidence']:.1f}%)")
`

**Output:**
`	ext
Topilgan MWE'lar:
 -> [капалаги учди] (Ishonch: 100.0%)
 -> [ерга кирса] (Ishonch: 76.0%)
 -> [қулоғидан тортиб чиқарадиган] (Ishonch: 99.0%)
`

---

## 📖 Citation

If you use this software or model in your academic work, please cite the following publication:

`ibtex
@article{sharipov2026transformer,
  title={Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts},
  author={Sharipov, Maksud S.},
  journal={Bulletin of TUIT: Management and Communication Technologies},
  volume={2},
  number={14},
  pages={103--107},
  year={2026},
  doi={10.61663/262tuitmct14},
  url={https://uzjurnal.uz/2/2026/2/index?issue=14}
}
`

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact & Support
- **Author:** Maksud S. Sharipov
- **Affiliation:** Tashkent University of Information Technologies (TUIT)
- **Email:** maqsbek72@gmail.com
- **Hugging Face:** [@MaksudSharipov](https://huggingface.co/MaksudSharipov)
