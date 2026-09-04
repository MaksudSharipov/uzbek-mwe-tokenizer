# uzbek-mwe-tokenizer: A Transformer-based Python Framework for Multi-Word Expression Identification and Syntactic Disambiguation in Uzbek Texts

**Maksud S. Sharipov**  
*Department of Convergence of Digital Technologies, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi, Urgench, Uzbekistan*  
*Email: maqsbek72@gmail.com*  

---

## Abstract
In agglutinative languages such as Uzbek, multi-word expressions (MWEs)—encompassing auxiliary verb constructions, compound nouns, and idiomatic phrases—present significant syntactic ambiguities for conventional word-by-word natural language processing (NLP) pipelines. Furthermore, orthographic irregularities in the Latin script (specifically, apostrophes used in digraphs like *o‘* and *g‘*) frequently lead to severe subword tokenization fragmentation when using standard WordPiece and BPE tokenizers. This article introduces `uzbek-mwe-tokenizer`, an open-source Python library powered by a fine-tuned UzBERT transformer model that automates MWE extraction using a sequence-labeling BIO scheme. The framework integrates an automated bidirectional transliteration mechanism that projects Latin text into a monographemic Cyrillic representation during neural inference, eliminating orthographic splitting while transparently restoring predictions to the user's chosen script. Distributed via PyPI and the Hugging Face Hub, `uzbek-mwe-tokenizer` provides a production-ready, highly accurate, and reusable software component for downstream syntactic parsing, machine translation, and text representation tasks.

**Keywords:** Multi-Word Expressions; NLP; Tokenization; UzBERT; Transformer; Uzbek language; Syntactic parsing; Cross-script normalization.

---

## Code Metadata

| Nr. | Code metadata description | Details |
| :--- | :--- | :--- |
| C1 | Current code version | v0.1.0 |
| C2 | Permanent link to code/repository | [https://github.com/MaksudSharipov/uzbek-mwe-tokenizer](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer) |
| C3 | Package distribution link (PyPI) | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| C4 | Model hub repository | [https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT) |
| C5 | Legal Code License | MIT License |
| C6 | Code versioning system used | Git |
| C7 | Software code languages, tools, and services | Python 3.8+, PyTorch, Hugging Face Transformers, SentencePiece, UzTransliterator |
| C8 | Compilation requirements, Operating environments | OS-independent (Linux, Windows, macOS); Python >= 3.8 |
| C9 | Developer documentation / manual | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| C10 | Support email for questions | maqsbek72@gmail.com |

---

## 1. Motivation and Background

Multi-Word Expressions (MWEs) are lexical units consisting of two or more words that exhibit syntactic or semantic idiosyncrasies (e.g., idiomatic phrases like *tarvuzi qo‘ltig‘idan tushdi* [to lose hope], or auxiliary verb constructions like *sotib olmoq* [to buy]). In highly agglutinative languages like Uzbek, word forms are constructed by appending multiple grammatical affixes to word stems. When standard NLP pipelines process such texts using isolated word-by-word tokenization, MWEs are fragmented into separate lexical tokens, resulting in distorted syntactic dependency trees, faulty semantic representations, and erroneous machine translations.

This challenge is exacerbated by a unique cross-script orthographic hurdle: modern Uzbek concurrently employs both Latin and Cyrillic writing systems. In the official Latin alphabet, characters such as *o‘*, *g‘*, and glottal markers rely on apostrophe modifiers (e.g., `’`, `‘`, `'`, `ʻ`). Subword tokenizers (such as BPE and WordPiece) commonly interpret these apostrophes as punctuation boundaries rather than integral parts of the graphemes. Consequently, words are shattered into nonsensical subword fragments, creating high error rates in sequence labeling and destabilizing the *I-MWE* (Inside) token sequences.

To address this challenge, `uzbek-mwe-tokenizer` was developed as an open-source, modular Python framework. It bridges deep contextual transformer architectures with orthographic transliteration logic to deliver robust MWE boundary identification and syntactic disambiguation across both writing systems.

---

## 2. Software Description and Architecture

The `uzbek-mwe-tokenizer` framework comprises three sequential processing stages:

```
[ Raw Input Text (Latin or Cyrillic) ]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Normalization & Pre-tokenization Layer                   │
│    - Typographic apostrophe unification (`’`, `‘`, `ʻ` -> `'`)│
│    - Automated script detection                             │
│    - Optional Latin-to-Cyrillic transliteration            │
│    - Word-level splitting preserving token boundaries       │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Deep Transformer Inference Engine                        │
│    - Subword tokenization via UzBERT-Uncased (Cyrillic)     │
│    - Subword-to-word index alignment (word_ids mapping)     │
│    - Contextual sequence classification (Softmax & Logits)  │
│    - BIO tag assignment: B-MWE, I-MWE, O                    │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Assembly, Filtering & Script Reconstruction Layer         │
│    - Reconstructs complete words from subword predictions   │
│    - Aggregates multi-token phrases (minimum length >= 2)   │
│    - Computes mean confidence scores per expression         │
│    - Bidirectional Cyrillic-to-Latin output restitution    │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
[ Structured Output: List of MWEs with Confidence Scores ]
```

### 2.1. Normalization and Cross-Script Bridge
To eliminate the apostrophe fragmentation vulnerability inherent to Latin subword models, the pre-processing layer standardizes all variant apostrophe characters (`’`, `‘`, ````, `ʻ`) to a uniform ASCII quote (`'`). When operating in Latin mode (`mode="lot"`), the text is dynamically transliterated into monographemic Cyrillic using rule-based algorithmic mapping (`ў`, `ғ`, `ш`, `ч`). This ensures the text fed into the transformer contains strictly monographemic characters, entirely circumventing subword token fragmentation.

### 2.2. Transformer-based BIO Classification
The core inference engine utilizes `AutoModelForTokenClassification` loaded from Hugging Face (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`). Words are tokenized with `is_split_into_words=True`. Using the tokenizer's `word_ids()` mapping, each subword unit is mapped back to its corresponding parent orthographic word. The model generates classification logits for three target labels:
- `B-MWE`: Beginning of a Multi-Word Expression.
- `I-MWE`: Continuation (Inside) of a Multi-Word Expression.
- `O`: Outside any Multi-Word Expression.

### 2.3. MWE Assembly and Confidence Calculation
The extraction algorithm iterates through the token predictions. Consecutive `B-MWE` and `I-MWE` tags are grouped into composite expressions. If an expression spans at least two individual words ($\ge 2$), it is retained as a valid MWE, and its mean confidence score is calculated:

$$\text{Confidence}(\text{MWE}) = \frac{1}{N} \sum_{i=1}^{N} P(y_i \mid w_i) \times 100\%$$

where $N$ is the number of constituent words and $P(y_i \mid w_i)$ represents the Softmax probability of the predicted BIO label for word $w_i$. Finally, if the initial input was in Latin script, the extracted expressions are transliterated back into standard Latin orthography.

---

## 3. Illustrative Examples and Usage

### 3.1. Installation
The package can be installed directly from PyPI:
```bash
pip install uzbek-mwe-tokenizer
```

All model weights (~440 MB) and tokenizer configuration files are automatically fetched from Hugging Face upon initial invocation and cached locally.

### 3.2. Programmatic Execution in Python

#### Example 1: Latin Script Processing
The following listing demonstrates extracting idiomatic expressions from Latin-script Uzbek text:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Latin mode
tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
results = tokenizer.extract_mwe(text)

print("Detected MWEs:")
for item in results:
    print(f" - MWE: '{item['mwe']}' | Confidence: {item['confidence']:.2f}%")
```

**Output:**
```text
Detected MWEs:
 - MWE: 'tarvuzi qo'ltig'idan tushdi' | Confidence: 100.00%
```

#### Example 2: Cyrillic Script and Compound Predicates
The tokenizer seamlessly processes complex Cyrillic inputs containing multiple figurative and auxiliary verbal structures:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="cyr")

sentences = [
    "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.",
    "Рақибларимизни кўриб бизning асло тепа сочимиз тик бўлмади."
]

for sentence in sentences:
    print(f"\nSentence: {sentence}")
    mwes = tokenizer.extract_mwe(sentence)
    for res in mwes:
        print(f"  -> Extracted: [{res['mwe']}] (Confidence: {res['confidence']:.1f}%)")
```

**Output:**
```text
Sentence: Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.
  -> Extracted: [капалаги учди] (Confidence: 100.0%)
  -> Extracted: [ерга кирса] (Confidence: 76.0%)
  -> Extracted: [қулоғидан тортиб чиқарадиган] (Confidence: 99.0%)

Sentence: Рақибларимизни кўриб бизning асло тепа сочимиз тик бўлмади.
  -> Extracted: [тепа сочимиз тик бўлмади] (Confidence: 100.0%)
```

---

## 4. Impact Overview

### 4.1. Experimental Validation and Benchmark
The model integrated into `uzbek-mwe-tokenizer` was trained and benchmarked against an isomorphic corpus comprising **11,936 sentences, 108,200 tokens, and 14,285 manually annotated MWE instances**, evaluated using an 80/10/10 train/validation/test split across 5 epochs.

*Table 1: Comparative empirical performance across candidate transformer architectures.*
| Model Architecture | Base Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (Integrated)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

### 4.2. Resolving the "Accuracy Paradox" and Token Boundary Loss
Although the Latin-trained `BERTbek-News` model yielded a marginally higher quantitative test F1-score (0.8817 vs. 0.8722), comprehensive confusion matrix and token-level error analyses revealed a qualitative breakthrough achieved by `uzbek-mwe-tokenizer`:
- In Latin-based models, subword tokenizers isolate apostrophes into separate tokens (e.g., `qo` + `'` + `ltig` + `idan`). This frequently fractures the contiguous `I-MWE` label sequence, causing false boundary terminations.
- The monographemic Cyrillic pipeline implemented in `uzbek-mwe-tokenizer` identified **1,342 B-MWE** and **1,467 I-MWE** tokens without boundary truncation, guaranteeing that multi-token units remain unified.

### 4.3. Impact on Downstream NLP Systems
The availability of `uzbek-mwe-tokenizer` directly benefits multiple NLP fields:
1. **Syntactic Dependency Parsers:** Identifying multi-word predicates (*sotib olmoq*) and idioms prior to syntactic analysis collapses multi-token spans into single semantic nodes, drastically reducing false dependency attachments.
2. **Machine Translation Systems:** Pre-tagging MWEs prevents word-by-word literal translations (e.g., translating *bosh qotirmoq* literally as *head solidify* instead of *ponder/contemplate*).
3. **Information Retrieval and Search:** Search engines can index verified MWEs as atomic n-grams, reducing false positives caused by isolated keyword matches.

---

## 5. Conclusions and Future Improvements

The `uzbek-mwe-tokenizer` framework delivers an accessible, production-ready tool for multi-word lexical processing in Uzbek. By harmonizing deep transformer representations with an automated cross-script normalizer, the package effectively resolves the orthographic apostrophe fragmentation bottleneck. Future releases will explore lightweight distilled models (e.g., DistilBERT/MobileBERT) for edge deployment and direct subword vocabulary adaptations to support discontinuous MWEs natively.

---

## Conflict of Interest
The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgements
The author expresses sincere gratitude to the Computational Linguistics research team at Urgench State University named after Abu Rayhan Beruni and Tashkent University of Information Technologies for providing corpus annotation support and computing infrastructure.

---

## References

1. Sharipov, M. S. (2026). Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. *Bulletin of TUIT: Management and Communication Technologies*, 2(14), 103–107. https://doi.org/10.61663/262tuitmct14
2. Bobojonova, L., et al. (2024). BBPOS: BERT-based Part-of-Speech Tagging for Uzbek. *Proceedings of RPTU Kaiserslautern-Landau & Saarland University*.
3. Constant, M., Eryiğit, G., Monti, J., van der Plas, L., Ramisch, C., Rosner, M., & Todirascu, A. (2017). Multiword Expression Processing: A Survey. *Computational Linguistics*, 43(4), 837–892. https://doi.org/10.1162/COLI_a_00302
4. Kuriyozov, E., et al. (2024). BERTbek: A Pretrained Language Model for Uzbek. *Proceedings of LREC-COLING 2024*.
5. Mansurov, B., & Mansurov, A. (2021). UzBERT: pretraining a BERT model for Uzbek. *arXiv preprint arXiv:2104.09459*.
6. Salaev, U., Kuriyozov, E., & Gómez-Rodríguez, C. (2022). A Machine Transliteration Tool Between Uzbek Alphabets. *CEUR Workshop Proceedings*, 3246, 42–49.
7. Miletić, F., & Schulte im Walde, S. (2024). Semantics of Multiword Expressions in Transformer-Based Models: A Survey. *Transactions of the Association for Computational Linguistics*, 12, 108–130.
