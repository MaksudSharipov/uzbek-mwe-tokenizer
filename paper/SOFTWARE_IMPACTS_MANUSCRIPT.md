# uzbek-mwe-tokenizer: A Transformer-based Python Framework for Multi-Word Expression Identification and Syntactic Disambiguation in Uzbek Texts

**Maksud S. Sharipov**  
*Department of Convergence of Digital Technologies, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi, Urgench, Uzbekistan*  
*Email:* `maqsbek72@gmail.com`  

---

## Abstract
In agglutinative languages such as Uzbek, multi-word expressions (MWEs)—encompassing auxiliary verb constructions, compound nouns, and idiomatic phrases—create severe syntactic ambiguities for conventional word-by-word natural language processing (NLP) pipelines. Furthermore, orthographic irregularities in the Latin script (specifically, apostrophe modifiers in digraphs such as *o‘* and *g‘*) frequently cause subword tokenization fragmentation in standard WordPiece and Byte-Pair Encoding (BPE) tokenizers. This paper presents `uzbek-mwe-tokenizer`, an open-source Python framework powered by a fine-tuned 108.5M-parameter UzBERT transformer backbone (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) that automates MWE identification using a BIO sequence-labeling scheme. The software incorporates an automated bidirectional transliteration bridge that projects Latin inputs into a monographemic Cyrillic representation during neural inference, completely eliminating apostrophe-induced token splitting while transparently restoring extracted expressions to the user's native script. Distributed via PyPI and the Hugging Face Hub—with over 1,411 package downloads and 138 model weight downloads since release—`uzbek-mwe-tokenizer` offers a reproducible, production-ready preprocessing module for downstream syntactic dependency parsing, machine translation, and information retrieval systems.

**Keywords:** Multi-Word Expressions (MWE); Uzbek NLP; Tokenization; UzBERT; Transformer; Syntactic Disambiguation; Cross-Script Normalization.

---

## Metadata

*Table 1: Code and Software Metadata (compliant with Elsevier SoftwareX and Software Impacts OSP guidelines).*

| Nr. | Metadata description | Details |
| :--- | :--- | :--- |
| **C1** | **Current code version** | `v0.1.0` |
| **C2** | **Permanent link to code/repository** | [https://github.com/MaksudSharipov/uzbek-mwe-tokenizer](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer) |
| **C3** | **Package distribution archive (PyPI)** | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| **C4** | **Pre-trained model hub repository** | [https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT) |
| **C5** | **Legal Code License** | MIT License (OSI-approved) |
| **C6** | **Code versioning system used** | Git |
| **C7** | **Software code languages, tools, and services** | Python $\ge$ 3.8, PyTorch, Hugging Face Transformers, SentencePiece, UzTransliterator |
| **C8** | **Compilation requirements & operating environments** | OS-independent (Linux, Windows, macOS); CPU or CUDA-enabled GPU |
| **C9** | **Link to developer documentation / manual** | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| **C10** | **Support email for questions** | `maqsbek72@gmail.com` |

---

## 1. Motivation and Significance

In computational linguistics, the quality of syntactic and semantic parsing directly depends on the granularity of initial tokenization. In strongly agglutinative languages with rich derivational and inflectional morphology, such as Uzbek, Multi-Word Expressions (MWEs) play a central role in expressing complex predicates and figurative meanings [1, 2]. Although MWEs consist of two or more space-separated orthographic words, they function semantically and syntactically as a single indivisible lexeme—for instance, auxiliary verb constructions (*sotib olmoq* [to purchase]), compound predicates (*qaror qilmoq* [to decide]), and idioms (*tarvuzi qo‘ltig‘idan tushdi* [to become deeply discouraged]). When traditional word-by-word tokenizers and dependency parsers analyze MWE components as independent tokens, the resulting dependency tree suffers from structural attachment errors, and downstream machine translation engines produce literal, semantically distorted outputs [3, 4].

In Uzbek NLP, this challenge is compounded by a dual-orthography ecosystem where both Latin and Cyrillic scripts are actively used. Within the official Uzbek Latin alphabet, letters such as *o‘*, *g‘*, and the glottal stop (*’*) rely on apostrophe-like modifier symbols (`’`, `‘`, `'`, `ʻ`). Standard subword tokenization algorithms (WordPiece and BPE) deployed in pre-trained multilingual and monolingual transformers treat these apostrophes as punctuation delimiters rather than integral character components [1]. Consequently, a single grammatical word such as *qo‘ltig‘idan* is artificially fractured into disconnected fragments (`qo` + `'` + `ltig` + `'` + `idan`), disrupting the continuity of `I-MWE` (Inside) sequence tags and degrading boundary recall.

To overcome these algorithmic and orthographic bottlenecks, we developed `uzbek-mwe-tokenizer`. The software encapsulates our fine-tuned `UzBERT-Uncased` token classification model [1] within an end-to-end Python library that harmonizes cross-script normalization, word-to-subword index alignment, and automated script restitution.

---

## 2. Software Description

### 2.1. Software Architecture
The `uzbek-mwe-tokenizer` package is designed as a lightweight, object-oriented Python framework centered around the `UzbekMWETokenizer` class. As illustrated in Figure 1, the internal pipeline operates across three decoupled stages:

```
[ Raw Input Text (Latin `mode="lot"` or Cyrillic `mode="cyr"`) ]
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 1: Orthographic Normalization & Cross-Script Bridge         │
│  • Standardizes typographic apostrophes (`’`, `‘`, ````, `ʻ` -> `'`)│
│  • Transliterates Latin text to monographemic Cyrillic (`ў`, `ғ`) │
│  • Splits whitespace tokens while preserving word indices         │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 2: Neural Sequence Labeling (`UzBERT-Uncased` Backbone)     │
│  • 12-layer Transformer (`hidden_size=768`, 108,493,059 params)   │
│  • Subword tokenization with `is_split_into_words=True`           │
│  • Subword-to-word alignment via `inputs.word_ids()`              │
│  • Contextual logits & Softmax probability over {O, B-MWE, I-MWE} │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 3: Phrase Aggregation & Bidirectional Script Restitution    │
│  • Reconstructs whole orthographic words from subword predictions │
│  • Enforces multi-word length constraint (word count >= 2)        │
│  • Computes arithmetic mean confidence across constituent words   │
│  • Restores extracted MWEs back to native Latin or Cyrillic script│
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
[ Structured Output: List[Dict{"mwe": str, "confidence": float}] ]
```
*Figure 1: Architectural dataflow of the `uzbek-mwe-tokenizer` framework.*

### 2.2. Software Functionalities and Core Implementation
The core logic resides in `uzbek_mwe_tokenizer/tokenizer.py` and exposes a clean, modular API:

1. **Automated Model Provisioning (`__init__`):** Initializes the operating script mode (`lot` or `cyr`), instantiates the rule-based `UzTransliterator` engine [6], and automatically downloads and caches the pre-trained neural weights (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) via Hugging Face's `AutoTokenizer` and `AutoModelForTokenClassification`.
2. **Orthographic Sanitization (`_prepare_text`):** Unifies disparate Unicode apostrophe variants (`U+2019`, `U+2018`, `U+02BB`, `U+0060`) into a canonical form and maps Latin sequences into monographemic Cyrillic representations (`o‘` $\rightarrow$ `ў`, `g‘` $\rightarrow$ `ғ`, `sh` $\rightarrow$ `ш`, `ch` $\rightarrow$ `ч`). This guarantees that the WordPiece tokenizer never encounters mid-word punctuation splits.
3. **Subword-Aligned BIO Inference (`extract_mwe`):** Rather than assigning tags to fragmented subwords, the method aligns first-subword logits back to full orthographic words using `word_ids()`, aggregates contiguous `B-MWE` and `I-MWE` spans of length $\ge 2$, and calculates phrase-level confidence:

$$\text{Confidence}(\text{MWE}) = \frac{1}{N} \sum_{i=1}^{N} \left( \max_{c \in \{0,1,2\}} \frac{\exp(z_{i,c})}{\sum_{k} \exp(z_{i,k})} \right) \times 100\%$$

where $N \ge 2$ is the number of constituent words in the candidate MWE and $z_{i,c}$ denotes the output logit of word $i$ for BIO class $c$.

Below is the core extraction implementation from `uzbek_mwe_tokenizer/tokenizer.py`:

```python
def extract_mwe(self, raw_text):
    if not raw_text.strip():
        return []

    text = self._prepare_text(raw_text)
    words = text.split()
    
    inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = inputs.word_ids()

    with torch.no_grad():
        outputs = self.model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    probabilities = torch.nn.functional.softmax(logits, dim=2)[0]

    extracted_mwes, current_mwe_words, current_mwe_scores = [], [], []
    previous_word_idx = None

    def save_mwe():
        if len(current_mwe_words) >= 2:
            avg_score = sum(current_mwe_scores) / len(current_mwe_scores)
            mwe_str_formatted = self._format_output_text(" ".join(current_mwe_words))
            extracted_mwes.append({
                "mwe": mwe_str_formatted,
                "confidence": avg_score
            })

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            label = self.id2label[predictions[idx]]
            score = probabilities[idx][predictions[idx]].item() * 100
            full_word = words[word_idx]

            if label == "B-MWE":
                save_mwe()
                current_mwe_words, current_mwe_scores = [full_word], [score]
            elif label == "I-MWE":
                if current_mwe_words:
                    current_mwe_words.append(full_word)
                    current_mwe_scores.append(score)
                else:
                    current_mwe_words, current_mwe_scores = [full_word], [score]
            else:
                save_mwe()
                current_mwe_words, current_mwe_scores = [], []
        previous_word_idx = word_idx

    save_mwe()
    return extracted_mwes
```

---

## 3. Illustrative Examples

### 3.1. Installation
The package is hosted on the Python Package Index (PyPI) and can be installed with a single command:
```bash
pip install uzbek-mwe-tokenizer
```

### 3.2. Cross-Script MWE Identification (Latin and Cyrillic)
The following script (`test_examples.py`) demonstrates extracting idiomatic and compound verbal expressions from both Latin and Cyrillic Uzbek sentences:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# 1. Latin Mode Execution
tokenizer_lot = UzbekMWETokenizer(mode="lot")
latin_text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
print("Latin Output:", tokenizer_lot.extract_mwe(latin_text))

# 2. Cyrillic Mode Execution (Multiple MWEs in a single sentence)
tokenizer_cyr = UzbekMWETokenizer(mode="cyr")
cyrillic_text = (
    "Қоронғи хонада унинг қаттиқ капалаги учди, "
    "гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
)
print("Cyrillic Output:", tokenizer_cyr.extract_mwe(cyrillic_text))
```

**Execution Output:**
```text
Latin Output: [{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
Cyrillic Output: [
    {'mwe': 'капалаги учди', 'confidence': 100.0},
    {'mwe': 'ерга кирса', 'confidence': 76.0},
    {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}
]
```

### 3.3. Computational Performance and Resource Specifications
Table 2 summarizes the technical specifications and runtime profile of the `uzbek-mwe-tokenizer` inference engine (`BertForTokenClassification`, 12 layers, 12 attention heads, `hidden_size=768`, `vocab_size=30,000`).

*Table 2: Model architecture parameters and computational resource footprint.*

| Parameter / Metric | Specification / Value |
| :--- | :--- |
| **Backbone Architecture** | `BertForTokenClassification` (`UzBERT-Uncased`) |
| **Total Trainable Parameters** | $108,493,059$ (~108.5M) |
| **Serialized Model Footprint (`safetensors`)** | ~414 MiB (FP32) |
| **Maximum Sequence Length** | 512 subword tokens |
| **Vocabulary Size** | 30,000 WordPiece tokens |
| **Output Tag Space** | 3 classes (`0: O`, `1: B-MWE`, `2: I-MWE`) |
| **Average Inference Latency (Standard CPU)** | ~85–150 ms / sentence (cached in RAM) |
| **Average Inference Latency (CUDA GPU, T4/A100)** | ~4.2–8.5 ms / sentence |

---

## 4. Impact Overview

### 4.1. Experimental Benchmark and Resolution of the "Accuracy Paradox"
The neural model underlying `uzbek-mwe-tokenizer` was trained and evaluated on an isomorphic Latin–Cyrillic corpus of **11,936 sentences (108,200 tokens) containing 14,285 annotated MWEs** across six transformer architectures [1]:

*Table 3: Comparative evaluation of six transformer backbones on the Uzbek MWE test split [1].*

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BERTbek-News [4] | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (Integrated)** [5] | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

While Latin-script models (`BERTbek-News`) achieved a slightly higher aggregate F1-score due to lower false-positive rates on `O` tokens, confusion matrix analyses revealed a critical **"accuracy paradox"** [1]: Latin tokenizers fracture apostrophe-bearing words into multiple subwords, causing mid-phrase tag dropouts. By routing Latin text through the monographemic Cyrillic `UzBERT-Uncased` backbone, `uzbek-mwe-tokenizer` correctly recovered **1,342 `B-MWE` and 1,467 `I-MWE` tokens** without boundary truncation—outperforming Latin models in full-phrase structural integrity.

### 4.2. Impact on Downstream NLP Pipelines
1. **Syntactic Dependency Parsing (UD):** Collapsing multi-word verbal and idiomatic spans into unified syntactic nodes prior to parsing eliminates spurious head-dependent arcs in Universal Dependencies (UD) trees for Uzbek.
2. **Neural Machine Translation (NMT) & LLMs:** Pre-identifying non-compositional idioms prevents literal word-by-word mistranslation and improves token efficiency in Large Language Model (LLM) prompting.
3. **Lexicography and Corpus Annotation:** Provides an automated pre-annotation tool for building large-scale phraseological dictionaries and treebanks for Turkic languages.

### 4.3. Community Adoption and Real-World Usage
Since its public deployment on May 29, 2026, `uzbek-mwe-tokenizer` has gained substantial traction within the computational linguistics community:
- **PyPI Package Adoption:** Recorded over **1,411 cumulative downloads** (including **432 direct `pip` client installations** excluding automated mirrors) as verified via PyPI download telemetry.
- **Hugging Face Model Hub:** Recorded **138 total model weight downloads** (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`), with steady monthly active usage by NLP practitioners and academic researchers.

---

## 5. Conclusions and Future Directions

`uzbek-mwe-tokenizer` bridges the gap between theoretical sequence-labeling research and practical software deployment for low-resource agglutinative languages. By combining a 108.5M-parameter UzBERT model with an automated cross-script normalization pipeline, the software completely eliminates apostrophe-induced subword fragmentation in Uzbek MWE identification. Future versions (`v0.2.0+`) will integrate Morphologically and Syntactically Informed Self-Attention (MISA) mechanisms to capture discontinuous MWEs (where external modifiers intervene between MWE components) and provide ONNX/quantized weights for ultra-low-latency edge deployment.

---

## Declaration of Competing Interest
The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgements
This research and software development were supported by the Computational Linguistics research group at Urgench State University named after Abu Rayhan Beruni and the Tashkent University of Information Technologies named after Muhammad al-Khwarizmi.

---

## References

1. Sharipov, M. S. (2026). Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. *Bulletin of TUIT: Management and Communication Technologies*, 2(14), 103–107. https://doi.org/10.61663/262tuitmct14
2. Bobojonova, L., et al. (2024). BBPOS: BERT-based Part-of-Speech Tagging for Uzbek. *RPTU Kaiserslautern-Landau & Saarland University*.
3. Constant, M., Eryiğit, G., Monti, J., van der Plas, L., Ramisch, C., Rosner, M., & Todirascu, A. (2017). Multiword Expression Processing: A Survey. *Computational Linguistics*, 43(4), 837–892. https://doi.org/10.1162/COLI_a_00302
4. Kuriyozov, E., et al. (2024). BERTbek: A Pretrained Language Model for Uzbek. *Proceedings of LREC-COLING 2024*.
5. Mansurov, B., & Mansurov, A. (2021). UzBERT: pretraining a BERT model for Uzbek. *arXiv preprint arXiv:2104.09459*.
6. Salaev, U., Kuriyozov, E., & Gómez-Rodríguez, C. (2022). A Machine Transliteration Tool Between Uzbek Alphabets. *CEUR Workshop Proceedings*, 3246, 42–49.
7. Miletić, F., & Schulte im Walde, S. (2024). Semantics of Multiword Expressions in Transformer-Based Models: A Survey. *Transactions of the Association for Computational Linguistics*, 12, 108–130.
