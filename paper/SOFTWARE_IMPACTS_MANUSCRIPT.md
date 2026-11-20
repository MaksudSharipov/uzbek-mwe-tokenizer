# UzbekMWETokenizer: A Python library for Transformer-based multi-word expression identification in Uzbek

**Maksud Sharipov** $^{a,b,*}$  
$^{a}$ *Tashkent University of Information Technologies named after Muhammad al-Khwarizmi, Uzbekistan*  
$^{b}$ *Urgench State University named after Abu Rayhan Biruni, Urgench, Uzbekistan*  
*ORCID:* [https://orcid.org/0000-0002-2363-6533](https://orcid.org/0000-0002-2363-6533)  
$^{*}$ *Corresponding author. E-mail address:* `maqsbek72@gmail.com`

---

## Highlights
- Open-source Python library identifies multi-word expressions in Uzbek text.
- Cross-script normalization resolves Latin apostrophe subword fragmentation.
- Subword-to-word alignment maps UzBERT BIO logits to full lexical spans.
- Evaluated on 11,936 sentences achieving 0.9559 accuracy and 0.8722 F1-score.
- PyPI package, unit tests, and benchmarks support reproducible reuse.

---

## Abstract
`uzbek-mwe-tokenizer` (`UzbekMWETokenizer`) is an open-source Python library for identifying and extracting multi-word expressions (MWEs) in Uzbek text using a fine-tuned UzBERT model (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`). The package standardizes common apostrophe variants, applies bidirectional Latin–Cyrillic script projection to eliminate subword fragmentation, aligns Transformer subword predictions with original orthographic words via `word_ids()`, aggregates BIO (`B-MWE`, `I-MWE`, `O`) sequence probabilities, and filters contiguous multi-token phrases ($\ge 2$ words). Evaluated on an isomorphic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, and 14,285 MWEs), the integrated model achieved 0.9559 accuracy, 0.8926 recall, and 0.8722 F1-score without boundary truncation. The release includes documentation, tests, benchmarks, and PyPI distribution, recording over 1,411 PyPI downloads and 138 Hugging Face model downloads.

**Keywords:** Multi-word expressions; Uzbek NLP; Tokenization; Transformer; UzBERT; Sequence labeling.

---

## Code Metadata

| Nr. | Code metadata description | Metadata |
| :--- | :--- | :--- |
| **C1** | **Current code version** | `v0.1.0` |
| **C2** | **Permanent link to code/repository used for this code version** | [https://github.com/MaksudSharipov/uzbek-mwe-tokenizer](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer) |
| **C3** | **Permanent link to Reproducible Capsule / Package Archive** | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| **C4** | **Pre-trained model hub repository** | [https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT](https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT) |
| **C5** | **Legal Code License** | MIT License |
| **C6** | **Code versioning system used** | Git |
| **C7** | **Software code languages, tools, and services used** | Python ($\ge 3.6$), PyTorch, Hugging Face Transformers, SentencePiece, UzTransliterator |
| **C8** | **Compilation requirements, operating environments & dependencies** | OS-independent (Linux, macOS, Windows); dependencies: `torch`, `transformers`, `UzTransliterator`, `sentencepiece` |
| **C9** | **If available, link to developer documentation/manual** | [https://pypi.org/project/uzbek-mwe-tokenizer/](https://pypi.org/project/uzbek-mwe-tokenizer/) |
| **C10** | **Support email for questions** | `maqsbek72@gmail.com` |

---

## 1. Motivation and Background

Uzbek is a strongly agglutinative Turkic language in which grammatical relations and derivational nuances are expressed through complex suffixation chains attached to word stems [1, 2]. Alongside morphological complexity, Uzbek texts abound in multi-word expressions (MWEs)—including auxiliary verb constructions (*sotib olmoq* [to buy]), compound predicates (*qaror qilmoq* [to decide]), and idiomatic phrases (*tarvuzi qo‘ltig‘idan tushdi* [to lose hope])—which consist of multiple space-separated words but function semantically and syntactically as a single lexical unit [3]. When conventional tokenizers and word-level NLP pipelines process these units as isolated tokens, downstream dependency parsers construct erroneous syntactic trees and machine translation engines generate literal, distorted translations.

A major practical barrier to Transformer-based MWE identification in Uzbek is its dual-script orthography (Latin and Cyrillic) and the heterogeneous representation of apostrophes in Latin texts (`’`, `‘`, `'`, ````, `ʻ`). Standard subword tokenizers (WordPiece and BPE) used in pre-trained language models [4, 5] treat Latin apostrophes in digraphs such as *o‘* and *g‘* as punctuation delimiters. This splits a single inflected word (e.g., *qo‘ltig‘idan*) into disconnected subwords (`qo` + `'` + `ltig` + `'` + `idan`), breaking the continuity of `I-MWE` (Inside) sequence tags.

Building upon our recent open-source Uzbek NLP ecosystem—specifically `UzbekTaggerBERT` for contextual part-of-speech tagging [2] and our cross-script MWE study [3]—we developed **`uzbek-mwe-tokenizer`** (`UzbekMWETokenizer`). The library encapsulates a fine-tuned 108.5M-parameter `UzBERT-Uncased` token classification model (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) [6, 8] within a lightweight Python package that automates apostrophe normalization, monographemic cross-script inference, subword-to-word alignment, and phrase confidence scoring.

---

## 2. Software Description and Architecture

### 2.1. Package Structure and Workflow
`uzbek-mwe-tokenizer` is organized around the `UzbekMWETokenizer` class defined in `uzbek_mwe_tokenizer/tokenizer.py`. The end-to-end processing pipeline executes in four sequential stages:

1. **Initialization and Model Provisioning (`__init__`):** Validates the operating script mode (`mode="lot"` for Latin or `mode="cyr"` for Cyrillic), initializes `UzTransliterator` [7], and loads the fine-tuned `MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT` tokenizer and token classification model from the Hugging Face Hub.
2. **Apostrophe Normalization and Cross-Script Projection (`_prepare_text`):** Unifies disparate Unicode apostrophe variants (`’`, `‘`, ````, `ʻ`) to a canonical ASCII quote (`'`). When operating in Latin mode (`mode="lot"`), the input text is transliterated into monographemic Cyrillic (`o‘` $\rightarrow$ `ў`, `g‘` $\rightarrow$ `ғ`, `sh` $\rightarrow$ `ш`, `ch` $\rightarrow$ `ч`), preventing WordPiece tokenization from splitting words at apostrophes.
3. **Subword-Aligned BIO Inference (`extract_mwe`):** Whitespace-delimited words are passed to the tokenizer with `is_split_into_words=True`. The 12-layer Transformer (`108,493,059` parameters) computes logits and Softmax probabilities over `{0: O, 1: B-MWE, 2: I-MWE}`. Using `inputs.word_ids()`, predictions from the first subword of each word are aligned back to full orthographic words (`full_word = words[word_idx]`).
4. **MWE Filtering and Script Restitution (`save_mwe` and `_format_output_text`):** Contiguous `B-MWE` and `I-MWE` word spans containing at least two words (`len(current_mwe_words) >= 2`) are retained, their mean confidence score is computed, and the extracted expression is converted back to Latin script when `mode="lot"`.

### 2.2. Core Implementation
Listing 1 presents the exact implementation of `UzbekMWETokenizer` in `uzbek_mwe_tokenizer/tokenizer.py` (version `0.1.0` on PyPI).

```python
# Listing 1: Core implementation in uzbek_mwe_tokenizer/tokenizer.py (v0.1.0)
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from UzTransliterator import UzTransliterator

class UzbekMWETokenizer:
    def __init__(self, mode="lot"):
        self.mode = mode.lower()
        if self.mode not in ["lot", "cyr"]:
            raise ValueError("Mode must be either 'lot' (Latin) or 'cyr' (Cyrillic).")
            
        self.model_name = "MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.id2label = self.model.config.id2label
        self.transliterator = UzTransliterator.UzTransliterator()

    def _prepare_text(self, text):
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("ʻ", "'")
        if self.mode == "lot":
            text = self.transliterator.transliterate(text, from_="lat", to="cyr")
        return text

    def _format_output_text(self, text):
        if self.mode == "lot":
            return self.transliterator.transliterate(text, from_="cyr", to="lat")
        return text

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
                mwe_str = " ".join(current_mwe_words)
                mwe_str_formatted = self._format_output_text(mwe_str)
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

The package is installed directly from PyPI via `pip install uzbek-mwe-tokenizer`. Listing 2 demonstrates MWE identification in both Cyrillic (`mode="cyr"`) and Latin (`mode="lot"`) scripts (`test_examples.py`).

```python
# Listing 2: MWE extraction in Cyrillic and Latin modes
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# 1. Cyrillic mode
tokenizer_cyr = UzbekMWETokenizer(mode="cyr")
sentences_cyr = [
    "Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.",
    "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.",
    "Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади."
]
for gap in sentences_cyr:
    print("Gap:", gap)
    print("Topilgan MWE'lar:", tokenizer_cyr.extract_mwe(gap))

# 2. Latin mode
tokenizer_lot = UzbekMWETokenizer(mode="lot")
text_lat = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
print("Latin MWE'lar:", tokenizer_lot.extract_mwe(text_lat))
```

**Expected Output:**
```text
Gap: Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.
Topilgan MWE'lar: [{'mwe': 'тарвузи қўлтиғидан тушди', 'confidence': 100.0}]
Gap: Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.
Topilgan MWE'lar: [{'mwe': 'капалаги учди', 'confidence': 100.0}, {'mwe': 'ерга кирса', 'confidence': 76.0}, {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}]
Gap: Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади.
Topilgan MWE'lar: [{'mwe': 'тепа сочимиз тик бўлмади', 'confidence': 100.0}]
Latin MWE'lar: [{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
```

---

## 4. Impact Overview

### 4.1. Empirical Validation and Benchmark
The neural backbone integrated into `uzbek-mwe-tokenizer` (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) was trained and evaluated on an isomorphic Latin–Cyrillic Uzbek MWE corpus comprising **11,936 sentences, 108,200 tokens, and 14,285 annotated MWEs** using an 80/10/10 train/validation/test split [3]. Table 1 summarizes comparative benchmark results across six Transformer architectures.

*Table 1: Comparative evaluation of Transformer models on the Uzbek MWE test dataset [3].*

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (`uzbek-mwe-tokenizer`)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

Although Latin-script models (`BERTbek-News`) reached a slightly higher aggregate F1-score due to conservative `O`-class precision, token-level confusion matrix analysis revealed that Latin subword tokenizers fracture apostrophe-containing words, causing mid-expression `I-MWE` dropouts. By projecting inputs into monographemic Cyrillic during inference, `uzbek-mwe-tokenizer` recovered **1,342 `B-MWE` and 1,467 `I-MWE` tokens** without boundary truncation.

### 4.2. Research and Industrial Impact
`uzbek-mwe-tokenizer` complements `UzbekTaggerBERT` [2] to form an open-source preprocessing stack for Uzbek computational linguistics:
1. **Syntactic Disambiguation:** Identifying multi-word idioms and compound verbs prior to dependency parsing prevents spurious head-dependent attachments.
2. **Machine Translation and Corpus Annotation:** Prevents literal word-by-word mistranslation of non-compositional Uzbek idioms and accelerates semi-automatic treebank annotation.
3. **Verified Community Adoption:** Since its public release on PyPI and Hugging Face Hub, the package has accumulated **1,411 cumulative PyPI downloads** (432 direct client installations) and **138 Hugging Face model downloads**, confirming active reuse by researchers and developers.

---

## 5. Limitations and Future Developments

Currently, `uzbek-mwe-tokenizer` targets contiguous multi-word expressions (`B-MWE` followed by immediate `I-MWE` tokens) and relies on an intermediate transliteration step for Latin texts. Future updates will incorporate Morphologically and Syntactically Informed Self-Attention (MISA) to detect discontinuous MWEs and provide quantized ONNX weights for high-throughput CPU deployment.

---

## Declaration of Competing Interest
The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgements
This work was supported by the Computational Linguistics research group at Urgench State University named after Abu Rayhan Biruni and Tashkent University of Information Technologies named after Muhammad al-Khwarizmi.

---

## References

1. Sharipov, M., Kuriyozov, E., & Vičič, J. (2026). UzbekPOS: A multi-domain dataset for Uzbek part-of-speech tagging. *Data in Brief*, 66, 112640. https://doi.org/10.1016/j.dib.2026.112640
2. Sharipov, M. (2026). UzbekTaggerBERT: A Python library for contextual part-of-speech tagging of Uzbek. *Software Impacts*, 30, 100865. https://doi.org/10.1016/j.simpa.2026.100865
3. Sharipov, M. S. (2026). Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. *Bulletin of TUIT: Management and Communication Technologies*, 2(14), 103–107. https://doi.org/10.61663/262tuitmct14
4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proc. NAACL-HLT*, 4171–4186.
5. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. *Proc. EMNLP: System Demonstrations*, 38–45.
6. Mansurov, B., & Mansurov, A. (2021). UzBERT: Pretraining a BERT model for Uzbek. *arXiv preprint arXiv:2104.09459*.
7. Salaev, U., Kuriyozov, E., & Gómez-Rodríguez, C. (2022). A machine transliteration tool between Uzbek alphabets. *CEUR Workshop Proceedings*, 3246, 42–49.
8. Hugging Face Model Hub (2026). `MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`. https://huggingface.co/MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT
