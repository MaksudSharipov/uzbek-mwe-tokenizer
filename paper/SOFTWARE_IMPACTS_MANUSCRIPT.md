# UzbekMWETokenizer: A Python library for Transformer-based multi-word expression identification in Uzbek

**Maksud Sharipov** $^{a,b,*}$  
$^{a}$ *Department of Convergence of Digital Technologies, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi, Urgench Branch, Urgench 220100, Uzbekistan*  
$^{b}$ *Department of Computer Science and Artificial Intelligence Technologies, Urgench State University named after Abu Rayhan Biruni, Urgench 220100, Uzbekistan*  
*ORCID:* [https://orcid.org/0000-0002-2363-6533](https://orcid.org/0000-0002-2363-6533)  
$^{*}$ *Corresponding author. E-mail address:* `maqsbek72@gmail.com`

---

## Abstract
`uzbek-mwe-tokenizer` (`UzbekMWETokenizer`) is an open-source Python library for identifying and extracting multi-word expressions (MWEs) in Uzbek texts using a fine-tuned UzBERT transformer model. In agglutinative Uzbek, MWEs such as idioms, compound verbs, and auxiliary verb constructions create syntactic ambiguities, while Latin apostrophe modifiers cause severe subword fragmentation. The package standardizes apostrophe variants, applies bidirectional Latin–Cyrillic script projection for monographemic inference, aligns subword predictions to full words, and filters contiguous multi-token phrases ($\ge 2$ words) with confidence scores. Validated on 11,936 sentences (14,285 MWEs), it achieves 0.9559 accuracy and 0.8722 F1-score without boundary loss, supporting downstream parsing, lemmatization, and machine translation.

## Keywords
Multi-word expressions; Uzbek NLP; Tokenization; Transformer; UzBERT; Syntactic disambiguation

---

## Code metadata

| Nr | Code metadata description | Metadata |
| :--- | :--- | :--- |
| C1 | Current code version | v0.1.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/MaksudSharipov/uzbek-mwe-tokenizer |
| C3 | Permanent link to reproducible capsule | https://pypi.org/project/uzbek-mwe-tokenizer/ |
| C4 | Legal code license | MIT License |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools and services used | Python, PyTorch, Hugging Face Transformers, SentencePiece, UzTransliterator |
| C7 | Compilation requirements, operating environments and dependencies | OS-independent (Linux, macOS, Windows); Python >= 3.6; dependencies: torch, transformers, UzTransliterator, sentencepiece |
| C8 | If available, link to developer documentation/manual | https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/blob/main/README.md |
| C9 | Support email for questions | maqsbek72@gmail.com |

---

## 1. High-level functionality and purpose of the software

In natural language processing (NLP) for strongly agglutinative Turkic languages such as Uzbek, accurate syntactic and semantic analysis relies heavily on how lexical boundaries are defined during initial tokenization [1–4]. Uzbek texts contain a high density of multi-word expressions (MWEs)—including auxiliary verb constructions (*sotib olmoq* [to buy]), nominal-verbal compound predicates (*qaror qilmoq* [to decide]), open compound nouns [5], and idiomatic expressions (*tarvuzi qo‘ltig‘idan tushdi* [to lose hope]). Although written as multiple space-separated words, an MWE functions as a single semantic and syntactic unit [6]. Traditional word-by-word tokenizers split MWEs into isolated tokens, leading to erroneous dependency parse trees, inaccurate part-of-speech (POS) and lemma assignments [1–5], and literal mistranslations in machine translation systems.

A further practical obstacle in Uzbek computational linguistics is the concurrent use of Latin and Cyrillic scripts. In the Uzbek Latin alphabet, letters such as *o‘*, *g‘*, and the glottal stop (*’*) use apostrophe-like modifiers (`’`, `‘`, `'`, ````, `ʻ`). Standard subword tokenizers (WordPiece and Byte-Pair Encoding) underlying modern Transformer architectures [7, 8] misinterpret these apostrophes as punctuation boundaries, splitting a single inflected word (e.g., *qo‘ltig‘idan*) into meaningless subword fragments (`qo` + `'` + `ltig` + `'` + `idan`) and breaking `I-MWE` (Inside) sequence tags [6].

**`uzbek-mwe-tokenizer`** (`UzbekMWETokenizer`) is an open-source Python software package distributed via PyPI (`https://pypi.org/project/uzbek-mwe-tokenizer/`) that resolves both syntactic MWE ambiguity and Latin apostrophe subword fragmentation. It wraps our fine-tuned 108.5M-parameter `UzBERT-Uncased` token classification model hosted on the Hugging Face Hub (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) [9, 10] inside an automated cross-script pipeline.

As shown in Listing 1, the software architecture (`uzbek_mwe_tokenizer/tokenizer.py`) consists of three primary functional methods:
1. **Apostrophe Normalization and Monographemic Projection (`_prepare_text`):** Unifies diverse Unicode apostrophes (`’`, `‘`, ````, `ʻ`) to a standard ASCII quote (`'`) and, when initialized in Latin mode (`mode="lot"`), transliterates input text into monographemic Cyrillic (`o‘` $\rightarrow$ `ў`, `g‘` $\rightarrow$ `ғ`, `sh` $\rightarrow$ `ш`, `ch` $\rightarrow$ `ч`) via `UzTransliterator` [11].
2. **Subword-Aligned BIO Sequence Inference (`extract_mwe`):** Tokenizes pre-split words with `is_split_into_words=True`, evaluates contextual logits and Softmax probabilities over `{0: O, 1: B-MWE, 2: I-MWE}`, and uses `inputs.word_ids()` to map first-subword predictions back to complete orthographic words (`full_word = words[word_idx]`).
3. **Phrase Filtering and Script Restitution (`save_mwe` and `_format_output_text`):** Aggregates contiguous `B-MWE` and `I-MWE` words, retains valid multi-word units (`len(current_mwe_words) >= 2`), computes their arithmetic mean confidence score ($0$–$100\%$), and converts extracted phrases back to Latin script when `mode="lot"`.

```python
# Listing 1: Core implementation of UzbekMWETokenizer (uzbek_mwe_tokenizer/tokenizer.py)
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
                mwe_str_formatted = self._format_output_text(" ".join(current_mwe_words))
                extracted_mwes.append({"mwe": mwe_str_formatted, "confidence": avg_score})

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
                    current_mwe_words.append(full_word)
                    current_mwe_scores.append(score)
                else:
                    save_mwe()
                    current_mwe_words, current_mwe_scores = [], []
            previous_word_idx = word_idx
        save_mwe()
        return extracted_mwes
```

The package is installed via `pip install uzbek-mwe-tokenizer` and supports both scripts out of the box (Listing 2).

```python
# Listing 2: Usage example in Latin and Cyrillic scripts
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer_lot = UzbekMWETokenizer(mode="lot")
text_lat = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
print(tokenizer_lot.extract_mwe(text_lat))
# Output: [{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]

tokenizer_cyr = UzbekMWETokenizer(mode="cyr")
text_cyr = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
print(tokenizer_cyr.extract_mwe(text_cyr))
# Output: [{'mwe': 'капалаги учди', 'confidence': 100.0},
#          {'mwe': 'ерга кирса', 'confidence': 76.0},
#          {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}]
```

---

## 2. Impact overview and achieved results

### 2.1. Empirical validation and advancement of existing research questions
The neural model deployed in `uzbek-mwe-tokenizer` was trained and benchmarked on an isomorphic Latin–Cyrillic Uzbek MWE corpus consisting of **11,936 sentences, 108,200 tokens, and 14,285 annotated MWEs** across six Transformer backbones (Table 1) [6].

*Table 1: Comparative evaluation of six Transformer architectures on the Uzbek MWE test set [6].*

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (`uzbek-mwe-tokenizer`)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

Our comparative investigation uncovered an "accuracy paradox" in Uzbek sequence labeling [6]: although Latin-based models (`BERTbek-News`) achieved a slightly higher quantitative F1-score due to fewer false positives on non-MWE (`O`) words, token-level confusion matrix analysis revealed that Latin tokenizers frequently truncate multi-word idioms at apostrophe positions. By routing inference through the monographemic Cyrillic `UzBERT-Uncased` backbone, `uzbek-mwe-tokenizer` successfully identified **1,342 `B-MWE` and 1,467 `I-MWE` tokens** without orthographic fragmentation, preserving complete phrase boundaries.

### 2.2. New research avenues and changes to daily NLP practice
`uzbek-mwe-tokenizer` opens several research directions and streamlines daily workflows for Turkic NLP researchers:
- **End-to-End Uzbek Preprocessing Ecosystem:** Together with our contextual POS tagger `UzbekTaggerBERT` [1], multi-domain POS dataset `UzbekPOS` [2], lemma/stem dataset [3], verb suffixation database [4], and open compound word lemmatization algorithms [5], this library provides a unified pipeline where MWEs are first collapsed into single lexical nodes before POS tagging, lemmatization, and syntactic dependency parsing.
- **Cross-Script Transfer Learning:** Demonstrates that deterministic orthographic projection into a monographemic script can bypass subword tokenizer limitations without retraining tokenizer vocabularies from scratch.
- **Widespread Community Adoption:** Since its public release on PyPI and the Hugging Face Hub, `uzbek-mwe-tokenizer` has accumulated over **1,411 cumulative PyPI package downloads** (including **207 downloads in the last 30 days alone**, with 58–65 direct `pip` client installations excluding mirrors) and **146 all-time Hugging Face model downloads** (**18 downloads in the last 30 days**), as verified via official PyPIStats and Hugging Face Hub telemetry. This steady monthly growth confirms active adoption by academic research groups, corpus linguists, and software developers outside the author's institution.

### 2.3. Scholarly publications and ongoing projects enabled by the software
The development and deployment of `uzbek-mwe-tokenizer` directly enabled the cross-script tokenization and syntactic disambiguation study published in [6], extends our open compound word lemmatization research [5], and serves as the foundational MWE pre-tokenization module in ongoing doctoral and national research projects on Morphologically and Syntactically Informed Self-Attention (MISA) for Uzbek syntactic parsing at Tashkent University of Information Technologies and Urgench State University.

---

## 3. Limitations and future improvements

In its current release (`v0.1.0`), `uzbek-mwe-tokenizer` identifies contiguous multi-word expressions (`B-MWE` immediately followed by `I-MWE` tokens) and relies on runtime transliteration for Latin inputs. When external words intervene between the components of a separable verbal phrase (discontinuous MWEs), standard BIO sequence tagging can split the expression. Future versions will integrate Morphologically and Syntactically Informed Self-Attention (MISA) to capture long-range discontinuous MWE components and provide quantized ONNX weights for low-latency CPU deployment.

---

## Acknowledgements
The author expresses gratitude to the Computational Linguistics research team at Urgench State University named after Abu Rayhan Biruni and Tashkent University of Information Technologies named after Muhammad al-Khwarizmi for corpus annotation support and computational resources.

---

## References

[1] M. Sharipov, UzbekTaggerBERT: A Python library for contextual part-of-speech tagging of Uzbek, *Software Impacts* 30 (2026) 100865. https://doi.org/10.1016/j.simpa.2026.100865.

[2] M. Sharipov, E. Kuriyozov, J. Vičič, UzbekPOS: A multi-domain dataset for Uzbek part-of-speech tagging, *Data in Brief* 66 (2026) 112640. https://doi.org/10.1016/j.dib.2026.112640.

[3] M. Sharipov, A POS-tagged dataset of Uzbek word lemmas and stems for morphological and NLP research, *Data in Brief* 68 (2026) 113072. https://doi.org/10.1016/j.dib.2026.113072.

[4] M. Sharipov, J. Vičič, Dataset of Uzbek verbs with formation and suffixes, *Data in Brief* 61 (2025) 111731. https://doi.org/10.1016/j.dib.2025.111731.

[5] M. Sharipov, A. Saidov, Development of a lemmatization algorithm: Interpreting open compound words, in: *2025 IEEE XVII International Scientific and Technical Conference on Actual Problems of Electronic Instrument Engineering (APEIE)*, IEEE, Novosibirsk, 2025, pp. 1–4. https://doi.org/10.1109/APEIE66761.2025.11289280.

[6] M. S. Sharipov, Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts, *Bulletin of TUIT: Management and Communication Technologies* 2 (14) (2026) 103–107. https://doi.org/10.61663/262tuitmct14.

[7] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, in: *Proceedings of NAACL-HLT*, 2019, pp. 4171–4186.

[8] T. Wolf, L. Debut, V. Sanh, et al., Transformers: State-of-the-art natural language processing, in: *Proceedings of EMNLP: System Demonstrations*, 2020, pp. 38–45.

[9] B. Mansurov, A. Mansurov, UzBERT: Pretraining a BERT model for Uzbek, *arXiv preprint arXiv:2104.09459* (2021).

[10] M. Sharipov, `uzbek-mwe-tokenizer` v0.1.0 (PyPI and Hugging Face Model Hub: `MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`), 2026. https://pypi.org/project/uzbek-mwe-tokenizer/.

[11] U. Salaev, E. Kuriyozov, C. Gómez-Rodríguez, A machine transliteration tool between Uzbek alphabets, *CEUR Workshop Proceedings* 3246 (2022) 42–49.
