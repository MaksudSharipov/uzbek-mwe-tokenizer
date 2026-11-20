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
- PyPI, CLI, unit tests, CI, and benchmarks support reproducible reuse.

---

## Abstract
UzbekMWETokenizer (distributed on PyPI as `uzbek-mwe-tokenizer`) is an open-source Python library for identifying and extracting multi-word expressions (MWEs) in Uzbek text using a fine-tuned UzBERT model. The package normalizes common apostrophe variants, applies bidirectional Latin–Cyrillic script projection to eliminate subword fragmentation, aligns Transformer subword predictions with original orthographic words, aggregates BIO (`B-MWE`, `I-MWE`, `O`) sequence probabilities, and filters contiguous multi-token phrases. It provides Python and command-line interfaces supporting phrase dictionary, slash, JSON, and CoNLL-like BIO outputs. Evaluated on an isomorphic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, and 14,285 MWEs), the integrated model achieved 0.9559 accuracy, 0.8926 recall, and 0.8722 F1-score without boundary truncation. The release includes documentation, unit tests, CI workflows, benchmarks, and PyPI distribution, recording over 1,411 PyPI downloads and 138 Hugging Face model downloads.

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
| **C7** | **Software code languages, tools, and services used** | Python ($\ge 3.8$), PyTorch, Hugging Face Transformers, SentencePiece, UzTransliterator |
| **C8** | **Compilation requirements, operating environments & dependencies** | OS-independent (Linux, macOS, Windows); dependencies: `torch>=1.9.0`, `transformers>=4.20.0`, `UzTransliterator>=0.0.18`, `sentencepiece>=0.1.96` |
| **C9** | **If available, link to developer documentation/manual** | [https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/tree/main/docs](https://github.com/MaksudSharipov/uzbek-mwe-tokenizer/tree/main/docs) |
| **C10** | **Support email for questions** | `maqsbek72@gmail.com` |

---

## 1. Motivation and Background

Uzbek is a strongly agglutinative Turkic language in which grammatical relations and derivational nuances are expressed through complex suffixation chains attached to word stems [1, 2]. Alongside morphological complexity, Uzbek texts abound in multi-word expressions (MWEs)—including auxiliary verb constructions (*sotib olmoq* [to buy]), compound predicates (*qaror qilmoq* [to decide]), and idiomatic phrases (*tarvuzi qo‘ltig‘idan tushdi* [to lose hope])—which consist of multiple space-separated words but function semantically and syntactically as a single lexical unit [3]. When conventional tokenizers and word-level NLP pipelines process these units as isolated tokens, downstream dependency parsers construct erroneous syntactic trees and machine translation engines generate literal, distorted translations.

A major practical barrier to Transformer-based MWE identification in Uzbek is its dual-script orthography (Latin and Cyrillic) and the heterogeneous representation of apostrophes in Latin texts (`’`, `‘`, `'`, `ʻ`, ````). Standard subword tokenizers (WordPiece and BPE) used in pre-trained language models [4, 5] treat Latin apostrophes in digraphs such as *o‘* and *g‘* as punctuation delimiters. This splits a single inflected word (e.g., *qo‘ltig‘idan*) into disconnected subwords (`qo` + `'` + `ltig` + `'` + `idan`), breaking the continuity of `I-MWE` (Inside) sequence tags.

Building upon our recent open-source Uzbek NLP ecosystem—specifically `UzbekTaggerBERT` for contextual part-of-speech tagging [2] and our cross-script MWE study [3]—we present **UzbekMWETokenizer** (`uzbek-mwe-tokenizer`). The library encapsulates a fine-tuned 108.5M-parameter `UzBERT-Uncased` token classification model (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) [6] within a modular Python package and command-line tool that automates apostrophe normalization, monographemic cross-script inference, subword-to-word alignment, and multi-format MWE export.

---

## 2. Software Description and Architecture

### 2.1. Package Structure and Workflow
`UzbekMWETokenizer` is structured as a modular, reproducible Python package matching modern research software engineering standards:

- `uzbek_mwe_tokenizer/tokenizer.py` — core `UzbekMWETokenizer` class implementing model loading, cross-script projection, subword-aligned BIO inference, and output formatting (`mwe`, `slash`, `conll`, `json`).
- `uzbek_mwe_tokenizer/text_utils.py` — apostrophe normalization (`normalize_apostrophes`) and orthographic word splitting (`split_words`).
- `uzbek_mwe_tokenizer/label_map.py` — BIO tag definitions (`O`, `B-MWE`, `I-MWE`) and bijective ID mappings.
- `uzbek_mwe_tokenizer/cli.py` — command-line interface (`uzbek-mwe`) supporting script selection and multiple output formats.
- `examples/`, `tests/`, `benchmark/`, and `docs/` — runnable examples, `pytest` unit/integration tests, CoNLL evaluation and speed-test scripts, and user documentation.

The end-to-end processing pipeline executes in five deterministic steps:

1. **Text Normalization (`text_utils.py`):** Standardizes all Unicode apostrophe variants (`‘`, `’`, ````, `´`, `ʼ`, `ʻ`) into a canonical form (`'`) and normalizes whitespace.
2. **Cross-Script Monographemic Projection (`_prepare_text`):** When operating in Latin mode (`mode="lot"`), input text is transliterated into monographemic Cyrillic (`o‘` $\rightarrow$ `ў`, `g‘` $\rightarrow$ `ғ`, `sh` $\rightarrow$ `ш`, `ch` $\rightarrow$ `ч`) via `UzTransliterator` [7], preventing subword tokenizers from splitting words at apostrophes.
3. **Subword Tokenization and Index Alignment:** Pre-split orthographic words are passed to the fine-tuned `UzBERT-Uncased` tokenizer with `is_split_into_words=True`, and `inputs.word_ids()` maps each subword token back to its parent word index.
4. **Contextual BIO Prediction and Confidence Aggregation:** The 12-layer Transformer (`108,493,059` parameters) predicts logits over `{0: O, 1: B-MWE, 2: I-MWE}`. Softmax probabilities are extracted at word boundaries, and contiguous `B-MWE` + `I-MWE` spans of length $\ge 2$ words are assembled with their arithmetic mean confidence score.
5. **Bidirectional Script Restitution (`_format_output_text`):** Extracted MWE spans and word-level BIO sequences are transparently restored to the user's native script (`lot` or `cyr`).

### 2.2. Core Implementation
Listing 1 shows the core word-aligned BIO extraction and formatting methods in `uzbek_mwe_tokenizer/tokenizer.py`.

```python
# Listing 1: Subword-aligned MWE extraction and multi-format callable interface
def extract_mwe(self, raw_text: str) -> List[Dict[str, Union[str, float]]]:
    if not raw_text.strip():
        return []

    text = self._prepare_text(raw_text)
    words = text.split()

    inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = inputs.word_ids()
    encoded = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = self.model(**encoded)

    logits = outputs.logits.detach().cpu()
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    probabilities = torch.nn.functional.softmax(logits, dim=2)[0]

    extracted_mwes, current_mwe_words, current_mwe_scores = [], [], []
    previous_word_idx = None

    def save_mwe() -> None:
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

### 3.1. Installation and Python API
The library is installed from PyPI via `pip install uzbek-mwe-tokenizer`. Listing 2 demonstrates MWE extraction in both Latin and Cyrillic scripts, as well as word-level BIO tagging in `slash` and `conll` formats.

```python
# Listing 2: Python API usage across scripts and output formats
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# 1. Latin script MWE extraction
tokenizer_lot = UzbekMWETokenizer(mode="lot", quiet=True)
text_lat = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
print(tokenizer_lot.extract_mwe(text_lat))
# Output: [{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]

# 2. Word-level BIO sequence in slash format
print(tokenizer_lot(text_lat, output_format="slash"))
# Output: Natijani/O ko'rib,/O hamma/O o'quvchilarning/O birdaniga/O tarvuzi/B-MWE qo'ltig'idan/I-MWE tushdi./I-MWE

# 3. Cyrillic script MWE extraction (multiple MWEs in a single sentence)
tokenizer_cyr = UzbekMWETokenizer(mode="cyr", quiet=True)
text_cyr = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
print(tokenizer_cyr.extract_mwe(text_cyr))
# Output: [{'mwe': 'капалаги учди', 'confidence': 100.0},
#          {'mwe': 'ерга кирса', 'confidence': 76.0},
#          {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}]
```

### 3.2. Command-Line Interface (`uzbek-mwe`)
Upon installation, the package registers the `uzbek-mwe` console entry point:

```bash
# Default MWE dictionary extraction (JSON output)
uzbek-mwe "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."

# Word-level BIO tagging in CoNLL format
uzbek-mwe "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." --format conll
```

---

## 4. Impact Overview

### 4.1. Empirical Validation and Benchmark
The neural backbone integrated into `UzbekMWETokenizer` (`MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT`) was trained and evaluated on an isomorphic Latin–Cyrillic Uzbek MWE corpus comprising **11,936 sentences, 108,200 tokens, and 14,285 annotated MWEs** using an 80/10/10 train/validation/test split [3]. Table 1 summarizes comparative benchmark results across six Transformer architectures.

*Table 1: Comparative evaluation of Transformer models on the Uzbek MWE test dataset [3].*

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (UzbekMWETokenizer)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |

Although Latin-script models (`BERTbek-News`) reached a slightly higher aggregate F1-score due to conservative `O`-class precision, token-level confusion matrix analysis revealed that Latin subword tokenizers fracture apostrophe-containing words, causing mid-expression `I-MWE` dropouts. By projecting inputs into monographemic Cyrillic during inference, `UzbekMWETokenizer` recovered **1,342 `B-MWE` and 1,467 `I-MWE` tokens** without boundary truncation.

### 4.2. Research and Industrial Impact
`UzbekMWETokenizer` complements `UzbekTaggerBERT` [2] to form an end-to-end open-source preprocessing stack for Uzbek computational linguistics:
1. **Syntactic Disambiguation:** Collapsing multi-word idioms and compound verbs into single syntactic units prior to dependency parsing eliminates spurious head-dependent attachments.
2. **Machine Translation and Corpus Annotation:** Prevents literal word-by-word mistranslation of non-compositional Uzbek idioms and accelerates semi-automatic treebank annotation.
3. **Verified Community Adoption:** Since its public release on PyPI and Hugging Face Hub, the package has accumulated **1,411 cumulative PyPI downloads** (432 direct client installations) and **138 Hugging Face model downloads**, confirming active reuse by researchers and developers.

---

## 5. Limitations and Future Developments

Currently, `UzbekMWETokenizer` targets contiguous multi-word expressions (`B-MWE` followed by immediate `I-MWE` tokens) and relies on an intermediate transliteration step for Latin texts. Future updates (`v0.2.0+`) will incorporate Morphologically and Syntactically Informed Self-Attention (MISA) to detect discontinuous MWEs (where external modifiers intervene between MWE components) and provide quantized ONNX weights for high-throughput CPU deployment.

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
