# Benchmark

This directory contains evaluation and inference-speed scripts for **UzbekMWETokenizer** (`uzbek-mwe-tokenizer`).

## Files

- `evaluate.py` — evaluates the BIO sequence tagger on a two-column CoNLL-like MWE test file.
- `speed_test.py` — measures approximate runtime, sentences/second, and tokens/second.
- `results.csv` — stores comparative benchmark results across six Transformer architectures for the *Software Impacts* manuscript.

## Evaluation input format

The expected test file format is two tab-separated columns (`TOKEN\tBIO_TAG`), with sentences separated by blank lines:

```text
Натижани	O
кўриб,	O
ҳамма	O
ўқувчиларнинг	O
бирданига	O
тарвузи	B-MWE
қўлтиғидан	I-MWE
тушди.	I-MWE
```

## Run evaluation

```bash
python benchmark/evaluate.py --test-file path/to/test.conll --output benchmark/eval_summary.csv
```

## Run speed test

```bash
python benchmark/speed_test.py
```

## Model Evaluation Results

Models were evaluated on an isomorphic Latin–Cyrillic Uzbek MWE corpus (11,936 sentences, 108,200 tokens, and 14,285 MWEs) using an 80/10/10 train/validation/test split:

| Model Architecture | Script | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BERTbek-News | Latin | 0.9602 | 0.8632 | 0.9011 | 0.8817 |
| UzRoBERTa-v2 | Latin | 0.9582 | 0.8659 | 0.8947 | 0.8801 |
| mBERT-Cased | Latin | 0.9531 | 0.8619 | 0.8954 | 0.8783 |
| XLM-RoBERTa | Latin | 0.9568 | 0.8591 | 0.8919 | 0.8752 |
| **UzBERT-Uncased (UzbekMWETokenizer)** | **Cyrillic** | **0.9559** | **0.8528** | **0.8926** | **0.8722** |
| Tahrirchi-BERT | Latin | 0.9541 | 0.8597 | 0.8834 | 0.8714 |
