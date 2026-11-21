# Usage

This document explains how to use **`uzbek-mwe-tokenizer`** (`UzbekMWETokenizer`) for Multi-Word Expression (MWE) identification in Uzbek texts.

## 1. Latin Script Mode (`mode="lot"`)

In Latin mode, the package standardizes apostrophes, decouples punctuation, projects input text to Cyrillic for monographemic inference, and restores extracted MWEs back to Latin script:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer_lot = UzbekMWETokenizer(mode="lot")
text_lat = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
print(tokenizer_lot.extract_mwe(text_lat))
```

Output:

```python
[{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 99.98}]
```

## 2. Cyrillic Script Mode (`mode="cyr"`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer_cyr = UzbekMWETokenizer(mode="cyr")
text_cyr = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
print(tokenizer_cyr.extract_mwe(text_cyr))
```

Output:

```python
[
    {'mwe': 'капалаги учди', 'confidence': 99.99},
    {'mwe': 'ерга кирса', 'confidence': 76.42},
    {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 98.08}
]
```
