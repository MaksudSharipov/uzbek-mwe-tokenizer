# Usage

This document explains how to use **UzbekMWETokenizer** (`uzbek-mwe-tokenizer`) for multi-word expression (MWE) extraction and BIO sequence tagging in Uzbek texts.

## Basic Python API (Latin Script)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
mwes = tokenizer.extract_mwe(text)

print(mwes)
```

Expected output:

```python
[{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
```

## Cyrillic Script Mode (`mode="cyr"`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="cyr")

sentence = "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди."
print(tokenizer.extract_mwe(sentence))
```

Expected output:

```python
[
    {'mwe': 'капалаги учди', 'confidence': 100.0},
    {'mwe': 'ерга кирса', 'confidence': 76.0},
    {'mwe': 'қулоғидан тортиб чиқарадиган', 'confidence': 99.0}
]
```

## Output Formats (`mwe`, `slash`, `conll`, `json`)

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

tokenizer = UzbekMWETokenizer(mode="lot", quiet=True)
text = "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi."

# 1. Extracted MWE list
print(tokenizer(text, output_format="mwe"))

# 2. Slash format
print(tokenizer(text, output_format="slash"))
# Natijani/O ko'rib,/O tarvuzi/B-MWE qo'ltig'idan/I-MWE tushdi./I-MWE

# 3. CoNLL two-column BIO format
print(tokenizer(text, output_format="conll"))
```

## Command-Line Interface (`uzbek-mwe`)

```bash
uzbek-mwe "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
uzbek-mwe "Қоронғи хонада унинг қаттиқ капалаги учди." --mode cyr
uzbek-mwe "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." --format slash
uzbek-mwe "Natijani ko'rib, tarvuzi qo'ltig'idan tushdi." --format conll
```
