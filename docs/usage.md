# Usage

This document explains how to use **`uzbek-mwe-tokenizer`** (`UzbekMWETokenizer`) for Multi-Word Expression (MWE) identification in Uzbek texts.

## 1. Latin Script Mode (`mode="lot"`)

In Latin mode, the package normalizes apostrophes, transliterates input text to Cyrillic for neural inference, and translates extracted MWEs back to Latin script:

```python
from uzbek_mwe_tokenizer import UzbekMWETokenizer

# Initialize tokenizer in Latin mode
tokenizer = UzbekMWETokenizer(mode="lot")

text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
mwes = tokenizer.extract_mwe(text)

print(mwes)
```

Expected output:

```python
[{'mwe': "tarvuzi qo'ltig'idan tushdi", 'confidence': 100.0}]
```

## 2. Cyrillic Script Mode (`mode="cyr"`)

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
