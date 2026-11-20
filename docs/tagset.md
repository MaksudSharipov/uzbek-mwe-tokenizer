# BIO Tagset and Covered Multi-Word Expression Categories

## 1. BIO Tagging Scheme

`UzbekMWETokenizer` uses the standard **BIO (Beginning, Inside, Outside)** sequence labeling scheme:

| Tag ID | BIO Tag | Description | Uzbek Explanation |
| :---: | :--- | :--- | :--- |
| `0` | `O` | Outside any multi-word expression | Oddiy (mustaqil) so‘z |
| `1` | `B-MWE` | Beginning token of a multi-word expression | Ko‘p so‘zli birikmaning birinchi so‘zi |
| `2` | `I-MWE` | Continuation (Inside) token of a multi-word expression | Ko‘p so‘zli birikmaning davom etuvchi qismi |

## 2. Linguistic Categories of Uzbek MWEs Covered

1. **Idiomatic Expressions (Frazeologizmlar / Iboralar):**
   - *tarvuzi qo‘ltig‘idan tushdi* (`тарвузи қўлтиғидан тушди`) — to lose hope / become discouraged
   - *kapalagi uchdi* (`капалаги учди`) — to be startled
   - *tepa sochi tik bo‘ldi* (`тепа сочи тик бўлди`) — to be horrified
2. **Auxiliary Verb Constructions (Ko‘makchi fe’lli so‘z qo‘shilmalari):**
   - *sotib olmoq* (`сотиб олмоқ`), *o‘qib chiqmoq* (`ўқиб чиқмоқ`), *aytib bermoq* (`айтиб бермоқ`)
3. **Compound Verbs & Nominal-Verbal Predicates (Qo‘shma fe’llar):**
   - *qaror qilmoq* (`қарор қилмоқ`), *yordam bermoq* (`ёрдам бермоқ`), *e’tibor qaratmoq* (`эътибор қаратмоқ`)
4. **Compound Nouns, Adverbs, and Pronouns (Qo‘shma ot, ravish va olmoshlar):**
   - *ota-ona*, *har kim*, *hech qachon*, *bir pasda*
