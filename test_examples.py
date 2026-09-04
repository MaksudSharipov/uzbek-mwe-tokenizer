from uzbek_mwe_tokenizer import UzbekMWETokenizer

print("=== CYRILLIC MODE TEST ===")
tokenizer_cyr = UzbekMWETokenizer(mode="cyr")

sentences = [
    "Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.",
    "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.",
    "Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади."
]

for gap in sentences:
    print("\nGap:", gap)
    mwes = tokenizer_cyr.extract_mwe(gap)
    print("Topilgan MWE'lar:", mwes)

print("\n\n=== LATIN MODE TEST ===")
tokenizer_lot = UzbekMWETokenizer(mode="lot")

latin_sentences = [
    "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi.",
    "Qorong'i xonada uning qattiq kapalagi uchdi, go'yo yerga kirsa, qulog'idan tortib chiqaradigan holat edi.",
    "Raqiblarimizni ko'rib bizning aslo tepa sochimiz tik bo'lmadi."
]

for gap in latin_sentences:
    print("\nGap:", gap)
    mwes = tokenizer_lot.extract_mwe(gap)
    print("Topilgan MWE'lar:", mwes)
