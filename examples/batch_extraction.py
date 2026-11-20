"""Multiple-sentence MWE extraction example for UzbekMWETokenizer.

Run:
    python examples/batch_extraction.py
"""

from uzbek_mwe_tokenizer import UzbekMWETokenizer


def main() -> None:
    tokenizer = UzbekMWETokenizer(mode="cyr")

    sentences = [
        "Натижани кўриб, ҳамма ўқувчиларнинг бирданига тарвузи қўлтиғидан тушди.",
        "Қоронғи хонада унинг қаттиқ капалаги учди, гўё ерга кирса, қулоғидан тортиб чиқарадиган ҳолат эди.",
        "Рақибларимизни кўриб бизнинг асло тепа сочимиз тик бўлмади.",
    ]

    for gap in sentences:
        print("Gap:", gap)
        mwes = tokenizer.extract_mwe(gap)
        print("Topilgan MWE'lar:", mwes)
        print("-" * 50)


if __name__ == "__main__":
    main()
