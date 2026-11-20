"""Batch MWE extraction example for UzbekMWETokenizer.

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

    batch_results = tokenizer.extract_batch(sentences)
    for sent, mwes in zip(sentences, batch_results):
        print("Gap:", sent)
        print("Topilgan MWE'lar:", mwes)
        print("-" * 50)


if __name__ == "__main__":
    main()
