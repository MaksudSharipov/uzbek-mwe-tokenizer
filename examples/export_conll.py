"""Export word-level BIO MWE predictions in CoNLL and slash formats.

Run:
    python examples/export_conll.py
"""

from uzbek_mwe_tokenizer import UzbekMWETokenizer


def main() -> None:
    tokenizer = UzbekMWETokenizer(mode="lot", quiet=True)
    sentence = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."

    print("=== SLASH FORMAT ===")
    print(tokenizer(sentence, output_format="slash"))

    print("\n=== CONLL BIO FORMAT ===")
    print(tokenizer(sentence, output_format="conll"))


if __name__ == "__main__":
    main()
