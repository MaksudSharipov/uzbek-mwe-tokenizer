"""Basic usage example for UzbekMWETokenizer.

Run:
    python examples/basic_usage.py
"""

from uzbek_mwe_tokenizer import UzbekMWETokenizer


def main() -> None:
    tokenizer = UzbekMWETokenizer(mode="lot")

    text = "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi."
    mwes = tokenizer.extract_mwe(text)

    print("Input:", text)
    print("Extracted MWEs:", mwes)


if __name__ == "__main__":
    main()
