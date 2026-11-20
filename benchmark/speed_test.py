"""Inference-speed benchmark for UzbekMWETokenizer.

Run:
    python benchmark/speed_test.py

The script reports approximate runtime, sentences/second, and tokens/second on the current machine.
"""

from __future__ import annotations

import time
from uzbek_mwe_tokenizer import UzbekMWETokenizer

SENTENCES = [
    "Natijani ko'rib, hamma o'quvchilarning birdaniga tarvuzi qo'ltig'idan tushdi.",
    "Qorong'i xonada uning qattiq kapalagi uchdi, go'yo yerga kirsa, qulog'idan tortib chiqaradigan holat edi.",
    "Raqiblarimizni ko'rib bizning aslo tepa sochimiz tik bo'lmadi.",
    "U bozordan yangi kitob sotib oldi va xursand bo'ldi.",
    "Bu masala ustida ko'p bosh qotirdik.",
] * 20


def main() -> None:
    tokenizer = UzbekMWETokenizer(mode="lot", quiet=True)

    start = time.perf_counter()
    results = tokenizer.extract_batch(SENTENCES)
    elapsed = time.perf_counter() - start

    token_count = sum(len(s.split()) for s in SENTENCES)
    sent_count = len(SENTENCES)
    mwe_count = sum(len(r) for r in results)
    tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
    sentences_per_second = sent_count / elapsed if elapsed > 0 else 0.0

    print("Inference speed benchmark (UzbekMWETokenizer)")
    print("=============================================")
    print(f"Sentences: {sent_count}")
    print(f"Tokens: {token_count}")
    print(f"Extracted MWEs: {mwe_count}")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Sentences/sec: {sentences_per_second:.2f}")
    print(f"Tokens/sec: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
