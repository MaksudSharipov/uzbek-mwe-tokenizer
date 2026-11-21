"""CoNLL BIO evaluation script for UzbekMWETokenizer (PyPI v0.1.0 API).

Run:
    python benchmark/evaluate.py --test-file path/to/test.conll --output benchmark/eval_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from uzbek_mwe_tokenizer import UzbekMWETokenizer


def load_conll(path: Path) -> List[Tuple[List[str], List[str]]]:
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    tags: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            tokens.append(parts[0])
            tags.append(parts[-1])

    if tokens:
        sentences.append((tokens, tags))
    return sentences


def predict_word_tags(tokenizer: UzbekMWETokenizer, words: List[str]) -> List[str]:
    text = tokenizer._prepare_text(" ".join(words))
    prep_words = tokenizer._tokenize_lexical_and_punct(text)
    inputs = tokenizer.tokenizer(
        prep_words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_ids = inputs.word_ids()

    with torch.no_grad():
        outputs = tokenizer.model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    pred_tags: List[str] = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            pred_tags.append(tokenizer.id2label[predictions[idx]])
        prev_word_idx = word_idx
    return pred_tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UzbekMWETokenizer on BIO CoNLL data.")
    parser.add_argument("--test-file", type=Path, required=True, help="Path to BIO CoNLL test file.")
    parser.add_argument("--mode", choices=["lot", "cyr"], default="cyr", help="Script mode.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    dataset = load_conll(args.test_file)
    tokenizer = UzbekMWETokenizer(mode=args.mode)

    y_true: List[str] = []
    y_pred: List[str] = []

    for words, gold_tags in dataset:
        pred_tags = predict_word_tags(tokenizer, words)
        min_len = min(len(gold_tags), len(pred_tags))
        y_true.extend(gold_tags[:min_len])
        y_pred.extend(pred_tags[:min_len])

    total = len(y_true)
    correct = sum(1 for g, p in zip(y_true, y_pred) if g == p)
    accuracy = correct / total if total else 0.0

    tp = sum(1 for g, p in zip(y_true, y_pred) if g != "O" and g == p)
    fp = sum(1 for g, p in zip(y_true, y_pred) if p != "O" and g != p)
    fn = sum(1 for g, p in zip(y_true, y_pred) if g != "O" and p != g)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"Evaluated tokens: {total}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tokens", "accuracy", "precision", "recall", "f1_score"])
            writer.writerow([total, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])


if __name__ == "__main__":
    main()
