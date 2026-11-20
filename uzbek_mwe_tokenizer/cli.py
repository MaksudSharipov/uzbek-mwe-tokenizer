"""Command-line interface for UzbekMWETokenizer."""

from __future__ import annotations

import argparse
import json

from .tokenizer import DEFAULT_MODEL_NAME, UzbekMWETokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transformer-based Multi-Word Expression (MWE) extraction for Uzbek text."
    )
    parser.add_argument(
        "text",
        type=str,
        help="Uzbek sentence or text to analyze.",
    )
    parser.add_argument(
        "--mode",
        choices=["lot", "cyr"],
        default="lot",
        help="Script mode: 'lot' (Latin) or 'cyr' (Cyrillic). Default: lot.",
    )
    parser.add_argument(
        "--format",
        choices=["mwe", "slash", "conll", "json"],
        default="mwe",
        help="Output format. Default: mwe.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tokenizer = UzbekMWETokenizer(
        mode=args.mode,
        model_name=args.model,
        device=args.device,
        quiet=True,
    )
    output = tokenizer(args.text, output_format=args.format)

    if args.format in ("mwe", "json"):
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(output)


if __name__ == "__main__":
    main()
