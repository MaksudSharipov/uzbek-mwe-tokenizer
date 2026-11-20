"""Main MWE extraction and BIO sequence tagging implementation for UzbekMWETokenizer."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from UzTransliterator import UzTransliterator

from .label_map import ID2LABEL_DEFAULT
from .text_utils import normalize_apostrophes

DEFAULT_MODEL_NAME = "MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT"


class UzbekMWETokenizer:
    """Transformer-based Multi-Word Expression (MWE) identifier for Uzbek texts.

    Parameters
    ----------
    mode:
        Script mode: ``"lot"`` (Latin) or ``"cyr"`` (Cyrillic). Default: ``"lot"``.
    model_name:
        Hugging Face model identifier or local model directory path.
    device:
        Target PyTorch device (``"auto"``, ``"cpu"``, ``"cuda"``).
    quiet:
        If ``True``, suppresses startup banner output.
    """

    def __init__(
        self,
        mode: str = "lot",
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = "auto",
        quiet: bool = False,
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in ["lot", "cyr"]:
            raise ValueError("Mode must be either 'lot' (Latin) or 'cyr' (Cyrillic).")

        if not quiet:
            print("=" * 70)
            print("[INFO] O'ZBEK TILI MWE ANIQLAGICH: UzBERT MODELIGA XUSH KELIBSIZ!")
            print("Iltimos, ushbu modeldan foydalanganda quyidagi maqolaga havola bering:")
            print(
                "Sharipov, M. S. (2026). Transformer-based intelligent model for identifying "
                "multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. "
                "Bulletin of TUIT: Management and Communication Technologies, 2(14), 103-107. "
                "DOI: 10.61663/262tuitmct14"
            )
            print("Link: https://uzjurnal.uz/2/2026/2/index?issue=14")
            print("=" * 70)

        self.model_name = model_name
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not quiet:
            print(f"[LOAD] Model va Tokenizer yuklanmoqda: {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        config_map = getattr(self.model.config, "id2label", None)
        if isinstance(config_map, dict) and config_map:
            self.id2label = {int(k): str(v) for k, v in config_map.items()}
        else:
            self.id2label = dict(ID2LABEL_DEFAULT)

        self.transliterator = UzTransliterator.UzTransliterator()

        if not quiet:
            print("[SUCCESS] Barcha komponentlar muvaffaqiyatli yuklandi!\n")

    def _prepare_text(self, text: str) -> str:
        """Normalize apostrophes and transliterate Latin to Cyrillic when in 'lot' mode."""
        normalized = normalize_apostrophes(text)
        if self.mode == "lot":
            return self.transliterator.transliterate(normalized, from_="lat", to="cyr")
        return normalized

    def _format_output_text(self, text: str) -> str:
        """Restore output string to Latin script when operating in 'lot' mode."""
        if self.mode == "lot":
            return self.transliterator.transliterate(text, from_="cyr", to="lat")
        return text

    def tag_tokens(self, raw_text: str) -> List[Dict[str, Union[str, float]]]:
        """Assign word-level BIO tags (``O``, ``B-MWE``, ``I-MWE``) and confidence scores."""
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        if not raw_text.strip():
            return []

        original_words = normalize_apostrophes(raw_text).split()
        prepared_text = self._prepare_text(raw_text)
        cyr_words = prepared_text.split()

        inputs = self.tokenizer(cyr_words, is_split_into_words=True, return_tensors="pt")
        word_ids = inputs.word_ids()
        encoded = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits[0].detach().cpu()
        predictions = torch.argmax(logits, dim=-1).tolist()
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        tagged: List[Dict[str, Union[str, float]]] = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                pred_id = int(predictions[idx])
                label = self.id2label.get(pred_id, "O")
                score = float(probabilities[idx][pred_id].item() * 100.0)
                token_str = (
                    original_words[word_idx]
                    if word_idx < len(original_words)
                    else self._format_output_text(cyr_words[word_idx])
                )
                tagged.append(
                    {
                        "token": token_str,
                        "tag": label,
                        "confidence": round(score, 2),
                    }
                )
            previous_word_idx = word_idx

        return tagged

    def extract_mwe(self, raw_text: str) -> List[Dict[str, Union[str, float]]]:
        """Extract multi-word expressions (length >= 2 words) with average confidence."""
        if not raw_text.strip():
            return []

        text = self._prepare_text(raw_text)
        words = text.split()

        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
        word_ids = inputs.word_ids()
        encoded = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits.detach().cpu()
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        probabilities = torch.nn.functional.softmax(logits, dim=2)[0]

        extracted_mwes: List[Dict[str, Union[str, float]]] = []
        current_mwe_words: List[str] = []
        current_mwe_scores: List[float] = []
        previous_word_idx = None

        def save_mwe() -> None:
            if len(current_mwe_words) >= 2:
                avg_score = sum(current_mwe_scores) / len(current_mwe_scores)
                mwe_str = " ".join(current_mwe_words)
                mwe_str_formatted = self._format_output_text(mwe_str)
                extracted_mwes.append(
                    {
                        "mwe": mwe_str_formatted,
                        "confidence": avg_score,
                    }
                )

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue

            if word_idx != previous_word_idx:
                label = self.id2label[predictions[idx]]
                score = probabilities[idx][predictions[idx]].item() * 100
                full_word = words[word_idx]

                if label == "B-MWE":
                    save_mwe()
                    current_mwe_words = [full_word]
                    current_mwe_scores = [score]
                elif label == "I-MWE":
                    if current_mwe_words:
                        current_mwe_words.append(full_word)
                        current_mwe_scores.append(score)
                    else:
                        current_mwe_words = [full_word]
                        current_mwe_scores = [score]
                else:
                    save_mwe()
                    current_mwe_words = []
                    current_mwe_scores = []

            previous_word_idx = word_idx

        save_mwe()
        return extracted_mwes

    def extract_batch(
        self, sentences: Sequence[str]
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Extract MWEs from a batch of sentences."""
        return [self.extract_mwe(sentence) for sentence in sentences]

    def to_slash(self, tagged_tokens: List[Dict[str, Union[str, float]]]) -> str:
        """Format word-level BIO tags as ``token/TAG`` string."""
        return " ".join(f"{item['token']}/{item['tag']}" for item in tagged_tokens)

    def to_conll(self, tagged_tokens: List[Dict[str, Union[str, float]]]) -> str:
        """Format word-level BIO tags in two-column CoNLL format."""
        return "\n".join(f"{item['token']}\t{item['tag']}" for item in tagged_tokens)

    def to_json(
        self, items: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Union[str, float]]]:
        """Return items as JSON-serializable list of dictionaries."""
        return items

    def __call__(self, text: str, output_format: str = "mwe"):
        """Unified callable interface supporting ``mwe``, ``slash``, ``conll``, and ``json``."""
        if output_format == "mwe":
            return self.extract_mwe(text)
        tagged = self.tag_tokens(text)
        if output_format == "slash":
            return self.to_slash(tagged)
        if output_format == "conll":
            return self.to_conll(tagged)
        if output_format == "json":
            return self.to_json(tagged)
        raise ValueError("output_format must be one of: mwe, slash, conll, json")


# Convenient alias
UzbekMWEExtractor = UzbekMWETokenizer
