import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from UzTransliterator import UzTransliterator

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9А-Яа-яЁёЎўҚқҒғҲҳ']+(?:-[A-Za-z0-9А-Яа-яЁёЎўҚқҒғҲҳ']+)*|[^\w\s]"
)
PUNCT_PATTERN = re.compile(r"^[^\w\s]+$")
APOSTROPHE_VARIANTS = ("’", "‘", "`", "´", "ʼ", "ʻ")


class UzbekMWETokenizer:
    def __init__(self, mode="lot", model_name="MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT"):
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

        self.mode = mode.lower()
        if self.mode not in ["lot", "cyr"]:
            raise ValueError("Mode must be either 'lot' (Latin) or 'cyr' (Cyrillic).")

        self.model_name = model_name
        print(f"[LOAD] Model va Tokenizer yuklanmoqda: {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

        if hasattr(UzTransliterator, "UzTransliterator") and callable(
            getattr(UzTransliterator, "UzTransliterator")
        ):
            self.transliterator = UzTransliterator.UzTransliterator()
        else:
            self.transliterator = UzTransliterator()

        print("[SUCCESS] Barcha komponentlar muvaffaqiyatli yuklandi!\n")

    @staticmethod
    def _normalize_apostrophes(text):
        for ch in APOSTROPHE_VARIANTS:
            text = text.replace(ch, "'")
        return text

    @staticmethod
    def _tokenize_lexical_and_punct(text):
        return TOKEN_PATTERN.findall(text)

    @staticmethod
    def _is_punct(token):
        return bool(PUNCT_PATTERN.match(token))

    def _prepare_text(self, text):
        text = self._normalize_apostrophes(text)
        if self.mode == "lot":
            text = self.transliterator.transliterate(text, from_="lat", to="cyr")
        return text

    def _format_output_text(self, text):
        if self.mode == "lot":
            text = self.transliterator.transliterate(text, from_="cyr", to="lat")
            text = self._normalize_apostrophes(text)
        return text

    def extract_mwe(self, raw_text):
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string.")
        if not raw_text.strip():
            return []

        text = self._prepare_text(raw_text)
        words = self._tokenize_lexical_and_punct(text)
        if not words:
            return []

        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = inputs.word_ids()

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        probabilities = torch.nn.functional.softmax(logits, dim=2)[0]

        extracted_mwes = []
        current_mwe_words = []
        current_mwe_scores = []
        previous_word_idx = None

        def save_mwe():
            if len(current_mwe_words) >= 2:
                avg_score = round(sum(current_mwe_scores) / len(current_mwe_scores), 2)
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
                full_word = words[word_idx]
                label = self.id2label[predictions[idx]]
                score = probabilities[idx][predictions[idx]].item() * 100.0

                if self._is_punct(full_word) or label == "O":
                    save_mwe()
                    current_mwe_words = []
                    current_mwe_scores = []
                elif label == "B-MWE":
                    save_mwe()
                    current_mwe_words = [full_word]
                    current_mwe_scores = [score]
                elif label == "I-MWE":
                    current_mwe_words.append(full_word)
                    current_mwe_scores.append(score)

            previous_word_idx = word_idx

        save_mwe()
        return extracted_mwes
