import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from UzTransliterator import UzTransliterator

class UzbekMWETokenizer:
    def __init__(self, mode="lot"):
        print("=" * 70)
        print("[INFO] O'ZBEK TILI MWE ANIQLAGICH: UzBERT MODELIGA XUSH KELIBSIZ!")
        print("Iltimos, ushbu modeldan foydalanganda quyidagi maqolaga havola bering:")
        print("Sharipov, M. S. (2026). Transformer-based intelligent model for identifying multi-word lexical units and reducing syntactic ambiguities in Uzbek language texts. Bulletin of TUIT: Management and Communication Technologies, 2(14), 103-107. DOI: 10.61663/262tuitmct14")
        print("Link: https://uzjurnal.uz/2/2026/2/index?issue=14")
        print("=" * 70)

        self.mode = mode.lower()
        if self.mode not in ["lot", "cyr"]:
            raise ValueError("Mode must be either 'lot' (Latin) or 'cyr' (Cyrillic).")
            
        self.model_name = "MaksudSharipov/Uzbek-MWE-Tokenizer-uzBERT"
        print(f"[LOAD] Model va Tokenizer yuklanmoqda: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.id2label = self.model.config.id2label
        self.transliterator = UzTransliterator.UzTransliterator()
        
        print("[SUCCESS] Barcha komponentlar muvaffaqiyatli yuklandi!\n")

    def _prepare_text(self, text):
        # Tutuq belgilarini bir xil standartga keltirish
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("ʻ", "'")
        
        if self.mode == "lot":
            # Agar rejim lotincha bo'lsa, uni kirillchaga o'giramiz
            text = self.transliterator.transliterate(text, from_="lat", to="cyr")
        return text

    def _format_output_text(self, text):
        if self.mode == "lot":
            return self.transliterator.transliterate(text, from_="cyr", to="lat")
        return text

    def extract_mwe(self, raw_text):
        if not raw_text.strip():
            return []

        text = self._prepare_text(raw_text)
        words = text.split()
        
        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
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
                avg_score = sum(current_mwe_scores) / len(current_mwe_scores)
                mwe_str = " ".join(current_mwe_words)
                mwe_str_formatted = self._format_output_text(mwe_str)
                extracted_mwes.append({
                    "mwe": mwe_str_formatted,
                    "confidence": avg_score
                })

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
