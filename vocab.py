import spacy
from collections import Counter
import torch
import json
import os

# --- Try loading spaCy tokenizer safely ---
try:
    spacy_eng = spacy.load("en_core_web_sm")
    def spacy_tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
except Exception as e:
    print("⚠️ Warning: spaCy model 'en_core_web_sm' not found. Falling back to .split() tokenizer.")
    def spacy_tokenizer(text):
        return text.lower().split()


class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        """
        freq_threshold: minimum word frequency to include in vocab
        """
        self.freq_threshold = freq_threshold
        # special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text: str):
        return spacy_tokenizer(text)

    def build_vocab(self, sentence_list):
        """
        sentence_list: list of strings (captions)
        """
        frequencies = Counter()
        idx = len(self.itos)  # start after special tokens

        for sentence in sentence_list:
            tokens = self.tokenizer_eng(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"✅ Vocab built with {len(self.itos)} tokens (freq_threshold={self.freq_threshold})")

    def numericalize(self, text):
        """
        Convert string caption into list of token indices
        """
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

    # --- Extra utilities ---
    def save_vocab(self, filepath: str):
        """
        Save vocab mappings to JSON file
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "itos": self.itos,
                "stoi": self.stoi,
                "freq_threshold": self.freq_threshold
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulary saved to {filepath}")

    @classmethod
    def load_vocab(cls, filepath: str):
        """
        Load vocab from JSON file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No vocab file found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(freq_threshold=data.get("freq_threshold", 5))
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        vocab.stoi = {k: int(v) for k, v in data["stoi"].items()}
        print(f"📂 Vocabulary loaded from {filepath} (size={len(vocab)})")
        return vocab
