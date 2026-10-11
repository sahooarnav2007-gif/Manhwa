"""T3 — Learned contextual OCR token refiner and sequence-level adaptation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import joblib

from src.data.loader import load_score_module
from src.eval.proxy_score import fast_distance

_SCORE = load_score_module()
norm = _SCORE.norm
TOKEN = _SCORE.TOKEN


class ContextualTokenRefiner:
    """Learns a domain vocabulary from development gold transcripts and fixes 1-edit-distance OCR typos."""

    def __init__(self, max_edit_ratio: float = 0.20) -> None:
        self.max_edit_ratio = max_edit_ratio
        self.vocab_counts: Counter = Counter()
        self.is_fitted = False

    def fit_from_sequences(self, train_sequences: List[Any]) -> "ContextualTokenRefiner":
        self.vocab_counts.clear()
        for seq in train_sequences:
            if not seq.gold_pages:
                continue
            for page in seq.gold_pages:
                for row in page:
                    for tok in TOKEN.findall(norm(row["text"])):
                        if len(tok) >= 4 and tok.isalpha():
                            self.vocab_counts[tok.upper()] += 1
        self.is_fitted = True
        return self

    def refine_text(self, text: str) -> str:
        if not self.is_fitted or not text:
            return text
        words = text.split(" ")
        new_words = []
        for w in words:
            # Strip leading/trailing punctuation around token
            prefix = ""
            suffix = ""
            core = w
            while core and not core[0].isalnum():
                prefix += core[0]
                core = core[1:]
            while core and not core[-1].isalnum():
                suffix = core[-1] + suffix
                core = core[:-1]

            core_up = core.upper()
            if len(core_up) >= 5 and core_up.isalpha() and core_up not in self.vocab_counts:
                # Find closest frequent domain word of same length with edit distance == 1
                best_cand = None
                best_freq = 1
                for cand, freq in self.vocab_counts.items():
                    if freq > best_freq and abs(len(cand) - len(core_up)) == 0:
                        if fast_distance(core_up, cand) == 1:
                            best_cand = cand
                            best_freq = freq
                if best_cand is not None:
                    core = best_cand if core.isupper() else best_cand.capitalize()
            new_words.append(prefix + core + suffix)

        candidate = " ".join(new_words)
        if fast_distance(text, candidate) / float(max(len(text), 1)) <= self.max_edit_ratio:
            return candidate
        return text

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vocab_counts": self.vocab_counts, "is_fitted": self.is_fitted}, p)

    @classmethod
    def load(cls, path: str | Path) -> "ContextualTokenRefiner":
        data = joblib.load(path)
        obj = cls()
        obj.vocab_counts = data["vocab_counts"]
        obj.is_fitted = data["is_fitted"]
        return obj
