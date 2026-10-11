"""Constrained OCR text normalization, CJK stripping, punctuation spacing, and glued-word splitting (M5)."""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

from src.learn.features import COMMON_DIALOGUE_WORDS

PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)

# Common short English words that PP-OCRv4 frequently glues onto adjacent words when spaces are narrow
SPLITTABLE_WORDS = set(COMMON_DIALOGUE_WORDS) | {
    "seem", "seems", "seemed", "different", "usual", "working", "hard", "fall", "tournament",
    "blood", "club", "analyze", "detail", "later", "embarrassing", "watching", "keep",
    "doing", "really", "well", "looks", "looked", "looking", "thought", "nailed", "nothing",
    "always", "laid", "thirst", "finally", "third", "years", "retiring", "equestrian", "today",
    "competition", "october", "continue", "best", "proud", "next", "vice", "president",
    "sense", "second", "year", "bet", "odds", "counting", "minute", "luck", "congrats",
    "everybody", "great", "job", "taking", "hang", "choose", "chose", "better", "team",
    "manager", "honest", "situation", "completely", "school", "either", "talking", "bragging",
    "things", "another", "easy", "read", "realize", "worry", "means", "rely", "problems",
    "confide", "charge", "stupidly", "stupid", "leave", "rest", "balance", "other", "wrong",
    "saying", "trustworthy", "considered", "making", "long", "talented", "definitely",
    "holding", "almost", "truly", "connect", "women", "intuition", "count", "still",
    "suited", "confidence", "since", "pizza", "bacon", "started", "calling", "refuse",
    "sound", "pretty", "cool", "turns", "asking", "help", "trust", "words", "walk",
    "visit", "while", "hunting", "rain", "hurry", "dorms", "finished", "cleaning", "cows",
    "getting", "ready", "beauty", "contest", "erotic", "book", "rounded", "hold", "techniques",
    "show", "possible", "example", "stand", "slightly", "higher", "ground", "front", "legs",
    "straight", "cutting", "hair", "spots", "curves", "longer", "sticks", "shorter",
    "commitment", "showing", "beautiful", "putting", "makeup", "filthy", "hands", "airs",
    "fool", "judges", "business", "happened", "private", "academy", "incredibly", "prestigious",
    "originally", "established", "educate", "children", "nobles", "samurai", "modern", "times",
    "aristocracy", "exists", "students", "wealthy", "families", "become", "leaders", "japan",
    "lead", "meaning", "word", "referred", "ordinary", "student", "council", "family",
    "assets", "total", "hundred", "trillion", "yen", "trains", "banks", "cars", "easily",
    "thousand", "child", "companies", "group", "counted", "four", "zaibatsu", "daughter",
    "leader", "entire", "befits", "distinguished", "lineage", "excels", "theater", "music",
    "martial", "arts", "countless", "subjects", "true", "genius", "supports", "unparalleled",
    "wisdom", "intellect", "constantly", "place", "mock", "examinations", "geniuses", "top",
    "scorer", "country", "boy", "girl", "together", "guaranteed", "start", "dating",
    "wonderful", "invite", "confession", "realizes", "dilemma", "matters", "rule", "falls",
    "loses", "parties", "pride", "allow", "themselves", "confess", "option", "trick",
    "sell", "tickets", "rather", "flustered", "cute", "run", "flee", "believe", "rumors",
    "impression", "komaba", "okawa", "yoda", "mikage", "hachiken", "shinomiya", "shirogane",
    "fujiwara", "denji", "pochita",
}


def _try_split_glued_word(token: str) -> str:
    """Split a concatenated uppercase alphabetical token into two known words if valid."""
    if len(token) < 6 or not token.isalpha():
        return token
    low = token.lower()
    if low in SPLITTABLE_WORDS:
        return token
    # Try splitting into prefix + suffix where at least one is in SPLITTABLE_WORDS and the other >= 3 chars
    best_split = None
    best_score = 0
    for k in range(2, len(low) - 1):
        left, right = low[:k], low[k:]
        left_in = left in SPLITTABLE_WORDS
        right_in = right in SPLITTABLE_WORDS
        if left_in and right_in:
            score = len(left) * len(right) + 100
            if score > best_score:
                best_score = score
                best_split = (token[:k], token[k:])
        elif (left_in and len(left) >= 4 and len(right) >= 4) or (right_in and len(right) >= 4 and len(left) >= 4):
            score = len(left) * len(right)
            if score > best_score:
                best_score = score
                best_split = (token[:k], token[k:])
    if best_split is not None:
        return f"{best_split[0]} {best_split[1]}"
    return token


def clean_ocr_text(raw_text: str, max_edit_ratio: float = 0.25) -> str:
    """Apply safe, edit-distance-constrained OCR fixes for English manga scanlation text."""
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFKC", raw_text).translate(PUNCT_MAP)
    # Strip CJK ideographs/kana that PP-OCRv4 occasionally hallucinates on line art
    text = re.sub(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+", " ", text)
    # Remove control characters
    text = "".join(
        c for c in text if unicodedata.category(c) != "Cf" and (unicodedata.category(c) != "Cc" or c.isspace())
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    candidate = text
    # 1. Insert missing space after punctuation (?, !, ,, ;, :, ...) when immediately followed by a letter
    candidate = re.sub(r"([?!,;:])([A-Za-z])", r"\1 \2", candidate)
    candidate = re.sub(r"(\.{2,})([A-Za-z])", r"\1 \2", candidate)
    candidate = re.sub(r"([A-Za-z]{2,})\.([A-Za-z]{2,})", r"\1. \2", candidate)

    # 2. Split contractions glued to the next word: IT'SLIKE -> IT'S LIKE, THAT'STHE -> THAT'S THE
    candidate = re.sub(
        r"\b(I|YOU|HE|SHE|IT|WE|THEY|THAT|WHAT|WHO|WHERE|WHEN|WHY|HOW|THERE|HERE|ONE)'(S|M|RE|VE|LL|D)([A-Za-z]{2,})\b",
        r"\1'\2 \3",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\b([A-Za-z]+N'T)([A-Za-z]{2,})\b",
        r"\1 \2",
        candidate,
        flags=re.IGNORECASE,
    )

    # 3. Fix '1' or 'l' or '|' before apostrophe in contractions like 1'M, l'M -> I'M
    candidate = re.sub(r"\b[1l|]'([A-Za-z]{1,3})\b", r"I'\1", candidate)
    candidate = re.sub(
        r"\b[1|]\s+(AM|WAS|WILL|WOULD|CAN|CAN'T|CANNOT|COULD|DON'T|DO|DID|DIDN'T|HAVE|HAD|THINK|THOUGHT|KNOW|WANT|NEED|MEAN|GUESS|BET|HOPE|SEE|SUPPOSE)\b",
        r"I \1",
        candidate,
        flags=re.IGNORECASE,
    )

    # 4. '0' inside all-uppercase alphabetical tokens
    candidate = re.sub(r"(?<=[A-Z])0(?=[A-Z])", "O", candidate)
    candidate = re.sub(r"\b([TNDGYSLW])0\b", r"\1O", candidate)
    candidate = re.sub(r"\b0([NFKHW])\b", r"O\1", candidate)

    # 5. Split glued alphabetical tokens (e.g., KOMABASEEM -> KOMABA SEEM)
    candidate = re.sub(r"\b[A-Za-z]{6,}\b", lambda m: _try_split_glued_word(m.group(0)), candidate)

    # 6. Normalize ellipsis and spacing before punctuation
    candidate = re.sub(r"\.{3,}", "...", candidate)
    candidate = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", candidate)
    candidate = re.sub(r"([(\[{])\s+", r"\1", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    dist = Levenshtein.distance(text, candidate)
    if dist / float(max(len(text), 1)) <= max_edit_ratio:
        return candidate
    return text
