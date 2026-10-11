"""Fast rapidfuzz-accelerated proxy scorer validated against dataset/score.py."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional

from rapidfuzz.distance import Levenshtein

from src.data.loader import load_score_module

_OFFICIAL = load_score_module()
norm = _OFFICIAL.norm
TOKEN = _OFFICIAL.TOKEN
best_mapping = _OFFICIAL.best_mapping


def fast_distance(a: str, b: str) -> int:
    """C++-accelerated Levenshtein edit distance via rapidfuzz."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    return int(Levenshtein.distance(a, b))


@lru_cache(maxsize=131072)
def fast_token_match(a: str, b: str) -> float:
    """Fast token similarity matching official score.py rules."""
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 3:
        return 0.0
    score = 1.0 - fast_distance(a, b) / max(len(a), len(b))
    return score if score >= 0.6 else 0.0


def tokens(rows: List[Dict[str, Optional[str]]]):
    """Tokenize a page's rows into (token, speaker) tuples."""
    return [(token, row["speaker"]) for row in rows for token in TOKEN.findall(norm(row["text"]))]


def fast_align_tokens(gold, predicted):
    """Maximum-weight monotonic token alignment using fast_token_match."""
    n, m = len(gold), len(predicted)
    if n == 0 or m == 0:
        return []
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        g_tok = gold[i - 1][0]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        back_i = back[i]
        for j in range(1, m + 1):
            score = fast_token_match(g_tok, predicted[j - 1][0])
            up = dp_prev[j]
            left = dp_i[j - 1]
            # Match official tie-breaking: max([(up, 1), (left, 2), (diag, 3)], key=lambda x: x[0])
            best_val = up
            best_move = 1
            if left > best_val:
                best_val = left
                best_move = 2
            if score:
                diag = dp_prev[j - 1] + score
                if diag > best_val:
                    best_val = diag
                    best_move = 3
            dp_i[j] = best_val
            back_i[j] = best_move
    pairs = []
    i, j = n, m
    while i and j:
        move = back[i][j]
        if move == 3:
            pairs.append((gold[i - 1], predicted[j - 1], fast_token_match(gold[i - 1][0], predicted[j - 1][0])))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def proxy_score_sequence(gold_pages, predicted_pages) -> Dict[str, Any]:
    """Compute sequence-level metrics using rapidfuzz C++ Levenshtein."""
    pairs, page_scores = [], {}
    active_page_scores = []
    gold_speakers: Counter = Counter()
    predicted_speakers: Counter = Counter()
    gold_count = predicted_count = known_gold = 0
    for page_number, (gold, predicted) in enumerate(zip(gold_pages, predicted_pages), 1):
        gold_text = norm(" ".join(row["text"] for row in gold))
        predicted_text = norm(" ".join(row["text"] for row in predicted))
        page_scores[str(page_number)] = max(
            0.0,
            1.0 - fast_distance(gold_text, predicted_text) / max(len(gold_text), len(predicted_text), 1),
        )
        if gold_text or predicted_text:
            active_page_scores.append(page_scores[str(page_number)])
        gold_tokens = tokens(gold)
        predicted_tokens = tokens(predicted)
        gold_speakers.update(speaker for _, speaker in gold_tokens if speaker is not None)
        predicted_speakers.update(speaker for _, speaker in predicted_tokens)
        gold_count += len(gold_tokens)
        predicted_count += len(predicted_tokens)
        known_gold += sum(speaker is not None for _, speaker in gold_tokens)
        pairs.extend(fast_align_tokens(gold_tokens, predicted_tokens))

    labels = [(gold[1], predicted[1], weight) for gold, predicted, weight in pairs if gold[1] is not None]
    mapping = best_mapping(labels)
    correct = sum(weight for gold, predicted, weight in labels if mapping.get(predicted) == gold)
    aligned = sum(weight for _, _, weight in labels)
    masked_predicted = sum(gold[1] is None for gold, _, _ in pairs)
    speaker_predicted = predicted_count - masked_predicted

    for gold, predicted, _ in pairs:
        if gold[1] is None:
            predicted_speakers[predicted[1]] -= 1
    predicted_speakers = +predicted_speakers

    balanced_pairs = [
        (gold, predicted, 2 * weight / (gold_speakers[gold] + predicted_speakers[predicted]))
        for gold, predicted, weight in labels
    ]
    balanced_mapping = best_mapping(balanced_pairs)
    balanced_sum = sum(
        weight for gold, predicted, weight in balanced_pairs if balanced_mapping.get(predicted) == gold
    )
    extra_speakers = sum(balanced_mapping.get(label) not in gold_speakers for label in predicted_speakers)
    balanced_f1 = min(1.0, balanced_sum / (len(gold_speakers) + extra_speakers)) if known_gold else None

    return {
        "text_order_score": sum(active_page_scores) / len(active_page_scores) if active_page_scores else 1.0,
        "active_text_pages": len(active_page_scores),
        "balanced_joint_f1": balanced_f1,
        "balanced_speaker_mapping": balanced_mapping,
        "speaker_accuracy_on_matched": correct / aligned if aligned else (0.0 if known_gold else None),
        "joint_f1": 2 * correct / (known_gold + speaker_predicted) if known_gold else None,
        "matched_token_coverage": (
            sum(weight for _, _, weight in pairs) / gold_count
            if gold_count
            else (1.0 if not predicted_count else 0.0)
        ),
        "gold_tokens": gold_count,
        "predicted_tokens": predicted_count,
        "page_text_scores": page_scores,
    }


def proxy_evaluate(references: Dict[str, Any], predictions: Dict[str, Any], pages: int = 3) -> Dict[str, Any]:
    """Fast proxy evaluation over a dictionary of sequence references and predictions."""
    if not references:
        raise ValueError("Reference file has no sequences")
    extra = sorted(predictions.keys() - references.keys())
    if extra:
        raise ValueError(f"Unknown extra sequence_id(s): {', '.join(extra)}")
    missing = sorted(references.keys() - predictions.keys())
    scores = {}
    for sequence_id, gold in references.items():
        predicted = predictions.get(sequence_id, [[] for _ in range(pages)])
        row = proxy_score_sequence(gold, predicted)
        row["missing_sequence"] = sequence_id in missing
        if row["missing_sequence"]:
            row["text_order_score"] = row["matched_token_coverage"] = 0.0
            row["page_text_scores"] = {str(page): 0.0 for page in range(1, pages + 1)}
        scores[sequence_id] = row

    macro = {}
    for key in (
        "text_order_score",
        "balanced_joint_f1",
        "speaker_accuracy_on_matched",
        "joint_f1",
        "matched_token_coverage",
    ):
        eligible = list(scores.values())
        if key in ("text_order_score", "matched_token_coverage"):
            active = [row for row in eligible if row["active_text_pages"] or row["missing_sequence"]]
            eligible = active or eligible
        values = [row[key] for row in eligible if row[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
    return {
        "version": "2.1-proxy",
        "pages_per_sequence": pages,
        "macro": macro,
        "missing_sequences": missing,
        "sequences": scores,
    }
