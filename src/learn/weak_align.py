"""Section 7.1 — Weak-label alignment harness turning page-level transcripts into box-level supervision."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.loader import Sequence, load_score_module

_SCORE = load_score_module()
norm = _SCORE.norm
TOKEN = _SCORE.TOKEN
token_match = _SCORE.token_match


def _tokens_list(text: str) -> List[str]:
    return TOKEN.findall(norm(text))


def _box_line_similarity(box_tokens: List[str], line_tokens: List[str], box_norm: str, line_norm: str) -> float:
    """Compute fuzzy token overlap using official score.py's norm, TOKEN, and token_match."""
    if not box_norm or not line_norm:
        return 0.0
    if box_norm == line_norm:
        return 1.0
    if not box_tokens or not line_tokens:
        return 0.0

    # Short punctuation-only or 1-token lines where token_match (<3 chars) requires exact match
    used_line = [False] * len(line_tokens)
    matched_weight = 0.0
    for bt in box_tokens:
        best_j = -1
        best_s = 0.0
        for j, lt in enumerate(line_tokens):
            if used_line[j]:
                continue
            s = 1.0 if bt == lt else token_match(bt, lt)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j >= 0 and best_s >= 0.6:
            used_line[best_j] = True
            matched_weight += best_s

    # Precision-recall harmonic / coverage blend so partial boxes (split balloon) and merged boxes both align
    box_cov = matched_weight / float(max(1, len(box_tokens)))
    line_cov = matched_weight / float(max(1, len(line_tokens)))
    f1 = (2.0 * box_cov * line_cov / (box_cov + line_cov)) if (box_cov + line_cov) > 0 else 0.0
    # If a box is a clean subset of a gold line (split balloon) or vice versa, max(box_cov, line_cov) is high
    return float(max(f1, 0.85 * box_cov + 0.15 * line_cov, 0.85 * line_cov + 0.15 * box_cov))


def align_sequence_boxes(
    seq: Sequence,
    page_boxes: List[List[Dict[str, Any]]],
    sim_threshold: float = 0.60,
) -> Dict[str, Any]:
    """Align detected OCR boxes on all 3 pages of a development sequence to its gold lines."""
    if not seq.gold_pages:
        return {"boxes": {}, "unmatched_gold_lines": [], "detector_token_recall": 0.0}

    box_labels: Dict[str, Dict[str, Any]] = {}
    unmatched_gold: List[Dict[str, Any]] = []
    total_gold_tokens = 0
    covered_gold_tokens = 0.0

    for page_idx in range(3):
        page_num = page_idx + 1
        boxes = page_boxes[page_idx] if page_idx < len(page_boxes) else []
        gold_lines = seq.gold_pages[page_idx] if page_idx < len(seq.gold_pages) else []

        g_norms = [norm(g["text"]) for g in gold_lines]
        g_toks = [_tokens_list(g["text"]) for g in gold_lines]
        g_covered = [False] * len(gold_lines)

        # Compute similarity matrix S[b_idx, g_idx]
        for b in boxes:
            bid = b["id"]
            b_text = b.get("text_raw") or b.get("pre_ocr_text") or ""
            b_norm = norm(b_text)
            b_tok = _tokens_list(b_text)

            best_g = -1
            best_sim = 0.0
            # Also check 2 consecutive gold lines in case one balloon spans 2 consecutive gold lines
            for g_i in range(len(gold_lines)):
                s1 = _box_line_similarity(b_tok, g_toks[g_i], b_norm, g_norms[g_i])
                if s1 > best_sim:
                    best_sim = s1
                    best_g = g_i
                if g_i + 1 < len(gold_lines):
                    comb_tok = g_toks[g_i] + g_toks[g_i + 1]
                    comb_norm = (g_norms[g_i] + " " + g_norms[g_i + 1]).strip()
                    s2 = _box_line_similarity(b_tok, comb_tok, b_norm, comb_norm)
                    if s2 > best_sim + 0.05:
                        best_sim = s2
                        best_g = g_i

            if best_g >= 0 and best_sim >= sim_threshold:
                g_Covered_idx = best_g
                g_covered[g_Covered_idx] = True
                spk = gold_lines[g_Covered_idx]["speaker"]
                box_labels[bid] = {
                    "gold_page": page_num,
                    "gold_line": g_Covered_idx + 1,
                    "coverage": round(float(best_sim), 4),
                    "gold_speaker": spk if spk is not None else "UNKNOWN",
                    "gold_rank": g_Covered_idx + 1,
                    "is_story": 1,
                    "kind": "narration" if spk == "NARRATION" else "speech",
                }
            else:
                box_labels[bid] = {
                    "gold_page": page_num,
                    "gold_line": None,
                    "coverage": round(float(best_sim), 4),
                    "gold_speaker": None,
                    "gold_rank": None,
                    "is_story": 0,
                    "kind": "exclude",
                }

        # Measure token recall on this page by comparing union of all detected box tokens against gold page tokens
        all_det_toks = [t for b in boxes for t in _tokens_list(b.get("text_raw") or b.get("pre_ocr_text") or "")]
        used_det = [False] * len(all_det_toks)
        for g_i, gt_list in enumerate(g_toks):
            total_gold_tokens += len(gt_list)
            line_matched = 0.0
            for gt in gt_list:
                best_dj = -1
                best_ds = 0.0
                for dj, dt in enumerate(all_det_toks):
                    if used_det[dj]:
                        continue
                    s = 1.0 if gt == dt else token_match(gt, dt)
                    if s > best_ds:
                        best_ds = s
                        best_dj = dj
                if best_dj >= 0 and best_ds >= 0.6:
                    used_det[best_dj] = True
                    covered_gold_tokens += best_ds
                    line_matched += best_ds
            if not g_covered[g_i] and (len(gt_list) == 0 or line_matched / float(len(gt_list)) < 0.5):
                unmatched_gold.append(
                    {
                        "page": page_num,
                        "line": g_i + 1,
                        "tokens": len(gt_list),
                        "text": gold_lines[g_i]["text"],
                    }
                )

    recall = covered_gold_tokens / float(max(1, total_gold_tokens)) if total_gold_tokens > 0 else 1.0
    return {
        "sequence_id": seq.id,
        "boxes": box_labels,
        "unmatched_gold_lines": unmatched_gold,
        "detector_token_recall": round(float(recall), 4),
        "total_gold_tokens": total_gold_tokens,
        "covered_gold_tokens": round(float(covered_gold_tokens), 2),
    }


def save_weak_labels(weak_data: Dict[str, Any], weak_labels_dir: str | Path = "artifacts/weak_labels") -> Path:
    """Save sequence weak labels JSON to artifacts/weak_labels/<seq_id>.json."""
    out_dir = Path(weak_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{weak_data['sequence_id']}.json"
    p.write_text(json.dumps(weak_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
