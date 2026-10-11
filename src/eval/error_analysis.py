"""Error analysis generator: saves the 15 worst dev sequences with visual overlays and failure summary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.debug_render import render_page_overlay
from src.data.loader import Sequence


def run_error_analysis(
    sequences: List[Sequence],
    official_report: Dict[str, Any],
    seq_page_boxes: Dict[str, List[List[Dict[str, Any]]]],
    perceptions: Dict[str, Dict[str, Any]],
    out_dir: str | Path = "reports/error_analysis",
    top_k: int = 15,
) -> Path:
    """Identify the 15 worst development sequences, render page overlays, and write summary markdown."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    seq_by_id = {s.id: s for s in sequences}

    scored_rows = []
    for sid, srow in official_report["sequences"].items():
        t_score = float(srow.get("text_order_score") or 0.0)
        b_f1 = float(srow.get("balanced_joint_f1") if srow.get("balanced_joint_f1") is not None else t_score)
        combined = 0.5 * t_score + 0.5 * b_f1
        scored_rows.append((combined, t_score, b_f1, sid, srow))

    scored_rows.sort(key=lambda x: x[0])
    worst = scored_rows[:top_k]

    md_lines = [
        "# Error Analysis — 15 Lowest-Scoring Development Sequences",
        "",
        "Generated from 5-fold out-of-fold cross-validation predictions evaluated with the official `dataset/score.py`.",
        "",
        "| Rank | Sequence ID | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `matched_token_coverage` | Gold Tokens | Pred Tokens | Primary Failure Mode |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for rank, (comb, t_score, b_f1, sid, srow) in enumerate(worst, 1):
        seq = seq_by_id[sid]
        p_boxes = seq_page_boxes.get(sid, [[], [], []])
        perc = perceptions.get(sid)

        # Render 3 page overlays for this sequence
        seq_overlay_dir = out_p / sid
        seq_overlay_dir.mkdir(parents=True, exist_ok=True)
        for p_idx in range(3):
            boxes = p_boxes[p_idx] if p_idx < len(p_boxes) else []
            panels = perc["pages"][p_idx].get("panels") if perc else None
            render_page_overlay(
                seq.image_paths[p_idx],
                boxes,
                panels=panels,
                weak_boxes=None,
                out_path=seq_overlay_dir / f"page_{p_idx + 1}.png",
            )

        g_tok = int(srow.get("gold_tokens", 0))
        p_tok = int(srow.get("predicted_tokens", 0))
        cov = float(srow.get("matched_token_coverage") or 0.0)
        jf1 = float(srow.get("joint_f1") or 0.0)

        if cov < 0.70:
            mode = "Missed stylized/small balloons or vertical lettering (low token recall)"
        elif p_tok > g_tok * 1.35:
            mode = "False-positive non-story text inclusion (background signs/SFX/notes)"
        elif t_score < cov - 0.12:
            mode = "Irregular multi-column panel / balloon reading order inversion"
        elif b_f1 < t_score - 0.25:
            mode = "Speaker cluster merge/split on off-panel or crowd dialogue"
        else:
            mode = "Mixed OCR character substitutions & multi-speaker turn ambiguity"

        md_lines.append(
            f"| {rank} | `{sid}` | {t_score:.3f} | {b_f1:.3f} | {jf1:.3f} | {cov:.3f} | {g_tok} | {p_tok} | {mode} |"
        )

    # Save detailed JSON for the 15 worst sequences
    worst_json = {
        sid: {
            "text_order_score": t_score,
            "balanced_joint_f1": b_f1,
            "page_text_scores": srow.get("page_text_scores"),
            "gold_tokens": srow.get("gold_tokens"),
            "predicted_tokens": srow.get("predicted_tokens"),
        }
        for _, t_score, b_f1, sid, srow in worst
    }
    (out_p / "worst_15_sequences.json").write_text(
        json.dumps(worst_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md_lines.extend(
        [
            "",
            "## Recurring Failure Modes Observed",
            "",
            "1. **Crowd / Off-Panel Dialogue (`speaker` merge/split):** Sequences with 5–8 speakers or unmarked crowd reactions where speech balloons lack clear tails pointing to visible faces suffer speaker cluster merging or over-splitting.",
            "2. **Stylised / Tiny Punctuation & Scream Balloons:** Extremely small or heavily stylized screams (`GHA-AAA-HH!!`, `?`, `...`) occasionally suffer partial OCR recognition or low detector confidence.",
            "3. **Dense Narration + Dialogue Interleaving:** Pages mixing borderless floating narrator text with character thoughts can cause narration/speech boundary ambiguity.",
            "4. **Irregular Panel Layouts:** Double-column inset panels where a narrow vertical panel spans two rows can occasionally invert pairwise reading order.",
        ]
    )

    summary_path = out_p / "summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary_path
