"""M10a — Post-processing of ordered, speaker-attributed boxes into clean page transcripts."""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional

from src.data.loader import load_score_module
from src.learn.lora_train import ContextualTokenRefiner

_SCORE = load_score_module()
norm = _SCORE.norm


def sanitize_text(text: str) -> str:
    """Remove control characters and collapse whitespace."""
    cleaned = "".join(
        c for c in text if unicodedata.category(c) != "Cf" and (unicodedata.category(c) != "Cc" or c.isspace())
    )
    return " ".join(cleaned.split()).strip()


def postprocess_sequence_pages(
    page_boxes: List[List[Dict[str, Any]]],
    merge_same_balloon: bool = True,
    token_refiner: Optional[ContextualTokenRefiner] = None,
    default_speaker: str = "char1",
) -> List[List[Dict[str, str]]]:
    """Convert 3 pages of ordered/clustered boxes into validated [{"speaker", "text"}] page lists."""
    output_pages: List[List[Dict[str, str]]] = []

    for page_idx in range(3):
        boxes = page_boxes[page_idx] if page_idx < len(page_boxes) else []
        included = [b for b in boxes if b.get("kind") != "exclude"]
        included.sort(key=lambda b: int(b.get("order_rank", 999)))

        merged_rows: List[Dict[str, Any]] = []
        for b in included:
            raw_txt = str(b.get("text_raw") or b.get("pre_ocr_text") or "")
            txt = sanitize_text(raw_txt)
            if token_refiner is not None and token_refiner.is_fitted:
                txt = sanitize_text(token_refiner.refine_text(txt))
            if not txt or not norm(txt):
                continue

            spk = str(b.get("cluster") or default_speaker).strip()
            if not spk or spk.upper() == "UNKNOWN":
                spk = default_speaker
            if b.get("kind") == "narration":
                spk = "NARRATION"

            bal_id = b.get("balloon_id")
            if (
                merge_same_balloon
                and merged_rows
                and bal_id is not None
                and "_bal" in str(bal_id)
                and merged_rows[-1].get("balloon_id") == bal_id
                and merged_rows[-1]["speaker"] == spk
            ):
                merged_rows[-1]["text"] = sanitize_text(merged_rows[-1]["text"] + " " + txt)
            else:
                merged_rows.append({"speaker": spk, "text": txt, "balloon_id": bal_id})

        page_out = [{"speaker": r["speaker"], "text": r["text"]} for r in merged_rows if norm(r["text"])]
        output_pages.append(page_out)

    while len(output_pages) < 3:
        output_pages.append([])
    return output_pages[:3]
