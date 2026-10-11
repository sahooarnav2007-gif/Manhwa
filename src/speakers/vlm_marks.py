"""M8 Signal 2 — Set-of-Marks visual grounding and spatial-orientation speaker attribution."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import cv2
import numpy as np

from src.vision.preprocess import PageImage


def render_set_of_marks_page(
    page_img: PageImage,
    boxes: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
) -> np.ndarray:
    """Render numbered speech balloons (red) and numbered characters (blue) onto a copy of the page."""
    canvas = page_img.rgb.copy()
    for idx, b in enumerate(boxes, 1):
        if b.get("kind") == "exclude":
            continue
        x0, y0, x1, y1 = b["bbox"]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 40, 40), 2)
        cv2.putText(canvas, f"B{idx}", (x0 + 2, max(14, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 40, 40), 2)

    for idx, ch in enumerate(characters, 1):
        x0, y0, x1, y1 = ch["bbox"]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 90, 220), 2)
        cv2.putText(canvas, f"C{idx}", (x0 + 2, max(14, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 90, 220), 2)

    return canvas


class VisualMarksAttributor:
    """Evaluates visual line-of-sight and lateral speaker placement (Set-of-Marks grounding signal)."""

    def run(
        self,
        page_img: PageImage,
        boxes: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
    ) -> None:
        h, w = page_img.height, page_img.width
        diag = math.hypot(w, h)
        gray = page_img.gray

        for box in boxes:
            if box.get("kind") in ("exclude", "narration"):
                box["vlm_candidates"] = []
                continue

            bx0, by0, bx1, by1 = box["bbox"]
            bcx = 0.5 * (bx0 + bx1)
            bcy = 0.5 * (by0 + by1)
            b_panel = box.get("panel_id")

            cands = []
            for ch in characters:
                cx0, cy0, cx1, cy1 = ch["bbox"]
                ccx = 0.5 * (cx0 + cx1)
                ccy = 0.5 * (cy0 + cy1)

                # In manga, speech balloons are typically placed above or immediately beside the speaker's head
                dy = ccy - bcy  # positive when character center is below balloon center
                dx = abs(ccx - bcx)
                dist = math.hypot(dx, dy) / max(1.0, diag)

                above_head_bonus = 0.20 if (-0.08 * h <= dy <= 0.45 * h and dx <= 0.35 * w) else 0.0
                same_panel = 0.45 if (b_panel and ch.get("panel_id") == b_panel) else 0.0

                # Check if a heavy dark panel border separates balloon center and character center
                num_samples = 12
                xs = np.linspace(bcx, ccx, num_samples).astype(np.int32).clip(0, w - 1)
                ys = np.linspace(bcy, ccy, num_samples).astype(np.int32).clip(0, h - 1)
                line_vals = gray[ys, xs]
                # Count transitions across white gutter -> dark border
                obstacle_penalty = 0.15 if (not same_panel and np.min(line_vals) < 40 and np.max(line_vals) > 245) else 0.0

                score = same_panel + above_head_bonus + 0.35 * math.exp(-4.0 * dist) - obstacle_penalty
                cands.append(
                    {
                        "char_id": ch["char_id"],
                        "score": round(float(min(0.99, max(0.01, score))), 4),
                        "source": "vlm_marks",
                    }
                )

            cands.sort(key=lambda c: c["score"], reverse=True)
            box["vlm_candidates"] = cands[:3]
