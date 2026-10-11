"""M8 Signal 1 — Tail-geometry and panel-proximity speaker attribution."""
from __future__ import annotations

import math
from typing import Any, Dict, List


class TailGeometryAttributor:
    """Attributes each speech balloon to candidate characters using tail tip, tail vector, and panel proximity."""

    def run(
        self,
        boxes: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
        page_w: int,
        page_h: int,
    ) -> None:
        diag = math.hypot(page_w, page_h)
        for box in boxes:
            if box.get("kind") in ("exclude", "narration"):
                box["tail_candidates"] = []
                continue

            bx0, by0, bx1, by1 = box["bbox"]
            bcx = 0.5 * (bx0 + bx1)
            bcy = 0.5 * (by0 + by1)
            tail_tip = box.get("tail_tip", [int(bcx), int(by1)])
            tail_vec = box.get("tail_vec", [0.0, 1.0])
            tx, ty = float(tail_tip[0]), float(tail_tip[1])
            vx, vy = float(tail_vec[0]), float(tail_vec[1])
            v_norm = math.hypot(vx, vy) or 1.0
            vx, vy = vx / v_norm, vy / v_norm

            b_panel = box.get("panel_id")
            candidates = []
            for ch in characters:
                cx0, cy0, cx1, cy1 = ch["bbox"]
                # Head anchor is upper-center of character box
                hcx = 0.5 * (cx0 + cx1)
                hcy = cy0 + 0.30 * (cy1 - cy0)

                dx = hcx - tx
                dy = hcy - ty
                dist = math.hypot(dx, dy)
                norm_dist = dist / max(1.0, diag)

                # Cosine alignment between balloon tail vector and vector from tail tip to character head
                if dist > 4.0:
                    cos_align = (dx * vx + dy * vy) / dist
                else:
                    cos_align = 1.0

                same_panel = 1.0 if (b_panel and ch.get("panel_id") == b_panel) else 0.0
                # Combine proximity, same-panel bonus, and tail ray alignment
                score = (
                    0.50 * same_panel
                    + 0.35 * math.exp(-3.5 * norm_dist)
                    + 0.15 * max(0.0, cos_align) * min(1.5, float(box.get("tail_prominence", 1.0)))
                )
                candidates.append(
                    {
                        "char_id": ch["char_id"],
                        "score": round(float(min(0.99, max(0.01, score))), 4),
                        "source": "tail",
                        "same_panel": bool(same_panel > 0.5),
                        "norm_dist": round(float(norm_dist), 4),
                    }
                )

            candidates.sort(key=lambda c: c["score"], reverse=True)
            box["tail_candidates"] = candidates[:3]
