"""M4a — Character detection on manga panels (excluding speech balloons)."""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from src.vision.preprocess import PageImage


class CharacterDetector:
    """Detects character head/upper-body regions inside manga panels using edge/tone density and cascades."""

    def __init__(self, det_conf_threshold: float = 0.30) -> None:
        self.det_conf_threshold = det_conf_threshold
        self._face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            c = cv2.CascadeClassifier(cascade_path)
            if not c.empty():
                self._face_cascade = c
        except Exception:
            self._face_cascade = None

    def run(
        self,
        page_img: PageImage,
        panels: List[Dict[str, Any]],
        text_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        gray = page_img.gray
        h, w = gray.shape[:2]

        # Mask out text bubbles so speech balloons are never detected as characters
        bubble_mask = np.zeros((h, w), dtype=np.uint8)
        for tb in text_boxes:
            bx0, by0, bx1, by1 = tb.get("bubble_bbox", tb["bbox"])
            bubble_mask[max(0, by0) : min(h, by1), max(0, bx0) : min(w, bx1)] = 255

        edges = cv2.Canny(gray, 60, 160)
        edges[bubble_mask > 0] = 0

        characters: List[Dict[str, Any]] = []
        char_idx = 0

        panel_list = panels if panels else [{"id": f"p{page_img.page_num}_panel1", "bbox": [0, 0, w, h]}]
        for panel in panel_list:
            px0, py0, px1, py1 = panel["bbox"]
            pw = px1 - px0
            ph = py1 - py0
            if pw < 40 or ph < 40:
                continue

            p_gray = gray[py0:py1, px0:px1]
            p_edges = edges[py0:py1, px0:px1]
            p_bmask = bubble_mask[py0:py1, px0:px1]

            panel_chars: List[Dict[str, Any]] = []

            # 1. Try Haar cascade for frontal face candidates if available
            if self._face_cascade is not None:
                faces = self._face_cascade.detectMultiScale(
                    p_gray,
                    scaleFactor=1.15,
                    minNeighbors=2,
                    minSize=(max(24, pw // 10), max(24, ph // 10)),
                )
                for fx, fy, fw, fh in faces:
                    if np.mean(p_bmask[fy : fy + fh, fx : fx + fw] > 0) > 0.35:
                        continue
                    # Expand to include hair and upper shoulders
                    cx0 = max(0, int(fx - 0.25 * fw))
                    cy0 = max(0, int(fy - 0.30 * fh))
                    cx1 = min(pw, int(fx + 1.25 * fw))
                    cy1 = min(ph, int(fy + 1.65 * fh))
                    panel_chars.append(
                        {
                            "bbox": [px0 + cx0, py0 + cy0, px0 + cx1, py0 + cy1],
                            "det_conf": 0.78,
                        }
                    )

            # 2. Connected-component figure / head-shoulder regions from non-balloon line art
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated = cv2.dilate(p_edges, kernel, iterations=2)
            dilated[p_bmask > 0] = 0
            num_l, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

            cand_stats = []
            for lbl in range(1, num_l):
                rx, ry, rw, rh, area = stats[lbl]
                if rw < pw * 0.10 or rh < ph * 0.12:
                    continue
                if rw * rh < (pw * ph * 0.025):
                    continue
                edge_density = float(np.mean(p_edges[ry : ry + rh, rx : rx + rw] > 0))
                if edge_density < 0.03:
                    continue
                score = edge_density * min(1.0, (rw * rh) / float(pw * ph * 0.35))
                cand_stats.append((score, rx, ry, rw, rh))

            cand_stats.sort(key=lambda x: x[0], reverse=True)
            for score, rx, ry, rw, rh in cand_stats[:3]:
                # Focus on upper-half of large figure blobs (head + shoulders)
                use_h = min(rh, int(max(rw * 1.35, rh * 0.65)))
                bx0, by0, bx1, by1 = px0 + rx, py0 + ry, px0 + rx + rw, py0 + ry + use_h
                # Skip if strongly overlapping an existing character box in this panel
                overlap = False
                for existing in panel_chars:
                    ex0, ey0, ex1, ey1 = existing["bbox"]
                    ix0, iy0 = max(bx0, ex0), max(by0, ey0)
                    ix1, iy1 = min(bx1, ex1), min(by1, ey1)
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    if inter > 0.35 * min((bx1 - bx0) * (by1 - by0), (ex1 - ex0) * (ey1 - ey0)):
                        overlap = True
                        break
                if not overlap:
                    conf = float(min(0.85, max(self.det_conf_threshold, 0.35 + score * 3.0)))
                    panel_chars.append({"bbox": [int(bx0), int(by0), int(bx1), int(by1)], "det_conf": conf})

            for pc in panel_chars[:4]:
                char_idx += 1
                characters.append(
                    {
                        "char_id": f"p{page_img.page_num}_c{char_idx}",
                        "page": page_img.page_num,
                        "panel_id": panel["id"],
                        "bbox": pc["bbox"],
                        "det_conf": pc["det_conf"],
                    }
                )

        return characters
