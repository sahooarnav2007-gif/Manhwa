"""M3 — Panel segmentation via recursive XY-cut on whitespace gutters and frame contours."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from src.vision.preprocess import PageImage


class PanelSegmenter:
    """Segments a manga page into panels using recursive XY-cut and frame contour fallback."""

    def __init__(
        self,
        gutter_white_thresh: int = 240,
        min_gutter_ratio: float = 0.68,
        min_panel_area_ratio: float = 0.03,
    ) -> None:
        self.gutter_white_thresh = gutter_white_thresh
        self.min_gutter_ratio = min_gutter_ratio
        self.min_panel_area_ratio = min_panel_area_ratio

    def _xy_cut(
        self,
        ink_mask: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        min_w: int,
        min_h: int,
        depth: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        w = x1 - x0
        h = y1 - y0
        if w < min_w or h < min_h or depth > 6:
            return [(x0, y0, x1, y1)]

        sub = ink_mask[y0:y1, x0:x1]
        # Prefer horizontal cuts first (rows of panels), then vertical cuts (columns within a row)
        row_ink = np.mean(sub > 0, axis=1)
        margin_y = max(12, int(0.08 * h))
        if h > 2 * min_h and len(row_ink) > 2 * margin_y:
            inner_rows = row_ink[margin_y : h - margin_y]
            gutter_rows = np.where(inner_rows < (1.0 - self.min_gutter_ratio) * 0.15)[0]
            if len(gutter_rows) >= 3:
                # Find longest contiguous gutter run
                splits = np.split(gutter_rows, np.where(np.diff(gutter_rows) > 2)[0] + 1)
                best_run = max(splits, key=len)
                if len(best_run) >= 2:
                    cut_y = y0 + margin_y + int(np.mean(best_run))
                    if (cut_y - y0) >= min_h and (y1 - cut_y) >= min_h:
                        top = self._xy_cut(ink_mask, x0, y0, x1, cut_y, min_w, min_h, depth + 1)
                        bot = self._xy_cut(ink_mask, x0, cut_y, x1, y1, min_w, min_h, depth + 1)
                        return top + bot

        col_ink = np.mean(sub > 0, axis=0)
        margin_x = max(12, int(0.08 * w))
        if w > 2 * min_w and len(col_ink) > 2 * margin_x:
            inner_cols = col_ink[margin_x : w - margin_x]
            gutter_cols = np.where(inner_cols < (1.0 - self.min_gutter_ratio) * 0.15)[0]
            if len(gutter_cols) >= 3:
                splits = np.split(gutter_cols, np.where(np.diff(gutter_cols) > 2)[0] + 1)
                best_run = max(splits, key=len)
                if len(best_run) >= 2:
                    cut_x = x0 + margin_x + int(np.mean(best_run))
                    if (cut_x - x0) >= min_w and (x1 - cut_x) >= min_w:
                        # Manga right-to-left: put right panel before left panel!
                        right = self._xy_cut(ink_mask, cut_x, y0, x1, y1, min_w, min_h, depth + 1)
                        left = self._xy_cut(ink_mask, x0, y0, cut_x, y1, min_w, min_h, depth + 1)
                        return right + left

        return [(x0, y0, x1, y1)]

    def run(self, page_img: PageImage) -> List[Dict[str, Any]]:
        h, w = page_img.height, page_img.width
        ink_mask = (page_img.gray < self.gutter_white_thresh).astype(np.uint8) * 255
        # Close small gaps in panel borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

        min_w = int(w * 0.18)
        min_h = int(h * 0.12)
        raw_boxes = self._xy_cut(closed, 0, 0, w, h, min_w, min_h)

        # If XY-cut produced only 1 full-page box, try contour-based panel detection
        if len(raw_boxes) <= 1:
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cand = []
            min_area = h * w * self.min_panel_area_ratio
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw * ch >= min_area and cw >= min_w * 0.8 and ch >= min_h * 0.8:
                    cand.append((x, y, x + cw, y + ch))
            if len(cand) >= 2:
                raw_boxes = cand

        # Order panels top-to-bottom by row band, right-to-left within row
        def _panel_sort_key(b: Tuple[int, int, int, int]) -> Tuple[int, float]:
            px0, py0, px1, py1 = b
            cy = 0.5 * (py0 + py1)
            cx = 0.5 * (px0 + px1)
            row_band = int(cy // max(1, int(h * 0.16)))
            return (row_band, -cx)

        raw_boxes.sort(key=_panel_sort_key)

        panels: List[Dict[str, Any]] = []
        for idx, (x0, y0, x1, y1) in enumerate(raw_boxes, 1):
            panels.append(
                {
                    "id": f"p{page_img.page_num}_panel{idx}",
                    "page": page_img.page_num,
                    "rank": idx,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                }
            )
        return panels

    @staticmethod
    def assign_boxes_to_panels(boxes: List[Dict[str, Any]], panels: List[Dict[str, Any]]) -> None:
        """Assign each text box a panel_id and panel_rank (in-place)."""
        if not panels:
            for b in boxes:
                b["panel_id"] = "p1_panel1"
                b["panel_rank"] = 1
            return

        for b in boxes:
            bx0, by0, bx1, by1 = b["bbox"]
            cx = 0.5 * (bx0 + bx1)
            cy = 0.5 * (by0 + by1)
            assigned = None
            best_dist = float("inf")
            for p in panels:
                px0, py0, px1, py1 = p["bbox"]
                if px0 <= cx <= px1 and py0 <= cy <= py1:
                    assigned = p
                    break
                pcx = 0.5 * (px0 + px1)
                pcy = 0.5 * (py0 + py1)
                dist = (cx - pcx) ** 2 + (cy - pcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    assigned = p
            b["panel_id"] = assigned["id"] if assigned else panels[0]["id"]
            b["panel_rank"] = assigned["rank"] if assigned else 1
