"""M2 — Text-region and speech-balloon detection (OpenCV baseline + Pretrained DBNet ONNX hybrid)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.vision.preprocess import PageImage

_RAPIDOCR_INSTANCE = None


def get_rapidocr():
    """Singleton loader for local RapidOCR (DBNet + SVTR/CRNN ONNX)."""
    global _RAPIDOCR_INSTANCE
    if _RAPIDOCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR_INSTANCE = RapidOCR(
            det_limit_side_len=736,
            intra_op_num_threads=2,
            inter_op_num_threads=1,
        )
    return _RAPIDOCR_INSTANCE


def box_iou(a: List[int] | Tuple[int, int, int, int], b: List[int] | Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union of two [x0, y0, x1, y1] boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


class TextDetector:
    """Detects speech balloons, narration captions, free text, and punctuation-only balloons."""

    def __init__(
        self,
        mode: str = "hybrid",
        det_conf_threshold: float = 0.35,
        nms_iou_threshold: float = 0.45,
        min_box_area: int = 180,
        max_box_area_ratio: float = 0.35,
        enable_punct_candidates: bool = True,
    ) -> None:
        self.mode = mode
        self.det_conf_threshold = det_conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_box_area = min_box_area
        self.max_box_area_ratio = max_box_area_ratio
        self.enable_punct_candidates = enable_punct_candidates

    def _detect_white_balloons(self, gray: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Find enclosed white speech balloons and rectangular caption boxes using OpenCV."""
        h, w = gray.shape[:2]
        page_area = float(h * w)

        # White interior threshold
        white = (gray >= 228).astype(np.uint8) * 255
        # Remove thin white leaks across balloon borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_eroded = cv2.erode(white, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_eroded, connectivity=8)
        balloon_map = np.zeros((h, w), dtype=np.int32)
        balloons: List[Dict[str, Any]] = []

        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area < 450 or area > page_area * self.max_box_area_ratio:
                continue
            # Reject page background touching 3+ image borders
            touches = int(x <= 2) + int(y <= 2) + int(x + bw >= w - 2) + int(y + bh >= h - 2)
            if touches >= 3 or (touches >= 2 and area > page_area * 0.08):
                continue
            aspect = bw / float(max(1, bh))
            if aspect > 8.0 or aspect < 0.12:
                continue

            # Extract component mask in ROI
            roi_mask = (labels[y : y + bh, x : x + bw] == lbl).astype(np.uint8) * 255
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 400:
                continue
            peri = max(1.0, cv2.arcLength(cnt, True))
            circularity = (4.0 * math.pi * cnt_area) / (peri * peri)
            rect_fill = cnt_area / float(max(1, bw * bh))
            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = cnt_area / hull_area

            if solidity < 0.45:
                continue

            # Check dark text strokes inside the filled balloon contour
            filled = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
            roi_gray = gray[y : y + bh, x : x + bw]
            dark_inside = (roi_gray < 140) & (filled > 0)
            dark_ratio = float(np.sum(dark_inside)) / float(max(1, np.sum(filled > 0)))
            if dark_ratio < 0.003 or dark_ratio > 0.55:
                continue

            # Estimate tail tip: point on contour farthest from centroid weighted by convexity defect
            cx_local = float(centroids[lbl][0] - x)
            cy_local = float(centroids[lbl][1] - y)
            pts = cnt.reshape(-1, 2).astype(np.float32)
            # Normalize by half-width and half-height so ellipse points have norm ~1
            norm_dx = (pts[:, 0] - cx_local) / max(10.0, bw * 0.5)
            norm_dy = (pts[:, 1] - cy_local) / max(10.0, bh * 0.5)
            norm_dist = np.hypot(norm_dx, norm_dy)
            max_idx = int(np.argmax(norm_dist))
            tail_tip = [int(x + pts[max_idx, 0]), int(y + pts[max_idx, 1])]
            tail_vec = [float(norm_dx[max_idx]), float(norm_dy[max_idx])]
            tail_prominence = float(norm_dist[max_idx])

            is_caption = bool(rect_fill > 0.84 and solidity > 0.92 and tail_prominence < 1.38)
            bid = len(balloons) + 1
            # Write balloon ID into balloon_map inside filled mask
            sub_map = balloon_map[y : y + bh, x : x + bw]
            sub_map[filled > 0] = bid

            balloons.append(
                {
                    "balloon_id": bid,
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "area": float(cnt_area),
                    "circularity": float(circularity),
                    "rect_fill": float(rect_fill),
                    "solidity": float(solidity),
                    "dark_ratio": float(dark_ratio),
                    "is_caption": is_caption,
                    "tail_tip": tail_tip,
                    "tail_vec": tail_vec,
                    "tail_prominence": tail_prominence,
                }
            )

        return balloon_map, balloons

    def _get_or_run_page_ocr(self, page_img: PageImage) -> List[ Any ]:
        """Run full-page DBNet+PP-OCRv4 once per image and cache raw line detections on disk."""
        import json
        from pathlib import Path

        seq_id = page_img.path.parent.name
        cache_file = Path("artifacts/cache/raw_page_ocr") / seq_id / f"page_{page_img.page_num}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        engine = get_rapidocr()
        ocr_res, _ = engine(page_img.rgb, use_cls=False)
        serializable = []
        if ocr_res:
            for item in ocr_res:
                pts = [[float(p[0]), float(p[1])] for p in item[0]]
                serializable.append([pts, str(item[1]), float(item[2])])

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        return serializable

    def _detect_opencv_only(
        self,
        page_img: PageImage,
        balloon_map: np.ndarray,
        balloons: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Tier 0 OpenCV-only text-block detector inside detected white balloons."""
        gray = page_img.gray
        raw_page_lines = []
        ocr_res = self._get_or_run_page_ocr(page_img)
        for item in ocr_res:
            pts = np.array(item[0], dtype=np.float32)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            y_top = float(np.min(pts[:, 1]))
            x_left = float(np.min(pts[:, 0]))
            raw_page_lines.append((cx, cy, y_top, x_left, str(item[1]).strip(), float(item[2])))

        boxes: List[Dict[str, Any]] = []
        for idx, bal in enumerate(balloons, 1):
            x0, y0, x1, y1 = bal["bbox"]
            roi = gray[y0:y1, x0:x1]
            bh, bw = roi.shape[:2]
            pad_x = max(3, int(bw * 0.08))
            pad_y = max(3, int(bh * 0.08))
            inner = roi[pad_y : bh - pad_y, pad_x : bw - pad_x]
            if inner.size == 0:
                continue
            ys, xs = np.where(inner < 130)
            if len(xs) < 8:
                continue
            tx0 = x0 + pad_x + int(np.min(xs))
            ty0 = y0 + pad_y + int(np.min(ys))
            tx1 = x0 + pad_x + int(np.max(xs)) + 1
            ty1 = y0 + pad_y + int(np.max(ys)) + 1
            if (tx1 - tx0) * (ty1 - ty0) < self.min_box_area:
                continue

            matched_lines = [
                (y_top, x_left, txt, conf)
                for (cx, cy, y_top, x_left, txt, conf) in raw_page_lines
                if x0 <= cx <= x1 and y0 <= cy <= y1 and txt
            ]
            matched_lines.sort(key=lambda item: (item[0], item[1]))
            pre_txt = " ".join(m[2] for m in matched_lines).strip()
            pre_conf = float(np.mean([m[3] for m in matched_lines])) if matched_lines else 0.0

            boxes.append(
                {
                    "id": f"p{page_img.page_num}_b{idx:02d}",
                    "page": page_img.page_num,
                    "bbox": [tx0, ty0, tx1, ty1],
                    "bubble_bbox": bal["bbox"],
                    "det_class": "caption" if bal["is_caption"] else "bubble",
                    "det_conf": float(min(0.95, 0.55 + bal["solidity"] * 0.4)),
                    "balloon_id": f"p{page_img.page_num}_bal{bal['balloon_id']}",
                    "inside_bubble": True,
                    "is_caption_box": bal["is_caption"],
                    "circularity": bal["circularity"],
                    "rect_fill": bal["rect_fill"],
                    "solidity": bal["solidity"],
                    "tail_tip": bal["tail_tip"],
                    "tail_vec": bal["tail_vec"],
                    "tail_prominence": bal["tail_prominence"],
                    "pre_ocr_text": pre_txt,
                    "pre_ocr_conf": pre_conf,
                }
            )
        return boxes

    def run(self, page_img: PageImage) -> List[Dict[str, Any]]:
        """Run M2 detection on a PageImage and return a list of detected text/balloon boxes."""
        gray = page_img.gray
        h, w = gray.shape[:2]
        balloon_map, balloons = self._detect_white_balloons(gray)
        bal_by_id = {b["balloon_id"]: b for b in balloons}

        if self.mode == "opencv":
            return self._detect_opencv_only(page_img, balloon_map, balloons)

        ocr_res = self._get_or_run_page_ocr(page_img)
        raw_lines: List[Dict[str, Any]] = []
        if ocr_res:
            for item in ocr_res:
                pts_list, text, conf = item[0], str(item[1]), float(item[2])
                conf_val = float(conf)
                if conf_val < self.det_conf_threshold * 0.5:
                    continue
                pts = np.array(pts_list, dtype=np.float32)
                lx0 = max(0, int(np.floor(np.min(pts[:, 0]))))
                ly0 = max(0, int(np.floor(np.min(pts[:, 1]))))
                lx1 = min(w, int(np.ceil(np.max(pts[:, 0]))))
                ly1 = min(h, int(np.ceil(np.max(pts[:, 1]))))
                if (lx1 - lx0) < 4 or (ly1 - ly0) < 4:
                    continue
                # Determine which balloon this line falls inside
                roi_bal = balloon_map[ly0:ly1, lx0:lx1]
                nonzero = roi_bal[roi_bal > 0]
                bid = int(np.bincount(nonzero).argmax()) if len(nonzero) > 0 else 0
                raw_lines.append(
                    {
                        "bbox": [lx0, ly0, lx1, ly1],
                        "text": text.strip(),
                        "conf": conf_val,
                        "bid": bid,
                    }
                )

        # Group lines into single-lobe vertical speech-balloon paragraphs.
        # Lines that are side-by-side horizontally (overlapping in Y) belong to separate lobes/balloons
        # and must NOT be merged into the same paragraph (otherwise their lines get interleaved by Y).
        used = [False] * len(raw_lines)
        raw_lines.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        groups: List[List[Dict[str, Any]]] = []
        for i, line in enumerate(raw_lines):
            if used[i]:
                continue
            used[i] = True
            grp = [line]
            changed = True
            while changed:
                changed = False
                gx0 = min(x["bbox"][0] for x in grp)
                gx1 = max(x["bbox"][2] for x in grp)
                last = grp[-1]
                lx0, ly0, lx1, ly1 = last["bbox"]
                lh = max(1, ly1 - ly0)
                lw = max(1, lx1 - lx0)
                avg_lh = float(np.mean([x["bbox"][3] - x["bbox"][1] for x in grp]))
                gbids = {x["bid"] for x in grp if x["bid"] > 0}

                best_j = -1
                best_gap = float("inf")
                for j, cand in enumerate(raw_lines):
                    if used[j]:
                        continue
                    cx0, cy0, cx1, cy1 = cand["bbox"]
                    ch = max(1, cy1 - cy0)
                    cw = max(1, cx1 - cx0)

                    same_pos_bid = (cand["bid"] > 0) and (cand["bid"] in gbids)
                    both_no_bid = (cand["bid"] == 0) and (len(gbids) == 0)
                    if not (same_pos_bid or both_no_bid):
                        continue

                    # Candidate line's vertical center must be below the last line's vertical center (not side-by-side)
                    l_cy = 0.5 * (ly0 + ly1)
                    c_cy = 0.5 * (cy0 + cy1)
                    dy_center = c_cy - l_cy
                    if dy_center < 0.45 * min(lh, ch):
                        continue
                    v_gap = cy0 - ly1
                    max_v_gap = avg_lh * (0.95 if same_pos_bid else 0.65)
                    if v_gap > max_v_gap:
                        continue

                    # Check horizontal column alignment with the group and last line
                    gw = max(1, gx1 - gx0)
                    h_overlap_grp = min(gx1, cx1) - max(gx0, cx0)
                    g_cx = 0.5 * (gx0 + gx1)
                    c_cx = 0.5 * (cx0 + cx1)
                    center_dx = abs(g_cx - c_cx)
                    max_dx = max(0.52 * gw, avg_lh * 2.2)

                    if h_overlap_grp >= 0.15 * min(gw, cw) and center_dx <= max_dx:
                        if 0.42 <= (ch / max(1.0, avg_lh)) <= 2.3:
                            if dy_center < best_gap:
                                best_gap = dy_center
                                best_j = j

                if best_j >= 0:
                    used[best_j] = True
                    grp.append(raw_lines[best_j])
                    changed = True
            groups.append(grp)

        boxes: List[Dict[str, Any]] = []
        covered_bids = set()

        for g_idx, grp in enumerate(groups, 1):
            # Sort lines inside a single balloon top-to-bottom, left-to-right (English reading order inside a balloon)
            grp.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
            bx0 = min(r["bbox"][0] for r in grp)
            by0 = min(r["bbox"][1] for r in grp)
            bx1 = max(r["bbox"][2] for r in grp)
            by1 = max(r["bbox"][3] for r in grp)

            bids = [r["bid"] for r in grp if r["bid"] > 0]
            bid = max(set(bids), key=bids.count) if bids else 0
            if bid > 0:
                covered_bids.add(bid)
            bal = bal_by_id.get(bid)

            joined_text = " ".join(r["text"] for r in grp if r["text"])
            mean_conf = float(np.mean([r["conf"] for r in grp]))

            # Compute local patch brightness and stroke contrast around the text box
            roi_gray = gray[by0:by1, bx0:bx1]
            bg_white_ratio = float(np.mean(roi_gray >= 215)) if roi_gray.size > 0 else 0.0
            stroke_dark_ratio = float(np.mean(roi_gray < 110)) if roi_gray.size > 0 else 0.0

            inside_bubble = bool(bid > 0 or bg_white_ratio >= 0.52)
            is_caption = bool(bal["is_caption"]) if bal else False
            if not is_caption and inside_bubble:
                # Check if there is a rectangular frame around the group
                pad = 8
                rx0, ry0 = max(0, bx0 - pad), max(0, by0 - pad)
                rx1, ry1 = min(w, bx1 + pad), min(h, by1 + pad)
                if bal and bal["rect_fill"] > 0.84 and bal["solidity"] > 0.92:
                    is_caption = True

            det_class = "caption" if is_caption else ("text_bubble" if inside_bubble else "text_free")
            boxes.append(
                {
                    "id": f"p{page_img.page_num}_b{g_idx:02d}",
                    "page": page_img.page_num,
                    "bbox": [int(bx0), int(by0), int(bx1), int(by1)],
                    "bubble_bbox": bal["bbox"] if bal else [int(bx0), int(by0), int(bx1), int(by1)],
                    "det_class": det_class,
                    "det_conf": mean_conf,
                    "balloon_id": f"p{page_img.page_num}_bal{bid}" if bid > 0 else f"p{page_img.page_num}_free{g_idx}",
                    "inside_bubble": inside_bubble,
                    "is_caption_box": is_caption,
                    "bg_white_ratio": bg_white_ratio,
                    "stroke_dark_ratio": stroke_dark_ratio,
                    "circularity": bal["circularity"] if bal else 0.5,
                    "rect_fill": bal["rect_fill"] if bal else 0.5,
                    "solidity": bal["solidity"] if bal else 0.7,
                    "tail_tip": bal["tail_tip"] if bal else [int(0.5 * (bx0 + bx1)), int(by1)],
                    "tail_vec": bal["tail_vec"] if bal else [0.0, 1.0],
                    "tail_prominence": bal["tail_prominence"] if bal else 1.0,
                    "line_count": len(grp),
                    "pre_ocr_text": joined_text,
                    "pre_ocr_conf": mean_conf,
                    "line_boxes": [r["bbox"] for r in grp],
                    "line_texts": [r["text"] for r in grp],
                }
            )

        # Punctuation-only balloon candidates (balloons with no DBNet text inside)
        if self.enable_punct_candidates:
            for bal in balloons:
                if bal["balloon_id"] in covered_bids:
                    continue
                bx0, by0, bx1, by1 = bal["bbox"]
                # Check if any existing box overlaps this balloon
                if any(box_iou(b["bbox"], bal["bbox"]) > 0.1 for b in boxes):
                    continue
                if bal["solidity"] < 0.78 or bal["circularity"] < 0.35:
                    continue
                roi = gray[by0:by1, bx0:bx1]
                bh, bw = roi.shape[:2]
                mx, my = max(4, int(bw * 0.18)), max(4, int(bh * 0.18))
                center_roi = roi[my : bh - my, mx : bw - mx]
                if center_roi.size == 0:
                    continue
                dark_cc = (center_roi < 120).astype(np.uint8) * 255
                n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(dark_cc, connectivity=8)
                valid_dots = [s for s in cc_stats[1:] if 4 <= s[4] <= (bw * bh * 0.08)]
                if 1 <= len(valid_dots) <= 6:
                    idx = len(boxes) + 1
                    boxes.append(
                        {
                            "id": f"p{page_img.page_num}_b{idx:02d}",
                            "page": page_img.page_num,
                            "bbox": [int(bx0 + mx), int(by0 + my), int(bx1 - mx), int(by1 - my)],
                            "bubble_bbox": bal["bbox"],
                            "det_class": "punct_bubble",
                            "det_conf": 0.65,
                            "balloon_id": f"p{page_img.page_num}_bal{bal['balloon_id']}",
                            "inside_bubble": True,
                            "is_caption_box": False,
                            "bg_white_ratio": 0.92,
                            "stroke_dark_ratio": bal["dark_ratio"],
                            "circularity": bal["circularity"],
                            "rect_fill": bal["rect_fill"],
                            "solidity": bal["solidity"],
                            "tail_tip": bal["tail_tip"],
                            "tail_vec": bal["tail_vec"],
                            "tail_prominence": bal["tail_prominence"],
                            "line_count": 1,
                            "pre_ocr_text": "",
                            "pre_ocr_conf": 0.5,
                            "line_boxes": [],
                            "line_texts": [],
                        }
                    )

        return boxes
