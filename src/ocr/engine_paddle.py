"""PaddleOCR v4 ONNX (RapidOCR) engine on upscaled manga text crops."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from src.vision.detect_text import get_rapidocr
from src.vision.preprocess import PageImage


class PaddleOCREngine:
    """Runs PaddleOCR v4 ONNX recognition on padded, upscaled text crops when full-page OCR confidence is low."""

    def __init__(self, upscale_factor: float = 1.5) -> None:
        self.upscale_factor = upscale_factor

    def recognize_box(self, page_img: PageImage, box: Dict[str, Any]) -> Tuple[str, float]:
        pre_text = str(box.get("pre_ocr_text", "")).strip()
        pre_conf = float(box.get("pre_ocr_conf", 0.0))
        # Full-resolution page DBNet+PP-OCRv4 already recognized the lines inside this box;
        # only re-run crop recognition if confidence is low (<0.72) or text is empty.
        if pre_text and pre_conf >= 0.72:
            return pre_text, pre_conf

        crop = page_img.crop(box["bbox"], pad=4, upscale=self.upscale_factor, grayscale=False)
        engine = get_rapidocr()
        res, _ = engine(crop)
        if not res:
            return pre_text, pre_conf

        items = sorted(res, key=lambda r: (r[0][0][1], r[0][0][0]))
        texts = [str(r[1]).strip() for r in items if str(r[1]).strip()]
        confs = [float(r[2]) for r in items]
        crop_text = " ".join(texts).strip()
        crop_conf = float(sum(confs) / len(confs)) if confs else 0.0

        if crop_conf >= pre_conf and crop_text:
            return crop_text, crop_conf
        return pre_text, pre_conf
