"""Contrast-enhanced multi-scale secondary OCR / VLM crop transcriber."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from src.vision.detect_text import get_rapidocr
from src.vision.preprocess import PageImage


class VLMCropOCREngine:
    """Secondary OCR engine using CLAHE contrast enhancement + sharpening on 2x upscaled crops."""

    def __init__(self, upscale_factor: float = 2.0) -> None:
        self.upscale_factor = upscale_factor

    def recognize_box(self, page_img: PageImage, box: Dict[str, Any]) -> Tuple[str, float]:
        crop = page_img.crop(box["bbox"], pad=8, upscale=self.upscale_factor, grayscale=True)
        # Apply CLAHE and unsharp masking to bring out faint/stylised lettering
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(crop)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)

        engine = get_rapidocr()
        res, _ = engine(sharp_rgb)
        if not res:
            return "", 0.0
        items = sorted(res, key=lambda r: (r[0][0][1], r[0][0][0]))
        texts = [str(r[1]).strip() for r in items if str(r[1]).strip()]
        confs = [float(r[2]) for r in items]
        return " ".join(texts).strip(), float(np.mean(confs)) if confs else 0.0
