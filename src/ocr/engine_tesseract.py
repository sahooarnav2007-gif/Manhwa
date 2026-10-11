"""Baseline OCR engine (Tesseract with local ONNX crop fallback if Tesseract binary is absent)."""
from __future__ import annotations

import shutil
from typing import Tuple

import cv2
import numpy as np

from src.vision.detect_text import get_rapidocr


class TesseractOCREngine:
    """Runs Tesseract OCR on a crop if installed, else falls back to unrefined single-crop recognition."""

    def __init__(self) -> None:
        self.has_tesseract = shutil.which("tesseract") is not None

    def recognize_crop(self, crop_rgb: np.ndarray) -> Tuple[str, float]:
        if self.has_tesseract:
            try:
                import pytesseract

                data = pytesseract.image_to_data(
                    crop_rgb,
                    config="--psm 6",
                    output_type=pytesseract.Output.DICT,
                )
                words = []
                confs = []
                for w, c in zip(data.get("text", []), data.get("conf", [])):
                    w_str = str(w).strip()
                    try:
                        c_val = float(c)
                    except Exception:
                        c_val = -1.0
                    if w_str and c_val >= 0:
                        words.append(w_str)
                        confs.append(c_val / 100.0)
                text = " ".join(words).strip()
                conf = float(np.mean(confs)) if confs else 0.0
                return text, conf
            except Exception:
                pass

        # Fallback for T0 when OS tesseract binary is not installed: low-res gray thresholded crop via local ONNX
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
        _, th = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
        th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        engine = get_rapidocr()
        res, _ = engine(th_rgb)
        if not res:
            return "", 0.0
        # Sort strictly top-to-bottom, left-to-right
        items = sorted(res, key=lambda r: (r[0][0][1], r[0][0][0]))
        texts = [str(r[1]).strip() for r in items if str(r[1]).strip()]
        confs = [float(r[2]) for r in items]
        return " ".join(texts).strip(), float(np.mean(confs)) if confs else 0.0
