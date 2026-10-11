"""M5 — OCR ensemble orchestrator combining primary OCR, secondary OCR, punctuation solver, and correction."""
from __future__ import annotations

from typing import Any, Dict, List

from rapidfuzz.distance import Levenshtein

from src.ocr.correct import clean_ocr_text
from src.ocr.engine_paddle import PaddleOCREngine
from src.ocr.engine_tesseract import TesseractOCREngine
from src.ocr.engine_vlm import VLMCropOCREngine
from src.ocr.punct import PunctuationResolver
from src.vision.preprocess import PageImage


class OCREnsemble:
    """Runs configured OCR engine(s), computes disagreement features, and applies constrained correction."""

    def __init__(
        self,
        primary_engine: str = "ensemble",
        fallback_engine: str = "rapidocr",
        upscale_factor: float = 2.0,
        enable_punct_solver: bool = True,
        max_edit_distance_correction: float = 0.25,
    ) -> None:
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.enable_punct_solver = enable_punct_solver
        self.max_edit_distance_correction = max_edit_distance_correction

        self.tesseract = TesseractOCREngine()
        self.paddle = PaddleOCREngine(upscale_factor=upscale_factor)
        self.vlm_crop = VLMCropOCREngine(upscale_factor=upscale_factor)
        self.punct = PunctuationResolver()

    def run(self, page_img: PageImage, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for box in boxes:
            if box.get("det_class") == "punct_bubble":
                if self.enable_punct_solver:
                    crop_gray = page_img.crop(box["bbox"], pad=2, upscale=1.0, grayscale=True)
                    p_txt, p_conf = self.punct.resolve(crop_gray)
                    box["text_raw"] = p_txt
                    box["text_alt"] = p_txt
                    box["ocr_conf"] = p_conf
                    box["ocr_disagreement"] = 0.0
                    box["is_punct_only"] = bool(p_txt)
                else:
                    box["text_raw"] = ""
                    box["text_alt"] = ""
                    box["ocr_conf"] = 0.0
                    box["ocr_disagreement"] = 0.0
                    box["is_punct_only"] = True
                continue

            if self.primary_engine == "tesseract":
                if not self.tesseract.has_tesseract and "pre_ocr_text" in box:
                    t1, c1 = str(box.get("pre_ocr_text", "")), float(box.get("pre_ocr_conf", 0.0))
                else:
                    crop_rgb = page_img.crop(box["bbox"], pad=4, upscale=1.0, grayscale=False)
                    t1, c1 = self.tesseract.recognize_crop(crop_rgb)
                t2, c2 = t1, c1
            elif self.primary_engine == "rapidocr":
                if "pre_ocr_text" in box:
                    t1, c1 = str(box.get("pre_ocr_text", "")).strip(), float(box.get("pre_ocr_conf", 0.0))
                else:
                    t1, c1 = self.paddle.recognize_box(page_img, box)
                t2, c2 = t1, c1
            else:
                if "pre_ocr_text" in box:
                    t1, c1 = str(box.get("pre_ocr_text", "")).strip(), float(box.get("pre_ocr_conf", 0.0))
                else:
                    t1, c1 = self.vlm_crop.recognize_box(page_img, box)
                t2, c2 = t1, c1

            if self.primary_engine == "tesseract":
                t1_clean, t2_clean = t1.strip(), t2.strip()
            else:
                t1_clean = clean_ocr_text(t1, max_edit_ratio=self.max_edit_distance_correction)
                t2_clean = clean_ocr_text(t2, max_edit_ratio=self.max_edit_distance_correction)

            # Pick higher-confidence or non-empty transcript
            if (c2 > c1 + 0.06 and t2_clean) or (not t1_clean and t2_clean):
                chosen_text, alt_text = t2_clean, t1_clean
                chosen_conf = c2
            else:
                chosen_text, alt_text = t1_clean, t2_clean
                chosen_conf = c1

            # If still empty and inside a small speech bubble, check punctuation resolver
            is_punct = False
            if not chosen_text and self.enable_punct_solver and box.get("inside_bubble", False):
                crop_gray = page_img.crop(box["bbox"], pad=4, upscale=1.0, grayscale=True)
                p_txt, p_conf = self.punct.resolve(crop_gray)
                if p_txt:
                    chosen_text = p_txt
                    alt_text = p_txt
                    chosen_conf = p_conf
                    is_punct = True

            if chosen_text and all(not ch.isalnum() for ch in chosen_text):
                is_punct = True

            max_l = max(len(chosen_text), len(alt_text), 1)
            disagreement = float(Levenshtein.distance(chosen_text.lower(), alt_text.lower())) / float(max_l)

            box["text_raw"] = chosen_text
            box["text_alt"] = alt_text
            box["ocr_conf"] = round(float(chosen_conf), 4)
            box["ocr_disagreement"] = round(disagreement, 4)
            box["is_punct_only"] = is_punct

        return boxes
