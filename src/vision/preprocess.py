"""M1 — Image preprocessing module for manga pages."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PageImage:
    """Container for a loaded manga page image in RGB and grayscale."""

    path: Path
    page_num: int
    rgb: np.ndarray
    gray: np.ndarray
    width: int
    height: int

    def crop(
        self,
        bbox: Tuple[int, int, int, int] | list[int],
        pad: int = 4,
        upscale: float = 1.0,
        grayscale: bool = False,
    ) -> np.ndarray:
        """Extract a padded, optionally upscaled crop from the page without cropping page margins."""
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(self.width, x1 + pad)
        y1 = min(self.height, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((16, 16) if grayscale else (16, 16, 3), dtype=np.uint8)
        src = self.gray if grayscale else self.rgb
        patch = src[y0:y1, x0:x1].copy()
        if upscale > 1.01:
            new_w = max(1, int(round(patch.shape[1] * upscale)))
            new_h = max(1, int(round(patch.shape[0] * upscale)))
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return patch

    def resized_for_vlm(self, max_pixels: int = 1200000) -> Tuple[np.ndarray, float]:
        """Return a resized RGB copy capped at max_pixels and its scale factor."""
        total = self.width * self.height
        if total <= max_pixels:
            return self.rgb, 1.0
        scale = math.sqrt(max_pixels / float(total))
        new_w = max(1, int(round(self.width * scale)))
        new_h = max(1, int(round(self.height * scale)))
        resized = cv2.resize(self.rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale


class Preprocessor:
    """M1 Preprocessor: loads PNG into RGB/gray and provides crop/resize helpers."""

    def __init__(self, max_pixels: int = 1200000, upscale_factor: float = 2.0) -> None:
        self.max_pixels = max_pixels
        self.upscale_factor = upscale_factor

    def run(self, image_path: str | Path, page_num: int = 1) -> PageImage:
        p = Path(image_path)
        pil_img = Image.open(p).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        return PageImage(path=p, page_num=page_num, rgb=rgb, gray=gray, width=w, height=h)
