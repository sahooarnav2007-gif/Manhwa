"""Punctuation-only balloon classifier (M5 component)."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PunctuationResolver:
    """Resolves punctuation-only balloon crops into '...', '?', '!', '?!', '!?', '!!', or empty string."""

    def resolve(self, crop_gray: np.ndarray) -> Tuple[str, float]:
        if crop_gray.size == 0:
            return "", 0.0
        h, w = crop_gray.shape[:2]
        # Ignore outer 10% border of the crop
        mx, my = max(1, int(w * 0.10)), max(1, int(h * 0.10))
        inner = crop_gray[my : h - my, mx : w - mx]
        if inner.size == 0:
            return "", 0.0

        dark = (inner < 130).astype(np.uint8) * 255
        n_cc, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
        comps = []
        ih, iw = inner.shape[:2]
        for idx in range(1, n_cc):
            x, y, cw, ch, area = stats[idx]
            if 3 <= area <= (ih * iw * 0.25):
                comps.append((x, y, cw, ch, area, centroids[idx][0], centroids[idx][1]))

        if not comps:
            return "", 0.0

        # Sort left to right
        comps.sort(key=lambda c: c[5])
        n = len(comps)
        aspects = [c[3] / float(max(1, c[2])) for c in comps]  # height / width

        # Case 1: 3 horizontally aligned small roundish dots -> "..."
        if n == 3 and all(a < 1.8 for a in aspects):
            ys = [c[6] for c in comps]
            if (max(ys) - min(ys)) < ih * 0.28:
                return "...", 0.78

        # Case 2: 2 components (one vertical stem + one dot below it) -> "!" or "?"
        if n == 2:
            tall = [c for c in comps if (c[3] / float(max(1, c[2]))) >= 1.6]
            dots = [c for c in comps if (c[3] / float(max(1, c[2]))) < 1.6]
            if len(tall) == 1 and len(dots) == 1:
                stem = tall[0]
                dot = dots[0]
                if dot[6] > stem[6] and abs(dot[5] - stem[5]) < stem[2] * 1.8:
                    # Distinguish '?' (wider top curve) vs '!' (narrow vertical stem)
                    if (stem[3] / float(max(1, stem[2]))) < 2.6:
                        return "?", 0.72
                    return "!", 0.74

        # Case 3: 4 components (2 stems + 2 dots) -> "?!" or "!!"
        if n == 4:
            tall = [c for c in comps if (c[3] / float(max(1, c[2]))) >= 1.5]
            dots = [c for c in comps if (c[3] / float(max(1, c[2]))) < 1.5]
            if len(tall) == 2 and len(dots) == 2:
                tall.sort(key=lambda c: c[5])
                a0 = tall[0][3] / float(max(1, tall[0][2]))
                a1 = tall[1][3] / float(max(1, tall[1][2]))
                if a0 < 2.6 and a1 >= 2.3:
                    return "?!", 0.75
                if a0 >= 2.3 and a1 < 2.6:
                    return "!?", 0.75
                return "!!", 0.72

        return "", 0.0
