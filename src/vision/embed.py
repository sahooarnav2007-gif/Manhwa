"""M4b — Grayscale line-art & screen-tone appearance embeddings + optional contrastive projection head."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.vision.preprocess import PageImage


class CharacterEmbedder:
    """Computes multi-crop grayscale appearance embeddings (hair tone, screen-tone texture, HOG contours)."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim
        self.projection_matrix: Optional[np.ndarray] = None

    def _compute_raw_descriptor(self, crop_gray: np.ndarray) -> np.ndarray:
        """Compute a 64-D L2-normalized descriptor from a character crop (head + upper body)."""
        resized = cv2.resize(crop_gray, (64, 64), interpolation=cv2.INTER_AREA)
        f_img = resized.astype(np.float32) / 255.0

        # 1. Spatial hair/tone intensity & std pyramid across 4x4 grid (32 features: mean + std per cell)
        means = []
        stds = []
        for r in range(4):
            for c in range(4):
                cell = f_img[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
                means.append(float(np.mean(cell)))
                stds.append(float(np.std(cell)))

        # 2. Gradient orientation histogram on upper (hair/face) and lower (collar/clothes) halves (16 features)
        gx = cv2.Sobel(f_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f_img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0, 1]
        bins = np.clip((ang * 8).astype(np.int32), 0, 7)

        orient_feats = []
        for half in (slice(0, 32), slice(32, 64)):
            b_half = bins[half, :]
            m_half = mag[half, :]
            hist = [float(np.sum(m_half[b_half == b])) for b in range(8)]
            norm_h = sum(hist) + 1e-6
            orient_feats.extend([v / norm_h for v in hist])

        # 3. Screen-tone & ink-density quantile histogram (16 features: 8 upper + 8 lower intensity bins)
        tone_feats = []
        for half in (slice(0, 32), slice(32, 64)):
            hist, _ = np.histogram(f_img[half, :], bins=8, range=(0.0, 1.0), density=False)
            hist_f = hist.astype(np.float32) / float(max(1, f_img[half, :].size))
            tone_feats.extend(hist_f.tolist())

        vec = np.array(means + stds + orient_feats + tone_feats, dtype=np.float32)
        if len(vec) != self.embedding_dim:
            vec = np.resize(vec, (self.embedding_dim,))
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def fit_projection(self, pairs_x1: np.ndarray, pairs_x2: np.ndarray, labels: np.ndarray) -> None:
        """Fit a regularized discriminative linear projection head on weak same-speaker character pairs."""
        if len(labels) < 10:
            return
        diffs = pairs_x1 - pairs_x2
        pos = diffs[labels == 1]
        neg = diffs[labels == 0]
        if len(pos) < 3 or len(neg) < 3:
            return
        cov_pos = (pos.T @ pos) / float(len(pos)) + 0.15 * np.eye(diffs.shape[1], dtype=np.float32)
        cov_neg = (neg.T @ neg) / float(len(neg)) + 0.05 * np.eye(diffs.shape[1], dtype=np.float32)
        try:
            mat = np.linalg.solve(cov_pos, cov_neg)
            eigvals, eigvecs = np.linalg.eigh(0.5 * (mat + mat.T))
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            weights = np.sqrt(np.clip(eigvals[order], 0.1, 10.0))
            self.projection_matrix = (eigvecs * weights).astype(np.float32)
        except Exception:
            self.projection_matrix = None

    def run(self, page_img: PageImage, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach L2-normalized embedding list to each character dict."""
        for ch in characters:
            crop = page_img.crop(ch["bbox"], pad=2, upscale=1.0, grayscale=True)
            vec = self._compute_raw_descriptor(crop)
            if self.projection_matrix is not None and self.projection_matrix.shape[0] == len(vec):
                vec = vec @ self.projection_matrix
                norm = float(np.linalg.norm(vec))
                if norm > 1e-6:
                    vec = vec / norm
            ch["embedding"] = [round(float(v), 5) for v in vec]
        return characters
