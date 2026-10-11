"""M6 — Learned story-text and narration classifier trained on weak box-level alignments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.loader import load_score_module
from src.learn.features import extract_box_features

_SCORE = load_score_module()
norm = _SCORE.norm


class StoryTextClassifier:
    """Learned gradient-boosted classifier predicting story inclusion probability and narration vs speech."""

    def __init__(
        self,
        enabled: bool = True,
        story_prob_threshold: float = 0.46,
        narration_prob_threshold: float = 0.52,
        seed: int = 42,
    ) -> None:
        self.enabled = enabled
        self.story_prob_threshold = story_prob_threshold
        self.narration_prob_threshold = narration_prob_threshold
        self.seed = seed
        self.story_model: Optional[HistGradientBoostingClassifier] = None
        self.narration_model: Optional[HistGradientBoostingClassifier] = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_story: np.ndarray,
        y_narration: np.ndarray,
    ) -> "StoryTextClassifier":
        if len(X) < 10 or len(np.unique(y_story)) < 2:
            self.is_fitted = False
            return self

        self.story_model = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            learning_rate=0.06,
            min_samples_leaf=8,
            l2_regularization=1.5,
            random_state=self.seed,
        )
        self.story_model.fit(X, y_story)

        # Fit narration model on positive story boxes
        pos_mask = y_story == 1
        X_pos = X[pos_mask]
        y_nar_pos = y_narration[pos_mask]
        if len(X_pos) >= 10 and len(np.unique(y_nar_pos)) >= 2:
            self.narration_model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=6,
                l2_regularization=2.0,
                class_weight="balanced",
                random_state=self.seed,
            )
            self.narration_model.fit(X_pos, y_nar_pos)

        self.is_fitted = True
        return self

    def predict_page_boxes(
        self,
        boxes: List[Dict[str, Any]],
        page_w: int,
        page_h: int,
    ) -> List[Dict[str, Any]]:
        if not boxes:
            return boxes

        if not self.enabled:
            for b in boxes:
                txt = norm(str(b.get("text_raw") or b.get("pre_ocr_text") or ""))
                b["story_prob"] = 1.0 if txt else 0.0
                b["narration_prob"] = 0.0
                b["kind"] = "speech" if txt else "exclude"
            return boxes

        feats = np.stack([extract_box_features(b, boxes, page_w, page_h) for b in boxes], axis=0)

        if self.is_fitted and self.story_model is not None:
            story_probs = self.story_model.predict_proba(feats)[:, 1]
        else:
            # Rule fallback when untrained
            story_probs = np.array(
                [0.80 if b.get("inside_bubble", False) else 0.30 for b in boxes],
                dtype=np.float32,
            )

        if self.is_fitted and self.narration_model is not None:
            # Adjust for balanced class_weight prior
            raw_nar = self.narration_model.predict_proba(feats)[:, 1]
            nar_probs = raw_nar * 0.65
        else:
            nar_probs = np.array(
                [0.70 if b.get("is_caption_box", False) else 0.05 for b in boxes],
                dtype=np.float32,
            )

        for idx, b in enumerate(boxes):
            txt = norm(str(b.get("text_raw") or b.get("pre_ocr_text") or ""))
            if not txt:
                b["story_prob"] = 0.0
                b["narration_prob"] = 0.0
                b["kind"] = "exclude"
                continue

            sp = float(story_probs[idx])
            np_prob = float(nar_probs[idx])

            # Hard guard against standalone page numbers in margins
            f_vec = feats[idx]
            is_digit_only = f_vec[13] > 0.5
            near_edge = f_vec[36] > 0.5
            if is_digit_only and (near_edge or len(txt) <= 3):
                sp = min(sp, 0.05)

            b["story_prob"] = round(sp, 4)
            b["narration_prob"] = round(np_prob, 4)
            if sp < self.story_prob_threshold:
                b["kind"] = "exclude"
            elif np_prob >= self.narration_prob_threshold:
                b["kind"] = "narration"
            else:
                b["kind"] = "speech"

        return boxes

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "enabled": self.enabled,
                "story_prob_threshold": self.story_prob_threshold,
                "narration_prob_threshold": self.narration_prob_threshold,
                "story_model": self.story_model,
                "narration_model": self.narration_model,
                "is_fitted": self.is_fitted,
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StoryTextClassifier":
        data = joblib.load(path)
        obj = cls(
            enabled=data["enabled"],
            story_prob_threshold=data["story_prob_threshold"],
            narration_prob_threshold=data["narration_prob_threshold"],
        )
        obj.story_model = data["story_model"]
        obj.narration_model = data["narration_model"]
        obj.is_fitted = data["is_fitted"]
        return obj
