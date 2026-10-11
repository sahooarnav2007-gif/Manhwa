"""M7 — Reading-order model (baseline right-to-left rules + learned pairwise ranker)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.learn.features import extract_order_pair_features


def baseline_rtl_sort_key(box: Dict[str, Any], page_w: int, page_h: int, use_panels: bool = True) -> tuple:
    """Manga right-to-left sort key: panel rank (if enabled), then vertical band, then right-to-left."""
    bx0, by0, bx1, by1 = box["bbox"]
    cx = 0.5 * (bx0 + bx1) / float(max(1, page_w))
    cy = 0.5 * (by0 + by1) / float(max(1, page_h))
    panel_rank = int(box.get("panel_rank", 1)) if use_panels else 1
    # Inside a panel, top-right comes before bottom-left
    score = cy - 0.36 * cx
    return (panel_rank, round(score, 4), -cx)


def compute_kendall_tau(ordered_boxes: List[Dict[str, Any]], weak_box_labels: Dict[str, Any]) -> Optional[float]:
    """Compute Kendall's tau rank correlation on story boxes matched to distinct gold lines."""
    gold_ranks = []
    for b in ordered_boxes:
        if b.get("kind") == "exclude":
            continue
        w = weak_box_labels.get(b["id"])
        if w and w.get("is_story") == 1 and w.get("gold_rank") is not None:
            gold_ranks.append(int(w["gold_rank"]))

    n = len(gold_ranks)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if gold_ranks[i] < gold_ranks[j]:
                concordant += 1
            elif gold_ranks[i] > gold_ranks[j]:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return float(concordant - discordant) / float(denom)


class ReadingOrderRanker:
    """Learned pairwise reading-order model with Copeland score ranking and panel constraints."""

    def __init__(
        self,
        learned_ranker: bool = True,
        panel_hard_constraint: bool = True,
        seed: int = 42,
    ) -> None:
        self.learned_ranker = learned_ranker
        self.panel_hard_constraint = panel_hard_constraint
        self.seed = seed
        self.pair_model: Optional[HistGradientBoostingClassifier] = None
        self.is_fitted = False

    def fit(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> "ReadingOrderRanker":
        if len(X_pairs) < 20 or len(np.unique(y_pairs)) < 2:
            self.is_fitted = False
            return self
        self.pair_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.06,
            min_samples_leaf=10,
            l2_regularization=1.5,
            random_state=self.seed,
        )
        self.pair_model.fit(X_pairs, y_pairs)
        self.is_fitted = True
        return self

    def order_page_boxes(
        self,
        boxes: List[Dict[str, Any]],
        page_w: int,
        page_h: int,
    ) -> List[Dict[str, Any]]:
        included = [b for b in boxes if b.get("kind") != "exclude"]
        excluded = [b for b in boxes if b.get("kind") == "exclude"]

        if len(included) <= 1 or not self.learned_ranker or not self.is_fitted or self.pair_model is None:
            included.sort(
                key=lambda b: baseline_rtl_sort_key(
                    b, page_w, page_h, use_panels=self.panel_hard_constraint
                )
            )
        else:
            n = len(included)
            pair_feats = []
            pair_indices = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    pair_feats.append(extract_order_pair_features(included[i], included[j], page_w, page_h))
                    pair_indices.append((i, j))

            X_p = np.stack(pair_feats, axis=0)
            probs = self.pair_model.predict_proba(X_p)[:, 1]

            # Copeland / expected win-count score per box
            win_scores = np.zeros(n, dtype=np.float64)
            for (i, j), p_ij in zip(pair_indices, probs):
                pa = int(included[i].get("panel_rank", 1))
                pb = int(included[j].get("panel_rank", 1))
                if self.panel_hard_constraint and pa != pb and 0.12 < p_ij < 0.88:
                    # Respect panel order unless pairwise ranker is >88% confident
                    p_eff = 1.0 if pa < pb else 0.0
                else:
                    p_eff = float(p_ij)
                win_scores[i] += p_eff
                win_scores[j] += 1.0 - p_eff

            order_idx = sorted(
                range(n),
                key=lambda idx: (
                    -win_scores[idx],
                    baseline_rtl_sort_key(included[idx], page_w, page_h, use_panels=self.panel_hard_constraint),
                ),
            )
            included = [included[idx] for idx in order_idx]

        for rank, b in enumerate(included, 1):
            b["order_rank"] = rank
        for b in excluded:
            b["order_rank"] = 999
        return included + excluded

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "learned_ranker": self.learned_ranker,
                "panel_hard_constraint": self.panel_hard_constraint,
                "pair_model": self.pair_model,
                "is_fitted": self.is_fitted,
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReadingOrderRanker":
        data = joblib.load(path)
        obj = cls(
            learned_ranker=data["learned_ranker"],
            panel_hard_constraint=data["panel_hard_constraint"],
        )
        obj.pair_model = data["pair_model"]
        obj.is_fitted = data["is_fitted"]
        return obj
