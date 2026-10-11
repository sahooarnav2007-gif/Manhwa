"""M9 — Learned pairwise same-speaker probability model across a 3-page sequence."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def extract_speaker_pair_features(
    box_a: Dict[str, Any],
    box_b: Dict[str, Any],
    char_by_id: Dict[str, Dict[str, Any]],
    page_w: int = 900,
    page_h: int = 1300,
) -> np.ndarray:
    """Compute a 18-D pairwise feature vector for P(same_speaker | box_a, box_b)."""
    pa = int(box_a.get("page", 1))
    pb = int(box_b.get("page", 1))
    page_gap = float(abs(pa - pb))
    same_page = float(pa == pb)

    same_panel = float(same_page > 0.5 and box_a.get("panel_id") == box_b.get("panel_id"))
    same_balloon = float(
        same_page > 0.5
        and box_a.get("balloon_id") is not None
        and box_a.get("balloon_id") == box_b.get("balloon_id")
    )

    ra = int(box_a.get("order_rank", 1))
    rb = int(box_b.get("order_rank", 1))
    rank_gap = float(abs(ra - rb)) if same_page > 0.5 else float(abs(ra - rb) + 5.0 * page_gap)
    is_consecutive = float(same_page > 0.5 and abs(ra - rb) == 1)
    is_step2_same_panel = float(same_panel > 0.5 and abs(ra - rb) == 2)

    ax0, ay0, ax1, ay1 = box_a["bbox"]
    bx0, by0, bx1, by1 = box_b["bbox"]
    acx = 0.5 * (ax0 + ax1) / float(max(1, page_w))
    acy = 0.5 * (ay0 + ay1) / float(max(1, page_h))
    bcx = 0.5 * (bx0 + bx1) / float(max(1, page_w))
    bcy = 0.5 * (by0 + by1) / float(max(1, page_h))

    dx = abs(acx - bcx)
    dy = abs(acy - bcy)
    dist = math.hypot(dx, dy)

    # Attribution features
    att_a = box_a.get("attrib", [])
    att_b = box_b.get("attrib", [])
    cid_a = att_a[0]["char_id"] if att_a else None
    cid_b = att_b[0]["char_id"] if att_b else None
    conf_a = float(att_a[0]["score"]) if att_a else 0.0
    conf_b = float(att_b[0]["score"]) if att_b else 0.0

    same_char_id = float(cid_a is not None and cid_a == cid_b)
    diff_char_id_same_page = float(same_page > 0.5 and cid_a is not None and cid_b is not None and cid_a != cid_b)

    emb_cos = 0.0
    if cid_a and cid_b and cid_a in char_by_id and cid_b in char_by_id:
        ea = np.array(char_by_id[cid_a].get("embedding", []), dtype=np.float32)
        eb = np.array(char_by_id[cid_b].get("embedding", []), dtype=np.float32)
        if ea.size > 0 and ea.shape == eb.shape:
            emb_cos = float(np.dot(ea, eb))

    # Discourse cues
    voc_a = set(box_a.get("addressed_names", []))
    voc_b = set(box_b.get("addressed_names", []))
    same_vocative = float(len(voc_a & voc_b) > 0)
    adj_vocative_turn = float(is_consecutive > 0.5 and (len(voc_a) > 0 or len(voc_b) > 0) and not same_vocative)

    qa_turn = float(
        is_consecutive > 0.5
        and dx > 0.16
        and (bool(box_a.get("is_question", False)) or bool(box_b.get("is_question", False)))
    )
    monologue_chain = float(
        same_balloon > 0.5
        or (
            is_consecutive > 0.5
            and same_panel > 0.5
            and dx < 0.14
            and dy < 0.22
            and not qa_turn
        )
        or (
            is_consecutive > 0.5
            and (bool(box_a.get("ends_with_ellipsis", False)) or bool(box_b.get("starts_with_ellipsis", False)))
            and dx < 0.20
        )
    )

    both_first_person = float(bool(box_a.get("has_first_person", False)) and bool(box_b.get("has_first_person", False)))

    return np.array(
        [
            page_gap,
            same_page,
            same_panel,
            same_balloon,
            rank_gap,
            is_consecutive,
            is_step2_same_panel,
            dx,
            dy,
            dist,
            same_char_id,
            diff_char_id_same_page,
            emb_cos,
            conf_a * conf_b,
            same_vocative,
            adj_vocative_turn,
            qa_turn,
            monologue_chain + 0.25 * both_first_person,
        ],
        dtype=np.float32,
    )


class SpeakerPairModel:
    """Learned gradient-boosted pairwise same-speaker probability estimator."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.is_fitted = False

    def fit(self, X_pairs: np.ndarray, y_same: np.ndarray) -> "SpeakerPairModel":
        if len(X_pairs) < 20 or len(np.unique(y_same)) < 2:
            self.is_fitted = False
            return self
        self.model = HistGradientBoostingClassifier(
            max_iter=110,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=self.seed,
        )
        self.model.fit(X_pairs, y_same)
        self.is_fitted = True
        return self

    def predict_similarity_matrix(
        self,
        boxes: List[Dict[str, Any]],
        char_by_id: Dict[str, Dict[str, Any]],
        page_w: int = 900,
        page_h: int = 1300,
    ) -> np.ndarray:
        """Compute symmetric N x N same-speaker probability matrix for non-narration story boxes."""
        n = len(boxes)
        sim = np.eye(n, dtype=np.float32)
        if n <= 1:
            return sim

        feats = []
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                f = extract_speaker_pair_features(boxes[i], boxes[j], char_by_id, page_w, page_h)
                feats.append(f)
                pairs.append((i, j, f))

        X = np.stack(feats, axis=0)
        if self.is_fitted and self.model is not None:
            probs = self.model.predict_proba(X)[:, 1]
        else:
            # Heuristic probability formula when untrained
            probs = []
            for _, _, f in pairs:
                # f indices: 3=same_balloon, 10=same_char_id, 12=emb_cos, 14=same_vocative, 16=qa_turn, 17=monologue
                p = (
                    0.32
                    + 0.45 * f[3]
                    + 0.28 * f[10]
                    + 0.22 * max(0.0, f[12] - 0.5)
                    + 0.20 * f[14]
                    + 0.22 * f[17]
                    - 0.25 * f[16]
                    - 0.18 * f[11]
                )
                probs.append(float(np.clip(p, 0.05, 0.95)))
            probs = np.array(probs, dtype=np.float32)

        for (i, j, f), p in zip(pairs, probs):
            # Enforce same-balloon hard link
            if f[3] > 0.5:
                p = max(float(p), 0.88)
            sim[i, j] = sim[j, i] = float(p)

        return sim

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "is_fitted": self.is_fitted, "seed": self.seed}, p)

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerPairModel":
        data = joblib.load(path)
        obj = cls(seed=data.get("seed", 42))
        obj.model = data["model"]
        obj.is_fitted = data["is_fitted"]
        return obj
