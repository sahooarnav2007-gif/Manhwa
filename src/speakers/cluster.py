"""M9 — Sequence-wide speaker clustering with metric-tuned agglomerative threshold and NARRATION handling."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.learn.speaker_pair import SpeakerPairModel


class SequenceSpeakerClusterer:
    """Assigns consistent char1..charN and NARRATION labels across a 3-page sequence."""

    def __init__(
        self,
        mode: str = "learned_agglomerative",
        distance_threshold: float = 0.52,
        attach_threshold: float = 0.42,
        max_clusters: int = 6,
        default_single_speaker: str = "char1",
    ) -> None:
        self.mode = mode
        self.distance_threshold = distance_threshold
        self.attach_threshold = attach_threshold
        self.max_clusters = max_clusters
        self.default_single_speaker = default_single_speaker

    def run(
        self,
        page_boxes: List[List[Dict[str, Any]]],
        page_characters: List[List[Dict[str, Any]]],
        pair_model: Optional[SpeakerPairModel] = None,
    ) -> List[List[Dict[str, Any]]]:
        # Flatten all characters across the 3 pages
        char_by_id: Dict[str, Dict[str, Any]] = {}
        for chars in page_characters:
            for ch in chars:
                char_by_id[ch["char_id"]] = ch

        # Collect included story boxes in reading order across pages 1..3
        dialogue_boxes: List[Dict[str, Any]] = []
        for p_boxes in page_boxes:
            for b in p_boxes:
                if b.get("kind") == "exclude":
                    b["cluster"] = None
                elif b.get("kind") == "narration":
                    b["cluster"] = "NARRATION"
                else:
                    dialogue_boxes.append(b)

        if not dialogue_boxes:
            return page_boxes

        if self.mode == "single" or len(dialogue_boxes) == 1:
            for b in dialogue_boxes:
                b["cluster"] = self.default_single_speaker
            return page_boxes

        if self.mode == "heuristic":
            # Heuristic attribution + appearance clustering without trained pairwise GBM
            untrained_model = SpeakerPairModel()
            sim = untrained_model.predict_similarity_matrix(dialogue_boxes, char_by_id)
        else:
            model = pair_model if pair_model is not None else SpeakerPairModel()
            sim = model.predict_similarity_matrix(dialogue_boxes, char_by_id)

        n = len(dialogue_boxes)
        dist_mat = np.clip(1.0 - sim, 0.0, 1.0)
        np.fill_diagonal(dist_mat, 0.0)

        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="average")
        raw_labels = fcluster(Z, t=self.distance_threshold, criterion="distance")

        # Enforce max_clusters cap if exceeded
        if len(np.unique(raw_labels)) > self.max_clusters:
            raw_labels = fcluster(Z, t=self.max_clusters, criterion="maxclust")

        # Singleton attachment policy: attach singleton clusters to the closest multi-box cluster if p >= attach_threshold
        counts = {int(lbl): int(np.sum(raw_labels == lbl)) for lbl in np.unique(raw_labels)}
        multi_clusters = [lbl for lbl, cnt in counts.items() if cnt >= 2]
        if multi_clusters:
            for idx in range(n):
                lbl = int(raw_labels[idx])
                if counts[lbl] == 1:
                    best_c = None
                    best_p = -1.0
                    for mc in multi_clusters:
                        members = [j for j in range(n) if int(raw_labels[j]) == mc]
                        mean_p = float(np.mean([sim[idx, j] for j in members]))
                        if mean_p > best_p:
                            best_p = mean_p
                            best_c = mc
                    if best_c is not None and best_p >= self.attach_threshold:
                        raw_labels[idx] = best_c

        # Name clusters char1..charN in order of first appearance across the 3 pages
        cluster_map: Dict[int, str] = {}
        next_char_num = 1
        for idx, b in enumerate(dialogue_boxes):
            c_id = int(raw_labels[idx])
            if c_id not in cluster_map:
                cluster_map[c_id] = f"char{next_char_num}"
                next_char_num += 1
            b["cluster"] = cluster_map[c_id]

        return page_boxes
