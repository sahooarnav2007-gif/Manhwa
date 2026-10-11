"""End-to-end modular manga transcription & speaker attribution pipeline with caching and grouped CV."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.loader import Sequence
from src.learn.features import extract_box_features, extract_order_pair_features
from src.learn.lora_train import ContextualTokenRefiner
from src.learn.order_rank import ReadingOrderRanker
from src.learn.speaker_pair import SpeakerPairModel, extract_speaker_pair_features
from src.learn.story_clf import StoryTextClassifier
from src.learn.weak_align import align_sequence_boxes
from src.ocr.ensemble import OCREnsemble
from src.pipeline.postprocess import postprocess_sequence_pages
from src.speakers.cluster import SequenceSpeakerClusterer
from src.speakers.cues import DiscourseCueAnalyzer
from src.speakers.tail import TailGeometryAttributor
from src.speakers.vlm_marks import VisualMarksAttributor
from src.vision.detect_chars import CharacterDetector
from src.vision.detect_text import TextDetector
from src.vision.embed import CharacterEmbedder
from src.vision.panels import PanelSegmenter
from src.vision.preprocess import Preprocessor


class MangaTranscriberPipeline:
    """Modular pipeline executing M1..M10 with disk caching for expensive perception stages."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))
        self.cache_dir = Path(cfg.get("cache_dir", "artifacts/cache"))

        prep_cfg = cfg.get("preprocess", {})
        self.preprocessor = Preprocessor(
            max_pixels=int(prep_cfg.get("max_pixels", 1200000)),
            upscale_factor=float(prep_cfg.get("upscale_factor", 2.0)),
        )

        det_cfg = cfg.get("detection", {})
        self.text_detector = TextDetector(
            mode=str(det_cfg.get("mode", "hybrid")),
            det_conf_threshold=float(det_cfg.get("det_conf_threshold", 0.35)),
            nms_iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
            min_box_area=int(det_cfg.get("min_box_area", 180)),
            max_box_area_ratio=float(det_cfg.get("max_box_area_ratio", 0.35)),
            enable_punct_candidates=bool(det_cfg.get("enable_punct_candidates", True)),
        )

        pan_cfg = cfg.get("panels", {})
        self.panel_segmenter = PanelSegmenter(
            gutter_white_thresh=int(pan_cfg.get("gutter_white_thresh", 242)),
            min_gutter_ratio=float(pan_cfg.get("min_gutter_ratio", 0.65)),
            min_panel_area_ratio=float(pan_cfg.get("min_panel_area_ratio", 0.03)),
        )

        char_cfg = cfg.get("characters", {})
        self.char_detector = CharacterDetector(
            det_conf_threshold=float(char_cfg.get("det_conf_threshold", 0.30)),
        )
        self.char_embedder = CharacterEmbedder(
            embedding_dim=int(char_cfg.get("embedding_dim", 64)),
        )

        ocr_cfg = cfg.get("ocr", {})
        self.ocr_ensemble = OCREnsemble(
            primary_engine=str(ocr_cfg.get("primary_engine", "ensemble")),
            fallback_engine=str(ocr_cfg.get("fallback_engine", "rapidocr")),
            upscale_factor=float(prep_cfg.get("upscale_factor", 2.0)),
            enable_punct_solver=bool(ocr_cfg.get("enable_punct_solver", True)),
            max_edit_distance_correction=float(ocr_cfg.get("max_edit_distance_correction", 0.20)),
        )

        s_cfg = cfg.get("story_classifier", {})
        self.story_clf = StoryTextClassifier(
            enabled=bool(s_cfg.get("enabled", True)),
            story_prob_threshold=float(s_cfg.get("story_prob_threshold", 0.46)),
            narration_prob_threshold=float(s_cfg.get("narration_prob_threshold", 0.52)),
            seed=self.seed,
        )

        o_cfg = cfg.get("reading_order", {})
        self.order_ranker = ReadingOrderRanker(
            learned_ranker=bool(o_cfg.get("learned_ranker", True)),
            panel_hard_constraint=bool(o_cfg.get("panel_hard_constraint", True)),
            seed=self.seed,
        )

        self.tail_attributor = TailGeometryAttributor()
        self.vlm_attributor = VisualMarksAttributor()
        self.cue_analyzer = DiscourseCueAnalyzer()
        self.speaker_pair_model = SpeakerPairModel(seed=self.seed)

        c_cfg = cfg.get("speaker_clustering", {})
        self.clusterer = SequenceSpeakerClusterer(
            mode=str(c_cfg.get("mode", "learned_agglomerative")),
            distance_threshold=float(c_cfg.get("distance_threshold", 0.52)),
            attach_threshold=float(c_cfg.get("attach_threshold", 0.42)),
            max_clusters=int(c_cfg.get("max_clusters", 6)),
            default_single_speaker=str(c_cfg.get("default_single_speaker", "char1")),
        )

        self.token_refiner: Optional[ContextualTokenRefiner] = (
            ContextualTokenRefiner() if bool(ocr_cfg.get("enable_learned_correction", False)) else None
        )

    def _cache_key_name(self) -> str:
        det_mode = self.cfg.get("detection", {}).get("mode", "hybrid")
        ocr_mode = self.cfg.get("ocr", {}).get("primary_engine", "ensemble")
        return f"{det_mode}_{ocr_mode}"

    def extract_or_load_perception(self, seq: Sequence, use_cache: bool = True) -> Dict[str, Any]:
        """Run or load cached M1–M5 + M8 spatial perception for a 3-page sequence."""
        cache_subdir = self.cache_dir / "perception" / self._cache_key_name() / seq.id
        cache_file = cache_subdir / "sequence_perception.json"
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        pages_data = []
        for idx, img_path in enumerate(seq.image_paths, 1):
            page_img = self.preprocessor.run(img_path, page_num=idx)
            panels = self.panel_segmenter.run(page_img)
            boxes = self.text_detector.run(page_img)
            self.panel_segmenter.assign_boxes_to_panels(boxes, panels)
            boxes = self.ocr_ensemble.run(page_img, boxes)
            chars = self.char_detector.run(page_img, panels, boxes)
            chars = self.char_embedder.run(page_img, chars)

            # Run M8 spatial attribution (tail geometry + visual set-of-marks grounding)
            self.tail_attributor.run(boxes, chars, page_img.width, page_img.height)
            self.vlm_attributor.run(page_img, boxes, chars)

            pages_data.append(
                {
                    "page": idx,
                    "width": page_img.width,
                    "height": page_img.height,
                    "panels": panels,
                    "boxes": boxes,
                    "characters": chars,
                }
            )

        payload = {"sequence_id": seq.id, "pages": pages_data}
        if use_cache:
            cache_subdir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            # Also save per-page box cache in artifacts/cache/boxes/<seq>/<page>.json per section 6 schema
            box_seq_dir = self.cache_dir / "boxes" / seq.id
            box_seq_dir.mkdir(parents=True, exist_ok=True)
            for p_data in pages_data:
                p_file = box_seq_dir / f"{p_data['page']}.json"
                p_file.write_text(
                    json.dumps({"page": p_data["page"], "boxes": p_data["boxes"]}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        return payload

    def build_training_matrices(
        self,
        sequences: List[Sequence],
        perceptions: Dict[str, Dict[str, Any]],
        weak_labels_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct training matrices for M6 (story/narration), M7 (order pairs), and M9 (speaker pairs)."""
        X_box_list, y_story_list, y_nar_list = [], [], []
        X_ord_list, y_ord_list = [], []
        X_spk_list, y_spk_list = [], []

        for seq in sequences:
            perc = perceptions[seq.id]
            w_seq = weak_labels_map.get(seq.id) or align_sequence_boxes(
                seq, [p["boxes"] for p in perc["pages"]]
            )
            w_boxes = w_seq["boxes"]

            char_by_id: Dict[str, Dict[str, Any]] = {}
            seq_story_boxes: List[Tuple[Dict[str, Any], str]] = []

            for p_data in perc["pages"]:
                pw, ph = int(p_data["width"]), int(p_data["height"])
                boxes = copy.deepcopy(p_data["boxes"])
                for ch in p_data["characters"]:
                    char_by_id[ch["char_id"]] = ch

                # Run discourse cues so features match inference time
                for b in boxes:
                    wb = w_boxes.get(b["id"], {})
                    b["kind"] = wb.get("kind", "exclude")
                self.cue_analyzer.run(boxes)

                for b in boxes:
                    wb = w_boxes.get(b["id"], {})
                    X_box_list.append(extract_box_features(b, boxes, pw, ph))
                    is_st = int(wb.get("is_story", 0))
                    is_nar = 1 if wb.get("kind") == "narration" else 0
                    y_story_list.append(is_st)
                    y_nar_list.append(is_nar)

                # Order pairs on story boxes of this page with distinct gold ranks
                matched_page = [
                    (b, int(w_boxes[b["id"]]["gold_rank"]))
                    for b in boxes
                    if w_boxes.get(b["id"], {}).get("is_story") == 1
                    and w_boxes[b["id"]].get("gold_rank") is not None
                ]
                for i in range(len(matched_page)):
                    for j in range(len(matched_page)):
                        if i == j:
                            continue
                        ba, ra = matched_page[i]
                        bb, rb = matched_page[j]
                        if ra == rb:
                            continue
                        X_ord_list.append(extract_order_pair_features(ba, bb, pw, ph))
                        y_ord_list.append(1 if ra < rb else 0)

                # Collect non-narration resolved speaker boxes for sequence-wide speaker pair training
                matched_page.sort(key=lambda x: x[1])
                for rank_idx, (b, _) in enumerate(matched_page, 1):
                    b["order_rank"] = rank_idx
                    g_spk = w_boxes[b["id"]].get("gold_speaker")
                    if g_spk and g_spk not in ("NARRATION", "UNKNOWN"):
                        seq_story_boxes.append((b, str(g_spk)))

            # Build speaker pairs across the sequence
            for i in range(len(seq_story_boxes)):
                for j in range(i + 1, len(seq_story_boxes)):
                    ba, sa = seq_story_boxes[i]
                    bb, sb = seq_story_boxes[j]
                    X_spk_list.append(extract_speaker_pair_features(ba, bb, char_by_id))
                    y_spk_list.append(1 if sa == sb else 0)

        X_box = np.stack(X_box_list, axis=0) if X_box_list else np.zeros((0, 37), dtype=np.float32)
        y_story = np.array(y_story_list, dtype=np.int32)
        y_nar = np.array(y_nar_list, dtype=np.int32)
        X_ord = np.stack(X_ord_list, axis=0) if X_ord_list else np.zeros((0, 14), dtype=np.float32)
        y_ord = np.array(y_ord_list, dtype=np.int32)
        X_spk = np.stack(X_spk_list, axis=0) if X_spk_list else np.zeros((0, 18), dtype=np.float32)
        y_spk = np.array(y_spk_list, dtype=np.int32)

        return X_box, y_story, y_nar, X_ord, y_ord, X_spk, y_spk

    def fit_on_sequences(
        self,
        train_sequences: List[Sequence],
        perceptions: Dict[str, Dict[str, Any]],
        weak_labels_map: Dict[str, Dict[str, Any]],
    ) -> "MangaTranscriberPipeline":
        """Fit M6, M7, M9, and optional T3 refiner on a list of training sequences."""
        X_box, y_story, y_nar, X_ord, y_ord, X_spk, y_spk = self.build_training_matrices(
            train_sequences, perceptions, weak_labels_map
        )
        if self.story_clf.enabled:
            self.story_clf.fit(X_box, y_story, y_nar)
        if self.order_ranker.learned_ranker:
            self.order_ranker.fit(X_ord, y_ord)
        if self.clusterer.mode == "learned_agglomerative":
            self.speaker_pair_model.fit(X_spk, y_spk)
        if self.token_refiner is not None:
            self.token_refiner.fit_from_sequences(train_sequences)
        return self

    def predict_sequence_from_perception(
        self,
        perception: Dict[str, Any],
    ) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, Any]]]]:
        """Run M6 -> M7 -> M8 cues -> M9 clustering -> M10 postprocessing on cached perception data."""
        page_boxes: List[List[Dict[str, Any]]] = []
        page_chars: List[List[Dict[str, Any]]] = []

        att_cfg = self.cfg.get("speaker_attribution", {})
        use_tail = bool(att_cfg.get("use_tail_geometry", True))
        use_vlm = bool(att_cfg.get("use_visual_marks", True))
        use_cues = bool(att_cfg.get("use_discourse_cues", True))

        for p_data in perception["pages"]:
            pw, ph = int(p_data["width"]), int(p_data["height"])
            boxes = copy.deepcopy(p_data["boxes"])
            chars = copy.deepcopy(p_data["characters"])

            if not use_tail:
                for b in boxes:
                    b["tail_candidates"] = []
            if not use_vlm:
                for b in boxes:
                    b["vlm_candidates"] = []

            # M6: Story-text and narration classification
            boxes = self.story_clf.predict_page_boxes(boxes, pw, ph)
            # M7: Reading-order ranking
            boxes = self.order_ranker.order_page_boxes(boxes, pw, ph)
            # M8: Discourse cue extraction & candidate fusion
            if use_cues:
                self.cue_analyzer.run(boxes)
            else:
                for b in boxes:
                    b["addressed_names"] = []
                    b["attrib"] = b.get("tail_candidates", [])[:2]

            page_boxes.append(boxes)
            page_chars.append(chars)

        # M9: Sequence-wide speaker clustering
        page_boxes = self.clusterer.run(
            page_boxes,
            page_chars,
            pair_model=self.speaker_pair_model if self.clusterer.mode == "learned_agglomerative" else None,
        )

        # M10: Post-processing into clean [{"speaker", "text"}] rows
        merge_bal = bool(self.cfg.get("reading_order", {}).get("merge_same_balloon", True))
        default_spk = str(self.cfg.get("speaker_clustering", {}).get("default_single_speaker", "char1"))
        pred_pages = postprocess_sequence_pages(
            page_boxes,
            merge_same_balloon=merge_bal,
            token_refiner=self.token_refiner,
            default_speaker=default_spk,
        )
        return pred_pages, page_boxes

    def run_sequence(self, seq: Sequence, use_cache: bool = True) -> List[List[Dict[str, str]]]:
        """Run full pipeline on a Sequence object."""
        perc = self.extract_or_load_perception(seq, use_cache=use_cache)
        pred_pages, _ = self.predict_sequence_from_perception(perc)
        return pred_pages
