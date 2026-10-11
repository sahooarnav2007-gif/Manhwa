"""Master CLI for caching, weak-labelling, 5-fold CV training, tuning, ablations, and test prediction."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from src.data.loader import (
    Sequence,
    load_config,
    load_score_module,
    load_sequences,
    set_global_seed,
    update_run_manifest,
)
from src.data.splits import get_fold_split, load_folds
from src.eval.error_analysis import run_error_analysis
from src.eval.proxy_score import proxy_evaluate
from src.eval.run_official import run_official_evaluation
from src.learn.order_rank import compute_kendall_tau
from src.learn.weak_align import align_sequence_boxes, save_weak_labels
from src.pipeline.run_sequence import MangaTranscriberPipeline
from src.pipeline.write_submission import write_and_validate_submission


def run_cache(split: str = "development", force_rebuild_from_raw: bool = False) -> Dict[str, Any]:
    """Run and cache detector + OCR + character + attribution perception for all sequences."""
    cfg_t0 = load_config("configs/tier0.yaml")
    cfg_t2 = load_config("configs/tier2.yaml")
    set_global_seed(cfg_t2["seed"])
    seqs = load_sequences(cfg_t2["dataset_dir"], split=split)
    pipe_t2 = MangaTranscriberPipeline(cfg_t2)
    pipe_t0 = MangaTranscriberPipeline(cfg_t0)

    t0_start = time.perf_counter()
    for idx, seq in enumerate(seqs, 1):
        if force_rebuild_from_raw:
            # Rebuild perception from cached raw_page_ocr
            for mode_name in (pipe_t2._cache_key_name(), pipe_t0._cache_key_name()):
                p_file = pipe_t2.cache_dir / "perception" / mode_name / seq.id / "sequence_perception.json"
                if p_file.exists():
                    p_file.unlink()
        pipe_t2.extract_or_load_perception(seq, use_cache=True)
        pipe_t0.extract_or_load_perception(seq, use_cache=True)
        if idx % 20 == 0 or idx == len(seqs):
            elapsed = time.perf_counter() - t0_start
            print(f"[{split}] Cached {idx}/{len(seqs)} sequences ({elapsed:.1f}s)", flush=True)

    total_time = time.perf_counter() - t0_start
    pages_count = len(seqs) * 3
    sec_per_page = total_time / float(max(1, pages_count))
    update_run_manifest(
        extra_data={
            f"cache_{split}": {
                "sequences": len(seqs),
                "pages": pages_count,
                "wall_clock_sec": round(total_time, 2),
                "sec_per_page": round(sec_per_page, 3),
            }
        }
    )
    return {"sequences": len(seqs), "sec_per_page": sec_per_page}


def run_weaklabels() -> Dict[str, Any]:
    """Align cached OCR boxes on dev pages to gold transcripts and save weak labels."""
    cfg = load_config("configs/tier2.yaml")
    seqs = load_sequences(cfg["dataset_dir"], split="development")
    pipe = MangaTranscriberPipeline(cfg)

    total_gold_tokens = 0
    covered_gold_tokens = 0.0
    total_boxes = 0
    matched_boxes = 0
    narration_boxes = 0

    for seq in seqs:
        perc = pipe.extract_or_load_perception(seq, use_cache=True)
        page_boxes = [p["boxes"] for p in perc["pages"]]
        weak = align_sequence_boxes(seq, page_boxes, sim_threshold=0.60)
        save_weak_labels(weak, cfg["weak_labels_dir"])

        total_gold_tokens += int(weak["total_gold_tokens"])
        covered_gold_tokens += float(weak["covered_gold_tokens"])
        for b_info in weak["boxes"].values():
            total_boxes += 1
            if b_info["is_story"] == 1:
                matched_boxes += 1
                if b_info["kind"] == "narration":
                    narration_boxes += 1

    global_recall = covered_gold_tokens / float(max(1, total_gold_tokens))
    match_rate = matched_boxes / float(max(1, total_boxes))
    stats = {
        "detector_token_recall": round(global_recall, 4),
        "weak_label_box_match_rate": round(match_rate, 4),
        "total_dev_boxes": total_boxes,
        "matched_story_boxes": matched_boxes,
        "narration_boxes": narration_boxes,
        "unmatched_exclude_boxes": total_boxes - matched_boxes,
        "total_gold_tokens": total_gold_tokens,
        "covered_gold_tokens": round(covered_gold_tokens, 2),
    }
    update_run_manifest(extra_data={"weak_labels": stats})
    print(json.dumps(stats, indent=2))
    return stats


def _load_dev_perceptions_and_weak(cfg: Dict[str, Any]) -> Tuple[List[Sequence], Dict[str, Any], Dict[str, Any]]:
    seqs = load_sequences(cfg["dataset_dir"], split="development")
    pipe = MangaTranscriberPipeline(cfg)
    perceptions = {}
    weak_map = {}
    weak_dir = Path(cfg["weak_labels_dir"])
    for seq in seqs:
        perc = pipe.extract_or_load_perception(seq, use_cache=True)
        perceptions[seq.id] = perc
        w_file = weak_dir / f"{seq.id}.json"
        if w_file.exists():
            weak_map[seq.id] = json.loads(w_file.read_text(encoding="utf-8"))
        else:
            w_data = align_sequence_boxes(seq, [p["boxes"] for p in perc["pages"]])
            save_weak_labels(w_data, weak_dir)
            weak_map[seq.id] = w_data
    return seqs, perceptions, weak_map


def run_cv_for_config(
    cfg: Dict[str, Any],
    seqs: List[Sequence],
    perceptions: Dict[str, Any],
    weak_map: Dict[str, Any],
    folds: Dict[str, int],
    save_fold_models: bool = False,
) -> Tuple[Dict[str, List[List[Dict[str, str]]]], Dict[str, List[List[Dict[str, Any]]]]]:
    """Run 5-fold cross-validation for a given configuration and return out-of-fold predictions and box states."""
    oof_preds: Dict[str, List[List[Dict[str, str]]]] = {}
    oof_boxes: Dict[str, List[List[Dict[str, Any]]]] = {}
    models_dir = Path(cfg.get("models_dir", "artifacts/models"))

    for val_fold in range(5):
        train_seqs, val_seqs = get_fold_split(seqs, folds, val_fold)
        pipe = MangaTranscriberPipeline(cfg)
        f_dir = models_dir / f"fold_{val_fold}"

        if not save_fold_models and (f_dir / "story_clf.joblib").exists():
            from src.learn.lora_train import ContextualTokenRefiner
            from src.learn.order_rank import ReadingOrderRanker
            from src.learn.speaker_pair import SpeakerPairModel
            from src.learn.story_clf import StoryTextClassifier

            if pipe.story_clf.enabled:
                loaded_s = StoryTextClassifier.load(f_dir / "story_clf.joblib")
                loaded_s.story_prob_threshold = pipe.story_clf.story_prob_threshold
                loaded_s.narration_prob_threshold = pipe.story_clf.narration_prob_threshold
                pipe.story_clf = loaded_s
            if pipe.order_ranker.learned_ranker:
                loaded_o = ReadingOrderRanker.load(f_dir / "order_ranker.joblib")
                loaded_o.panel_hard_constraint = pipe.order_ranker.panel_hard_constraint
                pipe.order_ranker = loaded_o
            if pipe.clusterer.mode == "learned_agglomerative":
                pipe.speaker_pair_model = SpeakerPairModel.load(f_dir / "speaker_pair.joblib")
            if pipe.token_refiner is not None and (f_dir / "token_refiner.joblib").exists():
                pipe.token_refiner = ContextualTokenRefiner.load(f_dir / "token_refiner.joblib")
        else:
            if (
                pipe.story_clf.enabled
                or pipe.order_ranker.learned_ranker
                or pipe.clusterer.mode == "learned_agglomerative"
                or pipe.token_refiner is not None
            ):
                pipe.fit_on_sequences(train_seqs, perceptions, weak_map)

        if save_fold_models:
            f_dir.mkdir(parents=True, exist_ok=True)
            pipe.story_clf.save(f_dir / "story_clf.joblib")
            pipe.order_ranker.save(f_dir / "order_ranker.joblib")
            pipe.speaker_pair_model.save(f_dir / "speaker_pair.joblib")
            if pipe.token_refiner is not None:
                pipe.token_refiner.save(f_dir / "token_refiner.joblib")

        for v_seq in val_seqs:
            pred_pages, p_boxes = pipe.predict_sequence_from_perception(perceptions[v_seq.id])
            oof_preds[v_seq.id] = pred_pages
            oof_boxes[v_seq.id] = p_boxes

    return oof_preds, oof_boxes


def run_train() -> None:
    """Train 5-fold CV models and full-development models for test inference."""
    cfg = load_config("configs/tier2.yaml")
    set_global_seed(cfg["seed"])
    folds = load_folds(cfg["folds_file"])
    seqs, perceptions, weak_map = _load_dev_perceptions_and_weak(cfg)

    t0 = time.perf_counter()
    cfg_all_models = copy.deepcopy(cfg)
    cfg_all_models["ocr"]["enable_learned_correction"] = True
    oof_preds, _ = run_cv_for_config(cfg_all_models, seqs, perceptions, weak_map, folds, save_fold_models=True)

    # Also fit final models on all 80 development sequences for test prediction
    full_pipe = MangaTranscriberPipeline(cfg)
    full_pipe.fit_on_sequences(seqs, perceptions, weak_map)
    full_dir = Path(cfg.get("models_dir", "artifacts/models")) / "full_dev"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_pipe.story_clf.save(full_dir / "story_clf.joblib")
    full_pipe.order_ranker.save(full_dir / "order_ranker.joblib")
    full_pipe.speaker_pair_model.save(full_dir / "speaker_pair.joblib")

    cfg_t3 = load_config("configs/tier3.yaml")
    pipe_t3 = MangaTranscriberPipeline(cfg_t3)
    pipe_t3.fit_on_sequences(seqs, perceptions, weak_map)
    if pipe_t3.token_refiner is not None:
        pipe_t3.token_refiner.save(full_dir / "token_refiner.joblib")

    write_and_validate_submission(
        oof_preds,
        "artifacts/dev_cv_predictions.jsonl",
        expected_sequence_ids=[s.id for s in seqs],
        dataset_dir=cfg["dataset_dir"],
    )
    elapsed = time.perf_counter() - t0
    update_run_manifest(extra_data={"train": {"wall_clock_sec": round(elapsed, 2)}})
    print(f"Trained 5-fold CV and full-dev models in {elapsed:.2f}s.")


def run_tune() -> Dict[str, Any]:
    """Fast grid-search of operating thresholds on 5-fold dev CV using proxy scorer, confirmed by official score.py."""
    base_cfg = load_config("configs/tier2.yaml")
    set_global_seed(base_cfg["seed"])
    folds = load_folds(base_cfg["folds_file"])
    seqs, perceptions, weak_map = _load_dev_perceptions_and_weak(base_cfg)

    score_mod = load_score_module(base_cfg["dataset_dir"])
    refs = score_mod.load_jsonl(base_cfg["dev_labels_file"], reference=True, pages=3)

    # Precompute out-of-fold scored boxes (with story_prob, narration_prob, order_rank, and cues) once per sequence
    models_dir = Path(base_cfg.get("models_dir", "artifacts/models"))
    precomputed: Dict[str, Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], Any]] = {}

    for val_fold in range(5):
        train_seqs, val_seqs = get_fold_split(seqs, folds, val_fold)
        p = MangaTranscriberPipeline(copy.deepcopy(base_cfg))
        f_dir = models_dir / f"fold_{val_fold}"
        if (f_dir / "story_clf.joblib").exists():
            from src.learn.order_rank import ReadingOrderRanker
            from src.learn.speaker_pair import SpeakerPairModel
            from src.learn.story_clf import StoryTextClassifier

            p.story_clf = StoryTextClassifier.load(f_dir / "story_clf.joblib")
            p.order_ranker = ReadingOrderRanker.load(f_dir / "order_ranker.joblib")
            p.speaker_pair_model = SpeakerPairModel.load(f_dir / "speaker_pair.joblib")
        else:
            p.fit_on_sequences(train_seqs, perceptions, weak_map)

        # Score all boxes with low story_prob_threshold=0.01 so order_ranker orders all candidate boxes once
        p.story_clf.story_prob_threshold = 0.01
        for v_seq in val_seqs:
            perc = perceptions[v_seq.id]
            p_boxes = []
            p_chars = []
            for p_data in perc["pages"]:
                pw, ph = int(p_data["width"]), int(p_data["height"])
                b_copy = copy.deepcopy(p_data["boxes"])
                c_copy = copy.deepcopy(p_data["characters"])
                b_copy = p.story_clf.predict_page_boxes(b_copy, pw, ph)
                b_copy = p.order_ranker.order_page_boxes(b_copy, pw, ph)
                p.cue_analyzer.run(b_copy)
                p_boxes.append(b_copy)
                p_chars.append(c_copy)
            precomputed[v_seq.id] = (p_boxes, p_chars, p.speaker_pair_model)

    from src.pipeline.postprocess import postprocess_sequence_pages
    from src.speakers.cluster import SequenceSpeakerClusterer

    story_thrs = [0.38, 0.44, 0.50]
    nar_thrs = [0.48, 0.55]
    dist_thrs = [0.46, 0.54, 0.62]
    max_clusts = [4, 5]

    candidates = []
    for st in story_thrs:
        for nt in nar_thrs:
            for dt in dist_thrs:
                for mc in max_clusts:
                    clusterer = SequenceSpeakerClusterer(
                        mode="learned_agglomerative",
                        distance_threshold=dt,
                        attach_threshold=max(0.30, dt - 0.10),
                        max_clusters=mc,
                    )
                    oof_preds = {}
                    for sid, (base_boxes, p_chars, spk_model) in precomputed.items():
                        seq_boxes = copy.deepcopy(base_boxes)
                        for page_b in seq_boxes:
                            for b in page_b:
                                sp = float(b.get("story_prob", 0.0))
                                np_p = float(b.get("narration_prob", 0.0))
                                if sp < st:
                                    b["kind"] = "exclude"
                                elif np_p >= nt or b.get("kind") == "narration":
                                    b["kind"] = "narration"
                                else:
                                    b["kind"] = "speech"
                        seq_boxes = clusterer.run(seq_boxes, p_chars, pair_model=spk_model)
                        oof_preds[sid] = postprocess_sequence_pages(seq_boxes, merge_same_balloon=False)

                    prx = proxy_evaluate(refs, oof_preds, pages=3)["macro"]
                    comb = 0.5 * prx["text_order_score"] + 0.5 * prx["balanced_joint_f1"]
                    candidates.append(
                        (
                            comb,
                            st,
                            nt,
                            dt,
                            mc,
                            prx["text_order_score"],
                            prx["balanced_joint_f1"],
                            oof_preds,
                        )
                    )

    candidates.sort(key=lambda x: x[0], reverse=True)
    top3 = candidates[:3]
    best_official_comb = -1.0
    best_cfg = copy.deepcopy(base_cfg)
    best_preds = None

    print("Confirming top-3 proxy configurations with official dataset/score.py:")
    for rank, (_, st, nt, dt, mc, prx_t, prx_b, preds) in enumerate(top3, 1):
        off = score_mod.evaluate(refs, preds, pages=3)["macro"]
        off_comb = 0.5 * off["text_order_score"] + 0.5 * off["balanced_joint_f1"]
        print(
            f"  #{rank}: story_thr={st}, nar_thr={nt}, tau={dt}, max_c={mc} -> "
            f"official text_order={off['text_order_score']:.4f}, balanced_joint_f1={off['balanced_joint_f1']:.4f}"
        )
        if off_comb > best_official_comb:
            best_official_comb = off_comb
            best_preds = preds
            best_cfg["story_classifier"]["story_prob_threshold"] = st
            best_cfg["story_classifier"]["narration_prob_threshold"] = nt
            best_cfg["speaker_clustering"]["distance_threshold"] = dt
            best_cfg["speaker_clustering"]["attach_threshold"] = round(max(0.30, dt - 0.10), 2)
            best_cfg["speaker_clustering"]["max_clusters"] = mc

    # Freeze chosen configuration in configs/final.yaml and update tier2.yaml
    final_path = Path("configs/final.yaml")
    final_path.write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")
    Path("configs/tier2.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")

    if best_preds is not None:
        write_and_validate_submission(
            best_preds,
            "artifacts/dev_cv_predictions.jsonl",
            expected_sequence_ids=[s.id for s in seqs],
            dataset_dir=base_cfg["dataset_dir"],
        )

    return best_cfg


def _compute_diagnostics(
    seqs: List[Sequence],
    oof_preds: Dict[str, List[List[Dict[str, str]]]],
    oof_boxes: Dict[str, List[List[Dict[str, Any]]]],
    weak_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute ordering Kendall tau, gold-empty page precision, and speaker merge/split statistics."""
    taus = []
    empty_pages_total = 0
    empty_pages_correct = 0
    gold_spk_counts = []
    pred_spk_counts = []
    merge_seqs = 0
    split_seqs = 0

    for seq in seqs:
        w_boxes = weak_map[seq.id]["boxes"]
        p_boxes = oof_boxes[seq.id]
        pred_pages = oof_preds[seq.id]

        for p_idx in range(3):
            tau = compute_kendall_tau(p_boxes[p_idx], w_boxes)
            if tau is not None:
                taus.append(tau)
            if seq.gold_pages and len(seq.gold_pages[p_idx]) == 0:
                empty_pages_total += 1
                if len(pred_pages[p_idx]) == 0:
                    empty_pages_correct += 1

        if seq.gold_pages:
            g_spks = {
                r["speaker"]
                for p in seq.gold_pages
                for r in p
                if r["speaker"] is not None and r["speaker"] != "NARRATION"
            }
            p_spks = {
                r["speaker"]
                for p in pred_pages
                for r in p
                if r["speaker"] is not None and r["speaker"] != "NARRATION"
            }
            gold_spk_counts.append(len(g_spks))
            pred_spk_counts.append(len(p_spks))
            if len(p_spks) < len(g_spks):
                merge_seqs += 1
            elif len(p_spks) > len(g_spks):
                split_seqs += 1

    return {
        "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else 0.0,
        "empty_page_precision": round(empty_pages_correct / float(max(1, empty_pages_total)), 4),
        "empty_pages_correct": empty_pages_correct,
        "empty_pages_total": empty_pages_total,
        "mean_gold_speakers": round(float(np.mean(gold_spk_counts)), 2),
        "mean_pred_speakers": round(float(np.mean(pred_spk_counts)), 2),
        "under_clustered_merge_sequences": merge_seqs,
        "over_clustered_split_sequences": split_seqs,
    }


def run_eval() -> Dict[str, Any]:
    """Run full 5-fold CV ablation table using official score.py and write reports/ablations.csv and reports/results.md."""
    cfg_t0 = load_config("configs/tier0.yaml")
    cfg_t2 = load_config("configs/final.yaml" if Path("configs/final.yaml").exists() else "configs/tier2.yaml")
    set_global_seed(cfg_t2["seed"])
    folds = load_folds(cfg_t2["folds_file"])

    seqs, perc_hybrid, weak_map = _load_dev_perceptions_and_weak(cfg_t2)
    pipe_t0 = MangaTranscriberPipeline(cfg_t0)
    perc_opencv = {seq.id: pipe_t0.extract_or_load_perception(seq, use_cache=True) for seq in seqs}

    # Define the 7 cumulative ablation configurations
    # 1. T0 baseline
    ablation_specs = [
        ("T0 baseline", cfg_t0, perc_opencv),
    ]

    # 2. + pretrained detector + better OCR (unfiltered, heuristic RTL order, 1 speaker)
    cfg_s2 = copy.deepcopy(cfg_t2)
    cfg_s2["story_classifier"]["enabled"] = False
    cfg_s2["reading_order"]["learned_ranker"] = False
    cfg_s2["reading_order"]["panel_hard_constraint"] = False
    cfg_s2["speaker_clustering"]["mode"] = "single"
    ablation_specs.append(("+ pretrained detector + better OCR", cfg_s2, perc_hybrid))

    # 3. + learned story-text filter
    cfg_s3 = copy.deepcopy(cfg_s2)
    cfg_s3["story_classifier"]["enabled"] = True
    ablation_specs.append(("+ learned story-text filter", cfg_s3, perc_hybrid))

    # 4. + learned reading order
    cfg_s4 = copy.deepcopy(cfg_s3)
    cfg_s4["reading_order"]["learned_ranker"] = True
    cfg_s4["reading_order"]["panel_hard_constraint"] = True
    ablation_specs.append(("+ learned reading order", cfg_s4, perc_hybrid))

    # 5. + speaker heuristics
    cfg_s5 = copy.deepcopy(cfg_s4)
    cfg_s5["speaker_clustering"]["mode"] = "heuristic"
    ablation_specs.append(("+ speaker heuristics", cfg_s5, perc_hybrid))

    # 6. + learned pairwise clustering (T2)
    cfg_s6 = copy.deepcopy(cfg_t2)
    cfg_s6["speaker_clustering"]["mode"] = "learned_agglomerative"
    ablation_specs.append(("+ learned pairwise clustering (T2)", cfg_s6, perc_hybrid))

    # 7. + LoRA VLM / contextual refiner (T3, optional)
    cfg_s7 = copy.deepcopy(cfg_s6)
    cfg_s7["ocr"]["enable_learned_correction"] = True
    ablation_specs.append(("+ LoRA VLM (T3, optional)", cfg_s7, perc_hybrid))

    ablation_rows = []
    best_system_report = None
    best_oof_boxes = None
    best_oof_preds = None
    best_score_pair = -1.0

    for sys_name, sys_cfg, sys_perc in ablation_specs:
        oof_preds, oof_boxes = run_cv_for_config(sys_cfg, seqs, sys_perc, weak_map, folds, save_fold_models=False)
        pred_file = Path("artifacts") / f"preds_{sys_name.split()[0].strip('+')}.jsonl"
        write_and_validate_submission(
            oof_preds,
            pred_file,
            expected_sequence_ids=[s.id for s in seqs],
            dataset_dir=sys_cfg["dataset_dir"],
        )
        rep = run_official_evaluation(
            references_path=sys_cfg["dev_labels_file"],
            predictions_path=pred_file,
            folds_path=sys_cfg["folds_file"],
        )
        diag = _compute_diagnostics(seqs, oof_preds, oof_boxes, weak_map)
        macro = rep["macro"]
        f_sum = rep["fold_summary"]

        row = {
            "system": sys_name,
            "text_order_score_mean": round(f_sum["text_order_score"]["mean"], 4),
            "text_order_score_std": round(f_sum["text_order_score"]["std"], 4),
            "balanced_joint_f1_mean": round(f_sum["balanced_joint_f1"]["mean"], 4),
            "balanced_joint_f1_std": round(f_sum["balanced_joint_f1"]["std"], 4),
            "joint_f1_mean": round(f_sum["joint_f1"]["mean"], 4),
            "joint_f1_std": round(f_sum["joint_f1"]["std"], 4),
            "matched_token_coverage_mean": round(f_sum["matched_token_coverage"]["mean"], 4),
            "matched_token_coverage_std": round(f_sum["matched_token_coverage"]["std"], 4),
            "speaker_accuracy_on_matched_mean": round(f_sum["speaker_accuracy_on_matched"]["mean"], 4),
            "speaker_accuracy_on_matched_std": round(f_sum["speaker_accuracy_on_matched"]["std"], 4),
            "kendall_tau": diag["kendall_tau_mean"],
            "empty_page_precision": diag["empty_page_precision"],
            "mean_pred_speakers": diag["mean_pred_speakers"],
            "merge_sequences": diag["under_clustered_merge_sequences"],
            "split_sequences": diag["over_clustered_split_sequences"],
        }
        ablation_rows.append(row)
        print(
            f"[Ablation] {sys_name:36s} | text_order={row['text_order_score_mean']:.3f}±{row['text_order_score_std']:.3f} "
            f"| balanced_joint_f1={row['balanced_joint_f1_mean']:.3f}±{row['balanced_joint_f1_std']:.3f} "
            f"| joint_f1={row['joint_f1_mean']:.3f} | cov={row['matched_token_coverage_mean']:.3f}"
        )

        comb = 0.5 * macro["text_order_score"] + 0.5 * macro["balanced_joint_f1"]
        if comb >= best_score_pair:
            best_score_pair = comb
            best_system_report = rep
            best_oof_boxes = oof_boxes
            best_oof_preds = oof_preds

    # Save best CV predictions to artifacts/dev_cv_predictions.jsonl and official scores
    if best_oof_preds is not None:
        write_and_validate_submission(
            best_oof_preds,
            "artifacts/dev_cv_predictions.jsonl",
            expected_sequence_ids=[s.id for s in seqs],
            dataset_dir=cfg_t2["dataset_dir"],
        )
        run_official_evaluation(
            references_path=cfg_t2["dev_labels_file"],
            predictions_path="artifacts/dev_cv_predictions.jsonl",
            output_json_path="artifacts/scores.json",
            output_csv_path="reports/dev_per_sequence_scores.csv",
            folds_path=cfg_t2["folds_file"],
        )

    # Write reports/ablations.csv
    reports_dir = Path(cfg_t2.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "ablations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    # Run error analysis on the 15 worst dev sequences
    if best_system_report is not None and best_oof_boxes is not None:
        run_error_analysis(
            seqs,
            best_system_report,
            best_oof_boxes,
            perc_hybrid,
            out_dir=reports_dir / "error_analysis",
            top_k=15,
        )

    return {"ablations": ablation_rows}


def run_predict() -> Path:
    """Generate submission.jsonl for the 15 test sequences using frozen config and dev-trained models."""
    cfg_path = "configs/final.yaml" if Path("configs/final.yaml").exists() else "configs/tier2.yaml"
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    score_mod = load_score_module(cfg["dataset_dir"])
    sample_sub = score_mod.load_jsonl(cfg["sample_submission_file"], reference=False, pages=3)
    expected_test_ids = list(sample_sub.keys())

    test_seqs = load_sequences(cfg["dataset_dir"], split="test")
    seq_by_id = {s.id: s for s in test_seqs}

    pipe = MangaTranscriberPipeline(cfg)
    full_dir = Path(cfg.get("models_dir", "artifacts/models")) / "full_dev"
    if (full_dir / "story_clf.joblib").exists():
        from src.learn.lora_train import ContextualTokenRefiner
        from src.learn.order_rank import ReadingOrderRanker
        from src.learn.speaker_pair import SpeakerPairModel
        from src.learn.story_clf import StoryTextClassifier

        pipe.story_clf = StoryTextClassifier.load(full_dir / "story_clf.joblib")
        pipe.story_clf.story_prob_threshold = float(cfg["story_classifier"]["story_prob_threshold"])
        pipe.story_clf.narration_prob_threshold = float(cfg["story_classifier"]["narration_prob_threshold"])
        pipe.order_ranker = ReadingOrderRanker.load(full_dir / "order_ranker.joblib")
        pipe.speaker_pair_model = SpeakerPairModel.load(full_dir / "speaker_pair.joblib")
        if pipe.token_refiner is not None and (full_dir / "token_refiner.joblib").exists():
            pipe.token_refiner = ContextualTokenRefiner.load(full_dir / "token_refiner.joblib")
    else:
        # Train on all dev sequences if full_dev models are not yet saved
        dev_seqs, dev_perc, dev_weak = _load_dev_perceptions_and_weak(cfg)
        pipe.fit_on_sequences(dev_seqs, dev_perc, dev_weak)

    t0 = time.perf_counter()
    test_preds: Dict[str, List[List[Dict[str, str]]]] = {}
    for sid in expected_test_ids:
        seq = seq_by_id[sid]
        test_preds[sid] = pipe.run_sequence(seq, use_cache=True)

    elapsed = time.perf_counter() - t0
    sub_path = write_and_validate_submission(
        test_preds,
        "submission.jsonl",
        expected_sequence_ids=expected_test_ids,
        dataset_dir=cfg["dataset_dir"],
    )
    update_run_manifest(
        extra_data={
            "predict_test": {
                "sequences": len(expected_test_ids),
                "pages": len(expected_test_ids) * 3,
                "wall_clock_sec": round(elapsed, 2),
                "sec_per_page": round(elapsed / float(len(expected_test_ids) * 3), 3),
            }
        }
    )
    print(f"Generated and validated {sub_path} for {len(expected_test_ids)} test sequences in {elapsed:.2f}s.")
    return sub_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Manga Transcriber build & evaluation orchestrator.")
    parser.add_argument(
        "--action",
        choices=["cache", "weaklabels", "train", "tune", "eval", "predict"],
        required=True,
    )
    parser.add_argument("--split", default="development")
    args = parser.parse_args()

    if args.action == "cache":
        run_cache(split=args.split)
    elif args.action == "weaklabels":
        run_weaklabels()
    elif args.action == "train":
        run_train()
    elif args.action == "tune":
        run_tune()
    elif args.action == "eval":
        run_eval()
    elif args.action == "predict":
        run_predict()


if __name__ == "__main__":
    main()
