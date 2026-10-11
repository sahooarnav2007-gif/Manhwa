# Experimental Results & Ablation Report (`reports/results.md`)

All metrics reported below were measured on the **80 development sequences (240 pages)** using **5-fold cross-validation grouped by `sequence_id`** (`configs/folds.json`) and evaluated with the official **`dataset/score.py`** (version 2.1).

---

## 1. Main 5-Fold Cross-Validation Ablation Table

Values are reported as **mean ± std across the 5 grouped folds** (`reports/ablations.csv`).

| System | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `matched_token_coverage` | `speaker_accuracy_on_matched` |
|---|---|---|---|---|---|
| **T0 baseline** (OpenCV bubbles + raw OCR + RTL sort + 1 speaker) | 0.692 ± 0.030 | 0.177 ± 0.022 | 0.377 ± 0.024 | 0.618 ± 0.032 | 0.556 ± 0.034 |
| **+ pretrained detector + better OCR** (DBNet + PP-OCRv4 + constrained cleanup) | 0.728 ± 0.024 | 0.198 ± 0.026 | 0.421 ± 0.030 | 0.717 ± 0.018 | 0.561 ± 0.043 |
| **+ learned story-text filter** (M6 `HistGradientBoosting` story + narration classifier) | 0.750 ± 0.027 | 0.213 ± 0.017 | 0.440 ± 0.019 | 0.705 ± 0.015 | 0.579 ± 0.029 |
| **+ learned reading order** (M7 pairwise GBM ranker + Copeland sort + panel constraints) | 0.810 ± 0.038 | 0.223 ± 0.013 | 0.463 ± 0.022 | 0.743 ± 0.020 | 0.580 ± 0.032 |
| **+ speaker heuristics** (M4/M8 tail geometry + visual marks + discourse cues) | 0.810 ± 0.038 | 0.282 ± 0.032 | 0.430 ± 0.014 | 0.743 ± 0.020 | 0.538 ± 0.024 |
| **+ learned pairwise clustering (T2)** (M9 pairwise GBM + metric-tuned threshold $\tau=0.62$) | **0.810 ± 0.038** | **0.320 ± 0.046** | **0.488 ± 0.022** | **0.743 ± 0.020** | **0.613 ± 0.026** |
| **+ LoRA VLM / contextual token refiner (T3, optional)** | 0.809 ± 0.038 | 0.319 ± 0.045 | 0.487 ± 0.021 | 0.743 ± 0.020 | 0.613 ± 0.024 |

---

## 2. Pipeline Diagnostics

| Diagnostic Metric | Value | Notes |
|---|---|---|
| **Detector + OCR token recall** (`weaklabels`) | **80.00%** (11,646.6 / 14,558 gold tokens) | Fraction of gold tokens across all 240 dev pages recovered by M2+M5 before filtering. |
| **Weak-label box match rate** | **74.23%** (1,743 / 2,348 boxes) | 1,743 boxes aligned to gold story lines (incl. 84 `NARRATION` boxes); 605 unmatched negative boxes (SFX, signs, headers/page numbers). |
| **Reading-order Kendall $\tau$** | **0.9236** (up from **0.6115** in T0) | Pairwise rank correlation on matched story boxes per page. |
| **Precision on gold-empty pages** | **80.00%** (8 / 10 empty pages) | Improved from 60.0% (unfiltered detector) to 80.0% via the learned M6 story filter. |
| **Mean predicted speakers per sequence** | **2.24** (T2) vs **2.94** (heuristics) | Gold mean is 3.2 speakers/sequence; T2 conservative threshold ($\tau=0.62$, `max_clusters=4`) avoids over-splitting penalties in `balanced_joint_f1`. |
| **Sequence speaker merge / split counts** | **49 merge / 8 split** (T2) vs **34 merge / 28 split** (heuristics) | Reducing over-splitting from 28 sequences to 8 sequences increased `balanced_joint_f1` from 0.282 to 0.320 and `joint_f1` from 0.430 to 0.488. |
| **Runtime per page (cached perception rebuild)** | **0.27 s / page** (64.9 s for 240 pages) | Full-page ONNX detection + OCR cold pass runs in ~3.5 s / page on CPU; 5-fold CV training takes 19.8 s. |

---

## 3. Key Findings from Ablations

1. **Multi-Lobe Column Grouping + Pretrained DBNet/PP-OCRv4 (+0.036 `text_order_score`, +0.099 coverage):** Replacing raw OpenCV bubble thresholding with pretrained DBNet line detection grouped strictly vertically by balloon column prevents side-by-side double-balloon lobes from interleaving row-by-row.
2. **Learned Story-Text & Narration Classifier (+0.022 `text_order_score`, +20% empty-page precision):** Training M6 (`HistGradientBoostingClassifier`) on weak box alignments filters out SFX (`SNAP`, `BAM`), margin page numbers, and background signs while identifying rectangular third-person narrator captions as `NARRATION`.
3. **Learned Pairwise Reading-Order Ranker (+0.060 `text_order_score`, Kendall $\tau$: 0.889 → 0.924):** Combining panel XY-cut boundaries with a learned pairwise ranker and Copeland win-score sort resolves multi-balloon ordering inside complex panels and boosts monotonic token alignment coverage from 0.705 to 0.743.
4. **Learned Sequence-Wide Speaker Clustering (+0.097 `balanced_joint_f1` over single-speaker, +0.038 over heuristics):** Fusing character appearance embeddings, balloon tail vectors, visual set-of-marks grounding, vocative exclusions, and Q&A turn-taking in a learned pairwise model (`SpeakerPairModel`) with threshold $\tau=0.62$ tuned on `balanced_joint_f1` achieves `0.320` `balanced_joint_f1` and `0.488` `joint_f1`.
5. **T3 Contextual Token Refiner (-0.0004 `text_order_score`):** Unsupervised/weak dictionary substitution of 1-edit-distance tokens slightly over-corrected rare character names/interjections; therefore **T2 (`configs/final.yaml`)** was selected as the frozen configuration for final test prediction.
