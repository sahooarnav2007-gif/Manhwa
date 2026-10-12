# Manga Transcriber — Story Text, Reading Order & Speaker Attribution

An end-to-end computer vision and machine learning pipeline that reads **three consecutive English manga pages**, extracts the **story text in reading order**, and attributes **who speaks each line** (`char1..charN` or `NARRATION`) consistently across all three pages of a sequence.

- **GitHub Repository:** `https://github.com/sahooarnav2007-gif/Manhwa`
- **Interactive Dashboard:** https://verceldist-ecru.vercel.app
- **Trained Weights Archive:** `https://github.com/sahooarnav2007-gif/Manhwa/raw/main/artifacts/trained_weights.zip`

---

## 1. Task Overview

- **Input:** 3 consecutive manga page images (`01.png`, `02.png`, `03.png`) per sequence (`80` labelled development sequences, `15` unlabelled test sequences defined in `dataset/sequences.json`).
- **Output (`submission.jsonl`):** One JSON object per sequence, one line per sequence:
  ```json
  {"sequence_id":"test_0001","pages":[
    [{"speaker":"char1","text":"Where are you going?"},{"speaker":"char2","text":"Home."}],
    [{"speaker":"char1","text":"Wait for me!"}],
    []
  ]}
  ```
- **Included:** Spoken dialogue, inner thoughts, narration (`NARRATION`), spoken screams/grunts, and punctuation-only speech bubbles (`...`, `?!`, `!`).
- **Excluded:** Background sound effects (`SFX`), tiny unclear breaths on artwork, titles, credits, page numbers, watermarks, scanlator notes, and background sign text.
- **Speaker Attribution:** Character IDs (`char1`, `char2`, ...) are canonicalized by order of first appearance within each 3-page sequence and restart cleanly for the next sequence. Narrator captions are labeled `NARRATION`.

---

## 2. System Architecture

Because the development set provides **sequence-level transcripts only** (`dataset/development/labels.jsonl`) without bounding boxes or page-level speaker coordinates, I designed a modular **detect $\to$ read $\to$ align $\to$ filter $\to$ order $\to$ attribute $\to$ cluster** pipeline trained via monotonic weak-supervision alignment:

```
3 Manga Pages (01.png, 02.png, 03.png)
  │
  ├─► [Stage 1] Vision & Layout Perception (src/vision/)
  │     ├─ Speech-Balloon Connected-Component Detector + Column-Aware DBNet Line Grouping
  │     ├─ Recursive XY-Cut Panel Segmentation (src/vision/panels.py)
  │     ├─ Anime Face Cascade + Upper-Body Figure Detector (src/vision/detect_chars.py)
  │     └─ 60-D Grayscale-Invariant Character Crop Embedder (src/vision/embed.py)
  │
  ├─► [Stage 2] Multi-Pass OCR & Manga Token Normalization (src/ocr/)
  │     ├─ PP-OCRv4 DBNet Detector (ch_PP-OCRv4_det_infer.onnx)
  │     ├─ PP-OCRv4 SVTR-LCNet Recognizer (ch_PP-OCRv4_rec_infer.onnx)
  │     └─ CJK Artifact Stripper, Glued-Word Splitter & Contraction Restorer
  │
  ├─► [Stage 3] Monotonic Weak-Supervision Aligner (src/learn/weak_align.py)
  │     └─ Sequence-Level Dynamic Programming Aligner (0.55 * Token Jaccard + 0.45 * Levenshtein)
  │
  ├─► [Stage 4] Learned Story vs. Non-Story / Narration Classifier (src/learn/story_clf.py)
  │     └─ 3-Class HistGradientBoostingClassifier over 25 Geometric, Visual & Lexical Features
  │
  ├─► [Stage 5] Learned Reading-Order Ranker (src/learn/order_rank.py)
  │     └─ Pairwise HistGradientBoostingClassifier (14 Spatial/Panel Features) + Tournament Sort
  │
  └─► [Stage 6] Sequence-Wide Speaker Attribution & Clustering (src/speakers/, src/learn/speaker_pair.py)
        ├─ Balloon Tail Apex Convex-Hull Tracer + Ray-Cast to Character Boxes
        ├─ Pairwise Same-Speaker Classifier (11 Visual, Turn-Taking & Spatial Features)
        ├─ Constrained Agglomerative Clustering (Complete-Linkage + Cannot-Link Constraints)
        └─ Canonical First-Appearance Renumbering (char1..charN, NARRATION)
```

---

## 3. Setup & Reproducibility

```bash
git clone https://github.com/sahooarnav2007-gif/Manhwa.git
cd Manhwa
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Verify models and run unit + compliance tests:
make setup
make test
```

### End-to-End Pipeline Commands (`Makefile` / `make.bat`)

```bash
make cache         # Extract & cache full-page OCR + visual perception features
make weaklabels    # Run monotonic DP alignment on 80 dev sequences -> artifacts/weak_labels/
make train         # Train 5-fold CV models + full-dev production models -> artifacts/models/
make tune          # Grid-search thresholds on out-of-fold predictions -> configs/final.yaml
make eval          # Run official dataset/score.py on 5-fold CV -> reports/
make predict       # Generate submission.jsonl for the 15 blind test sequences
make validate      # Validate submission.jsonl format, sequence IDs, and schema
```

To launch the local interactive web dashboard:
```bash
python scripts/serve_dashboard.py --port 8000
```

---

## 4. Official 5-Fold Cross-Validation Results

Evaluated on all 80 development sequences using grouped 5-fold cross-validation (`configs/folds.json`, 16 complete 3-page sequences per fold) and scored with the official `dataset/score.py` evaluator (mean $\pm$ std across the 5 folds):

| System | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `matched_token_coverage` | `speaker_accuracy_on_matched` | Kendall $\tau$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **T0 Baseline** (OpenCV bubbles + RTL sort + single speaker) | 0.692 ± 0.030 | 0.177 ± 0.022 | 0.377 ± 0.024 | 0.618 ± 0.032 | 0.556 ± 0.034 | 0.612 |
| **B0** (Raw PP-OCRv4 + top-to-bottom sort + all `char1`) | 0.785 ± 0.036 | 0.268 ± 0.024 | 0.453 ± 0.030 | 0.733 ± 0.019 | 0.587 ± 0.038 | 0.873 |
| **B1** (+ Panel XY-cut + column-aware bubble grouping + rule narration) | 0.808 ± 0.038 | 0.289 ± 0.031 | 0.457 ± 0.027 | 0.739 ± 0.020 | 0.591 ± 0.036 | 0.922 |
| **L1** (+ Learned `StoryTextClassifier` for SFX filtering & narration) | 0.810 ± 0.038 | 0.299 ± 0.035 | 0.456 ± 0.026 | 0.743 ± 0.020 | 0.587 ± 0.035 | 0.923 |
| **L2** (+ Learned pairwise `ReadingOrderRanker`) | 0.810 ± 0.038 | 0.300 ± 0.035 | 0.456 ± 0.026 | 0.743 ± 0.020 | 0.587 ± 0.035 | 0.924 |
| **L3** (+ Tail tracing + character embeddings + `SpeakerPairModel` clustering) | 0.810 ± 0.038 | 0.311 ± 0.043 | 0.478 ± 0.023 | 0.743 ± 0.020 | 0.602 ± 0.029 | 0.924 |
| **T2 Final** (+ Post-processing & CV-tuned thresholds in `configs/final.yaml`) | **0.810 ± 0.038** | **0.320 ± 0.046** | **0.488 ± 0.022** | **0.743 ± 0.020** | **0.613 ± 0.026** | **0.924** |

### Key Pipeline Diagnostics (`reports/results.md`)
- **Candidate OCR Token Recall:** `0.800` (`11,646.6 / 14,558` gold tokens recovered across 240 dev pages).
- **Weak-Supervision Alignment Match Rate:** `0.7423` (`1,743` story boxes matched including `84` `NARRATION` boxes; `605` non-story SFX/margin boxes isolated).
- **Ordering Kendall $\tau$:** `0.9236` across all matched multi-bubble pages.
- **Gold-Empty Page Precision:** `0.800` (`8 / 10` gold-empty pages predicted with `[]`).
- **Runtime:** `0.27 s/page` from cached perception (`3.5 s/page` cold ONNX CPU inference; `19.8 s` total 5-fold CV training time).

---

## 5. What I Built vs. External Tools & Citations

### My Contributions
1. **Sequence-Level Monotonic Weak-Supervision Aligner (`src/learn/weak_align.py`):** Converts ungrounded sequence-level transcripts (`labels.jsonl`) into box-level supervision for story vs. SFX filtering, reading-order ranking, and speaker clustering.
2. **Column-Aware Multi-Lobe Balloon Grouping (`src/vision/detect_text.py`):** Prevents horizontally adjacent lobes of double speech bubbles from interleaving row-by-row, lifting `text_order_score` to `0.810` and Kendall $\tau$ to `0.924`.
3. **Manga OCR Token Normalizer (`src/ocr/punct.py`, `src/ocr/correct.py`):** Strips bilingual CJK false positives on screentone line art, restores comic contractions (`DON'T`, `I'M`), and splits glued uppercase tokens, increasing token recall from `0.717` to `0.800`.
4. **Learned Story Classifier, Reading-Order Ranker & Pairwise Speaker Clusterer (`src/learn/`, `src/speakers/`):** Trained via grouped 5-fold cross-validation (`artifacts/models/fold_0..4/` and `artifacts/models/full_dev/`) and tuned directly against the official `balanced_joint_f1` metric.

### External Open-Weight Models Cited (Local Offline Inference Only)

| Model / Weights | Format / Source | License | URL | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **PP-OCRv4 Text Detector** (`ch_PP-OCRv4_det_infer.onnx`, DBNet) | PaddleOCR v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR | Detects speech-balloon text lines and free text across manga pages. |
| **PP-OCRv4 Text Recognizer** (`ch_PP-OCRv4_rec_infer.onnx`, SVTR-LCNet) | PaddleOCR v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/PaddlePaddle/PaddleOCR | Recognizes English dialogue crops (standard + CLAHE contrast-enhanced). |
| **PP-OCRv4 Direction Classifier** (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) | PaddleOCR v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR | Handles rotated/tilted text lines inside speech bubbles. |
| **Anime Face & Frontal Haar Cascades** (`lbpcascade_animeface.xml`, `haarcascade_frontalface_alt2.xml`) | OpenCV / nagadomi lbpcascade | BSD-2-Clause / Apache-2.0 | https://github.com/nagadomi/lbpcascade_animeface | Detects manga character faces and expands to upper-body bounding boxes. |

### Open-Source Libraries Cited

| Library | Version | License | URL | Role |
| :--- | :--- | :--- | :--- | :--- |
| **OpenCV** (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | https://opencv.org/ | Balloon contour extraction, XY-cut panel splitting, tail apex tracing, character descriptors. |
| **ONNX Runtime** (`onnxruntime`) | 1.31.0 | MIT | https://onnxruntime.ai/ | Local CPU execution of DBNet and SVTR-LCNet ONNX weights. |
| **RapidOCR** (`rapidocr-onnxruntime`) | 1.4.4 | Apache-2.0 | https://github.com/RapidAI/RapidOCR | Local ONNX pipeline wrapper for PP-OCRv4 weights. |
| **scikit-learn** | 1.9.1 | BSD-3-Clause | https://scikit-learn.org/ | `HistGradientBoostingClassifier` for story classification, reading order, and speaker pairs. |
| **LightGBM** | 4.7.0 | MIT | https://github.com/microsoft/LightGBM | Gradient boosting library for tabular weak-supervision models. |
| **SciPy** | 1.17.1 | BSD-3-Clause | https://scipy.org/ | Hierarchical agglomerative clustering & linear sum assignment. |
| **RapidFuzz** | 3.14.6 | MIT | https://github.com/maxbachmann/RapidFuzz | Fast Levenshtein similarity for weak-label DP alignment and proxy scoring. |
| **NumPy / Pillow / Shapely / PyClipper** | 2.4.6 / 12.3.0 / 2.2.0 / 1.4.0 | BSD / MIT | https://numpy.org/ | Array operations, image I/O, and polygon geometry. |

### Academic References Cited
1. **Sachdeva, R., & Zisserman, A. (2024).** *The Manga Whisperer: Automatically Transcribing Comics* (Magi). In **CVPR 2024**. https://github.com/ragavsachdeva/magi
2. **Sachdeva, R., & Zisserman, A. (2024).** *Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names* (Magi v2). https://arxiv.org/abs/2407.03540
3. **Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020).** *Real-time Scene Text Detection with Differentiable Binarization (DBNet).* In **AAAI 2020**.
4. **Du, Y., et al. (2022).** *SVTR: Scene Text Recognition with a Single Visual Model (PaddleOCR).* In **IJCAI 2022**.
5. **Matsui, Y., et al. (2017).** *Sketch-based Manga Retrieval using Manga109 Dataset.* **Multimedia Tools and Applications**.

---

## 6. Compliance Summary

- **Open-weight models & open-source tools only:** All weights (`PP-OCRv4` ONNX, Haar cascades, and our trained `.joblib` gradient-boosted models) are open-weight and documented in `THIRD_PARTY.md`.
- **Zero hosted inference APIs:** All training and inference run 100% locally offline (`tests/test_no_network.py` verifies both static imports and runtime inference with `socket.socket` disabled).
- **Trained weights committed:** All 5-fold CV weights (`artifacts/models/fold_0..4/`) and full-dev production weights (`artifacts/models/full_dev/`) plus `artifacts/trained_weights.zip` are committed in the repository.
- **Automatic test predictions:** `submission.jsonl` is generated strictly by `make predict` from the 15 test image triplets using frozen configuration `configs/final.yaml` with zero manual edits.
