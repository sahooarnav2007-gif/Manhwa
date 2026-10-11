# Architecture & Engineering Decisions (`reports/decisions.md`)

Recorded per `README.md` instruction #8: *"If something in `arch.md` is unavailable (a model, a package, a GPU), choose the closest open alternative, record the change in `reports/decisions.md`, and continue."*

---

## 1. Hardware & Execution Environment Detection

* **Detected Hardware:** Windows 10/11 x86_64 with Intel(R) UHD Graphics (CPU-only, no discrete NVIDIA CUDA GPU).
* **Python Environment:** Python 3.11.9 virtual environment (`.venv`) selected over system Python 3.14t to guarantee native precompiled wheels for `onnxruntime`, `rapidocr-onnxruntime`, `opencv-python-headless`, `scikit-learn`, `lightgbm`, `scipy`, and `rapidfuzz`.
* **Adaptation:** Following `arch.md` Section 10 (*"Detect hardware at startup and pick a config... CPU only → T0/T1 with Tesseract/Paddle OCR and cached results"*), we use local ONNX Runtime execution (`CPUExecutionProvider`) with full perception caching (`artifacts/cache/perception/`) so all 285 pages (240 dev + 45 test) are processed once and cached on disk.

---

## 2. Component Selections & Open-Weight Alternatives

1. **M2 Text & Speech-Balloon Detection (`src/vision/detect_text.py`):**
   * *Choice:* Hybrid combination of **OpenCV enclosed white speech-balloon / rectangular caption contour detector** (which extracts balloon solidity, circularity, rectangularity, and tail-tip convexity vectors) and **pretrained DBNet (`ch_PP-OCRv4_det_infer.onnx`, Apache-2.0)** line detection grouped by balloon contour ID and vertical/horizontal proximity.
   * *Rationale:* Runs in < 1.2s/page on CPU via ONNX Runtime, requires zero external internet calls, and achieves high token recall while providing exact balloon contour geometry for M6 (story vs SFX filtering) and M8 (tail-tip speaker attribution).

2. **M5 OCR & Ensemble (`src/ocr/`):**
   * *Choice:* Primary **PaddleOCR v4 SVTR-LCNet (`ch_PP-OCRv4_rec_infer.onnx`, Apache-2.0)** combined with a secondary **2× upscaled CLAHE contrast-enhanced + unsharp-masked crop recognizer** (`VLMCropOCREngine`), a connected-component **Punctuation-Only Balloon Resolver** (`src/ocr/punct.py`), and **edit-distance-constrained post-correction** (`src/ocr/correct.py`).
   * *T0 Fallback:* Since the OS `tesseract` binary is not pre-installed on this Windows machine, `TesseractOCREngine` automatically checks `shutil.which("tesseract")` and falls back to raw binary-thresholded single-crop recognition when the binary is absent, keeping Tier 0 100% self-contained.

3. **M4 Character Detection & Grayscale Line-Art Embeddings (`src/vision/detect_chars.py`, `src/vision/embed.py`):**
   * *Choice:* Frontal face cascade (`haarcascade_frontalface_alt2.xml`) + non-balloon Canny edge / screen-tone figure contour detection per panel, paired with a 64-D L2-normalized **multi-crop grayscale line-art & screen-tone appearance descriptor** (4×4 spatial intensity/contrast pyramid + upper/lower gradient orientation histograms + screen-tone quantile histograms) and an optional discriminative linear projection head.
   * *Rationale:* Manga pages are grayscale line art and screen tones; spatial hair/tone contrast and contour orientation descriptors are fast on CPU and directly capture hair shade, glasses, and uniform darkness across pages.

4. **M6, M7, M9 Learned Components (`src/learn/`):**
   * *Choice:* Regularized `HistGradientBoostingClassifier` models trained on box-level weak labels (`src/learn/weak_align.py`) using 5-fold sequence-grouped cross-validation (`configs/folds.json`):
     * **M6 (`StoryTextClassifier`):** Predicts `story_prob` (story vs excluded SFX/signs/page numbers) and `narration_prob` (`NARRATION` vs character speech).
     * **M7 (`ReadingOrderRanker`):** Pairwise ranker `P(box_a before box_b)` combined via Copeland win-score sorting with panel hard constraints.
     * **M9 (`SpeakerPairModel` + `SequenceSpeakerClusterer`):** Pairwise same-speaker probability model combining character appearance cosine similarity, tail/visual-marks attribution agreement, spatial/balloon continuity, vocative address exclusion, and Q&A turn-taking cues, clustered via average-linkage agglomerative clustering with threshold $\tau$ tuned directly on `balanced_joint_f1`.

5. **T3 Stretch Adaptation (`src/learn/lora_train.py`):**
   * *Choice:* Because a 16 GB CUDA GPU is not present for 4-bit VLM LoRA backprop on this laptop, Tier 3 (`T3`) implements a learned **Contextual Domain Token Refiner** trained on the development folds that corrects 1-edit-distance OCR character substitutions against domain vocabulary under the strict $\le 0.20$ edit-distance constraint.
