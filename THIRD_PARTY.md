# Third-Party Models, Tools & Citations (`THIRD_PARTY.md`)

All external models, open-source libraries, and reference works used in this project are listed below with their version, license, URL, and role in the pipeline, in compliance with **Rules R1, R2, and R5**.

---

## 1. Pretrained Open-Weight Models (Local Offline Inference Only)

| Model / Weights | Version / Format | License | URL | Role in Pipeline |
|---|---|---|---|---|
| **PP-OCRv4 Text Detector** (`ch_PP-OCRv4_det_infer.onnx`, DBNet architecture) | PaddleOCR v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR / https://github.com/PaddlePaddle/PaddleOCR | **M2 Text-region detection:** Detects manga speech-balloon text lines and free text at multiple scales before balloon-contour grouping. |
| **PP-OCRv4 / SVTR-LCNet Text Recognizer** (`ch_PP-OCRv4_rec_infer.onnx`) | PaddleOCR v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR / https://github.com/PaddlePaddle/PaddleOCR | **M5 OCR & Ensemble:** Recognizes text crops (both standard 2x upscaled crops and CLAHE/unsharp contrast-enhanced crops). |
| **PP-OCRv4 Text Direction Classifier** (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) | PaddleOCR v2/v4 / RapidOCR 1.4.4 (ONNX) | Apache-2.0 | https://github.com/RapidAI/RapidOCR | **M5 Orientation check:** Handles tilted/rotated text lines inside manga balloons. |
| **OpenCV Haar Frontal Face Cascade** (`haarcascade_frontalface_alt2.xml`) | OpenCV 5.0.0.93 | Apache-2.0 | https://github.com/opencv/opencv | **M4 Character Detection:** Provides candidate frontal face boxes fused with panel line-art figure contour detection. |

> **Verification:** Exact SHA-256 checksums and byte sizes of all local `.onnx` weight files are automatically recorded in `artifacts/run_manifest.json` by `scripts/download_models.py` (`make setup`). No network calls occur during inference (`tests/test_no_network.py`).

---

## 2. Open-Source Software Libraries

| Package | Version | License | URL | Role in Pipeline |
|---|---|---|---|---|
| **Python** | 3.11.9 | PSF-2.0 | https://www.python.org/ | Core runtime environment. |
| **OpenCV** (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | https://opencv.org/ | **M1–M4, M8:** Image preprocessing, connected-component speech balloon & caption detection, tail-tip convexity/contour geometry, XY-cut panel segmentation, HOG/screen-tone descriptors, debug overlays. |
| **ONNX Runtime** (`onnxruntime`) | 1.31.0 | MIT | https://onnxruntime.ai/ | **M2, M5:** Local offline execution of DBNet and SVTR-LCNet ONNX models. |
| **RapidOCR** (`rapidocr-onnxruntime`) | 1.4.4 | Apache-2.0 | https://github.com/RapidAI/RapidOCR | **M2, M5:** Local ONNX wrapper for PaddleOCR v4 detection and recognition weights. |
| **scikit-learn** | 1.9.1 | BSD-3-Clause | https://scikit-learn.org/ | **M6, M7, M9:** `HistGradientBoostingClassifier` for learned story-text filtering, narration classification, pairwise reading-order ranking, and pairwise same-speaker probability estimation. |
| **LightGBM** | 4.7.0 | MIT | https://github.com/microsoft/LightGBM | Gradient boosting toolkit for tabular weak-label learning. |
| **SciPy** | 1.17.1 | BSD-3-Clause | https://scipy.org/ | **M9:** Average-linkage hierarchical agglomerative clustering (`scipy.cluster.hierarchy`). |
| **RapidFuzz** | 3.14.6 | MIT | https://github.com/maxbachmann/RapidFuzz | **M0, M5, 7.1:** C++-accelerated Levenshtein distance for the fast proxy scorer (`src/eval/proxy_score.py`) and constrained OCR post-correction (`src/ocr/correct.py`). |
| **NumPy** | 2.4.6 | BSD-3-Clause | https://numpy.org/ | Array operations, embedding linear algebra, feature matrices. |
| **Pillow** (`PIL`) | 12.3.0 | MIT-CMU | https://python-pillow.org/ | PNG image loading and RGB conversion. |
| **PyTesseract** (`pytesseract`) | 0.3.13 | Apache-2.0 | https://github.com/madmaze/pytesseract | **T0 Baseline:** Wrapper for Tesseract OCR (`Apache-2.0`). |
| **Shapely** / **PyClipper** | 2.2.0 / 1.4.0 | BSD-3-Clause | https://shapely.readthedocs.io/ | Polygon clipping and geometric operations for text detector boxes. |
| **PyYAML** | 6.0.3 | MIT | https://pyyaml.org/ | Configuration parsing (`configs/*.yaml`). |
| **pytest** | 9.1.1 | MIT | https://docs.pytest.org/ | Automated test suite (`tests/`). |

---

## 3. Related Published Literature Cited

1. **Ragot, M., et al. (2024).** *The Manga Whisperer: Automatically Transcribingerek Comics* (Magi). In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)**. https://github.com/ragavsachdeva/magi
2. **Sachdeva, R., & Zisserman, A. (2024/2025).** *Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names* (Magi v2/v3). https://arxiv.org/abs/2407.03540
3. **Liao, M., et al. (2020).** *Real-time Scene Text Detection with Differentiable Binarization (DBNet).* In **AAAI 2020**.
4. **Du, Y., et al. (2022).** *SVTR: Scene Text Recognition with a Single Visual Model (PaddleOCR).* In **IJCAI 2022**.
5. **Matsui, Y., et al. (2017).** *Sketch-based Manga Retrieval using Manga109 Dataset.* **Multimedia Tools and Applications**.
