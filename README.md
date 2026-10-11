# Manga Transcriber — Story Text, Reading Order & Speaker Attribution

A pipeline that reads **three consecutive English manga pages**, extracts the **story text in reading order**, and labels **who speaks each line**, keeping character labels **consistent across all three pages** of a sequence.

Built for the *Manga Task [Advanced]* track using **open-weight models and open-source tools only**. Full technical specification: [`arch.md`](arch.md).

---

## 0. Instructions for the AI agent building this project

Read this whole file, then `arch.md`, then build in the order below.

1. **Obey the hard rules (section 2) before anything else.** If a step would break one, stop and choose another approach.
2. **Build tier by tier** (T0 → T1 → T2 → optional T3). Each tier must produce a valid `submission.jsonl` before you start the next.
3. **Measure everything with the official `score.py`** (use a fast proxy only for inner-loop tuning). Report the macro metrics: `text_order_score`, `balanced_joint_f1`, `joint_f1`, `matched_token_coverage`, `speaker_accuracy_on_matched`.
4. **Never invent results.** Every number in the results table must come from a run you executed. Leave a cell as `TBD` rather than guessing.
5. **Tune on development data only.** Test images are used once, by `make predict`, with frozen configs.
6. **Commit small, test often.** Keep `make test` green. Cache expensive model outputs.
7. **Document as you go**: `THIRD_PARTY.md` (every model/tool/dataset: licence, URL, role), `reports/ablations.csv`, `reports/error_analysis/`.
8. If something in `arch.md` is unavailable (a model, a package, a GPU), choose the closest open alternative, record the change in `reports/decisions.md`, and continue.
9. Ask the human only when blocked by something you cannot decide from these docs (e.g. missing files, hardware you cannot detect).

**Definition of done:** the checklist in `arch.md` section 11 is fully ticked, `make predict && make validate` succeeds, and the results table in section 7 is filled from real runs.

---

## 1. The task in one page

* **Input:** 3 page images per sequence. **80** labelled development sequences, **15** unlabelled test sequences. `sequences.json` lists the page triplets and their split.
* **Output:** one JSON object per sequence, one per line:

```json
{"sequence_id":"seq_001","pages":[
  [{"speaker":"char1","text":"Where are you going?"},{"speaker":"char2","text":"Home."}],
  [{"speaker":"char1","text":"Wait for me!"}],
  []
]}
```

* **Include:** dialogue, inner thoughts (assigned to the thinker), narration (`NARRATION`), clear spoken screams/grunts, punctuation-only balloons (`...`, `?!`, `!`).
* **Exclude:** sound effects and their captions; tiny unclear breaths on the art; titles, logos, credits, page numbers, ads, watermarks, character-intro labels, editor/scanlator notes; writing on signs/clothes/objects; text inside letters, diaries, phone screens, newspapers, UIs.
* **Order:** follow the intended visual order (usually right-to-left, panel by panel). Preserve stutters, repeats and meaningful punctuation; do not preserve printed line wraps.
* **Speakers:** labels like `char1`, `char2` are fine and restart for every sequence; they only need to be consistent within a sequence. Use the exact string `NARRATION` for narrator text. Do not emit `UNKNOWN`.

### Scoring

```bash
python dataset/score.py --references dataset/development/labels.jsonl \
                        --predictions your_predictions.jsonl --output scores.json
```

* `text_order_score` — page text recovery in reading order (edit-distance based; speaker ignored).
* `balanced_joint_f1` — text + speaker recovery with optimal one-to-one label matching; every character counts equally.

What we learned from the scorer (details in `arch.md` section 3):

| Experiment on dev labels | `text_order_score` | `balanced_joint_f1` |
|---|---|---|
| Gold text + gold speakers (oracle) | 1.000 | 1.000 |
| Perfect text, one label for everyone | 1.000 | 0.304 |
| Perfect text, a new label for every line | 1.000 | 0.139 |
| Empty submission | 0.000 | 0.000 |

So **both** merging all characters and over-splitting them are heavily penalised; speaker clustering must be tuned on the real metric.

---

## 2. Hard rules

1. Open-weight models and open-source tools only.
2. **No hosted AI inference APIs** for training targets or test predictions (no OpenAI/Anthropic/Google/etc.). Inference must run offline from local weights.
3. At least one learned component must be **adapted using the development data** (train, fine-tune, or build a harness). An unmodified model call alone is not enough.
4. Test predictions are generated **automatically** from the supplied images: no manual edits, no hand-written answers, no tuning on test images.
5. Cite all external code/models/data and explain our own contribution (section 8).
6. Keep each three-page sequence together in any split.

---

## 3. Approach in brief

A modular **detect → read → judge → order → attribute → cluster** pipeline, with learned components trained on **weak box-level labels** created by aligning OCR output to the gold transcripts (the dev data has no boxes).

```
3 pages ─► text/bubble detection ─► OCR (+ constrained correction)
        ├► panel segmentation ─────────────────┐
        └► character detection + embeddings ───┤
 story-text classifier (learned) ─► reading order (learned + rules)
        ─► speaker candidates (tail geometry, VLM set-of-marks, text cues)
        ─► sequence-wide clustering (learned pairwise + metric-tuned threshold)
        ─► NARRATION handling ─► validate ─► submission.jsonl
```

| Tier | What it adds |
|---|---|
| T0 | OpenCV bubbles + Tesseract + right-to-left sort (always-valid baseline) |
| T1 | Pretrained bubble/text detector, stronger OCR, learned filter and ordering |
| T2 | Character embeddings, tail/VLM speaker attribution, learned pairwise clustering |
| T3 (stretch) | LoRA fine-tuned open VLM for end-to-end page transcription, ensembled with T2 |

Learned parts: story-text/kind classifier, pairwise reading-order ranker, pairwise same-speaker model with clustering threshold tuned for `balanced_joint_f1`, optional appearance projection head and LoRA VLM.

---

## 4. Setup

```bash
git clone <your-repo-url> manga-transcriber && cd manga-transcriber
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sudo apt-get install tesseract-ocr                     # for the T0 baseline / fallback OCR

# place the supplied files here, then:
mkdir -p dataset
unzip development-*.zip -d dataset/
unzip test-*.zip        -d dataset/
cp sequences.json sample_submission.jsonl score.py dataset/

make setup    # download open-weight models, record versions/hashes
make test
```

Expected layout after unzipping:

```
dataset/
├── development/{images/<seq_id>/01.png..03.png, labels.jsonl}
├── test/images/<seq_id>/01.png..03.png
├── sequences.json
├── sample_submission.jsonl
└── score.py
```

Hardware: the code auto-selects a config. A 16 GB-class GPU runs everything; smaller GPUs use 4-bit models; CPU-only runs T0/T1 with cached outputs. Total data is only 285 pages.

---

## 5. Usage

```bash
make cache         # detector + OCR + VLM outputs for dev (cached)
make weaklabels    # align OCR boxes to gold text -> training labels
make train         # grouped 5-fold CV training of learned components
make tune          # tune thresholds/clustering on dev with the proxy scorer
make eval          # official score.py on dev CV predictions -> reports/
make predict       # generate submission.jsonl for the 15 test sequences
make validate      # strict format check using score.py's loader + ID check

# score a dev run manually
python dataset/score.py --references dataset/development/labels.jsonl \
                        --predictions artifacts/dev_cv_predictions.jsonl --output scores.json
```

Outputs: `submission.jsonl` (test), `reports/results.md`, `reports/ablations.csv`, `reports/error_analysis/`.

---

## 6. Repository layout

```
configs/    base + per-tier YAML, fixed CV folds
src/        data, vision, ocr, learn, speakers, pipeline, eval
scripts/    model download, debug overlays, fold creation
tests/      unit, integration, no-network, split-leakage, validator tests
artifacts/  caches, trained models, weak labels, run manifest (git-ignored)
reports/    results, ablations, figures, error analysis
```

See `arch.md` section 9 for the full tree and module responsibilities.

---

## 7. Results (fill from real runs only)

5-fold cross-validation on the 80 development sequences, grouped by sequence, official `score.py`, macro averages (mean ± std over folds).

| System | text_order_score | balanced_joint_f1 | joint_f1 | matched_token_coverage | speaker_accuracy_on_matched |
|---|---|---|---|---|---|
| T0 baseline | 0.692 ± 0.030 | 0.177 ± 0.022 | 0.377 ± 0.024 | 0.618 ± 0.032 | 0.556 ± 0.034 |
| + pretrained detector + better OCR | 0.728 ± 0.024 | 0.198 ± 0.026 | 0.421 ± 0.030 | 0.717 ± 0.018 | 0.561 ± 0.043 |
| + learned story-text filter | 0.750 ± 0.027 | 0.213 ± 0.017 | 0.440 ± 0.019 | 0.705 ± 0.015 | 0.579 ± 0.029 |
| + learned reading order | 0.810 ± 0.038 | 0.223 ± 0.013 | 0.463 ± 0.022 | 0.743 ± 0.020 | 0.580 ± 0.032 |
| + speaker heuristics | 0.810 ± 0.038 | 0.282 ± 0.032 | 0.430 ± 0.014 | 0.743 ± 0.020 | 0.538 ± 0.024 |
| + learned pairwise clustering (T2) | **0.810 ± 0.038** | **0.320 ± 0.046** | **0.488 ± 0.022** | **0.743 ± 0.020** | **0.613 ± 0.026** |
| + LoRA VLM / token refiner (T3, optional) | 0.809 ± 0.038 | 0.319 ± 0.045 | 0.487 ± 0.021 | 0.743 ± 0.020 | 0.613 ± 0.024 |

**Diagnostics measured on 5-fold dev CV (`reports/results.md`):**
* **Detector + OCR token recall:** `0.800` (11,646.6 / 14,558 gold tokens covered across 240 dev pages).
* **Weak-label box match rate:** `0.742` (1,743 story boxes matched incl. 84 `NARRATION` boxes; 605 non-story boxes excluded).
* **Ordering Kendall τ:** `0.924` (improved from `0.612` in T0 and `0.889` before learned pairwise ranking).
* **Precision on gold-empty pages:** `0.800` (8 / 10 gold-empty pages predicted completely empty, up from `0.600` without the learned story filter).
* **Speaker cluster merge/split counts:** `49` under-clustered (merged) sequences / `8` over-clustered (split) sequences in T2 (`mean_pred_speakers = 2.24` vs gold `3.20`), compared to `34` merge / `28` split in heuristic clustering.
* **Runtime per page:** `3.5 s/page` cold ONNX detection + OCR on CPU (`0.27 s/page` from cached perception; `19.8 s` total 5-fold CV training).

---

## 8. What is ours vs. external

**External (cited in `THIRD_PARTY.md`):** pretrained detectors, OCR engines, vision-language models, vision encoders, gradient-boosting/string-metric libraries, and the related published work on manga transcription and speaker association (e.g. the *Magi* line of research: *The Manga Whisperer*, CVPR 2024, and follow-ups). Verify licences when you download anything.

**Our contribution (to be described in the final report, edit to match what is really built):**

1. The **alignment-based weak-labelling harness** that turns transcript-only annotations into box-level training data, reusing the official scorer's normalisation and token matching.
2. The **learned story-text filter** and **reading-order ranker**, optimised against the task's text-order metric rather than box-level F1.
3. The **sequence-wide speaker clustering** that fuses geometry, VLM attribution, appearance embeddings and text cues, with its threshold tuned directly on `balanced_joint_f1`.
4. The **evaluation harness**: fast proxy scorer validated against `score.py`, grouped CV, ablations and error analysis.
5. (Optional) the LoRA-adapted VLM.

---

## 9. Compliance statement

* No hosted inference APIs are used anywhere; inference is offline (`tests/test_no_network.py`).
* All models are open-weight; the list with licences is in `THIRD_PARTY.md`.
* Learned components were trained only on development data, split by sequence.
* Test predictions are produced only by `make predict` from the test images with frozen configs; nothing is edited by hand.

---

## 10. Limitations & future work (update after experiments)

* Only 80 labelled sequences: models are deliberately low-capacity; generalisation to unseen series is not guaranteed.
* Stylised lettering, vertical text, overlapping balloons and off-panel speakers are the expected hard cases.
* Speaker identity from line art is inherently ambiguous; gold itself contains `UNKNOWN` identities.
* Future work: larger external box-annotated data (licence permitting), joint training of order + speaker, better tail detection, end-to-end VLM over all three pages.

---

## 11. Citations

Maintain full references in `THIRD_PARTY.md` and `reports/references.md`. Minimum set to cite if used: the Magi / *Manga Whisperer* papers (if used as component or baseline), every pretrained detector/OCR/VLM/encoder model card, every external dataset, and Tesseract, OpenCV, PyTorch, Transformers, LightGBM/scikit-learn, rapidfuzz.

## 12. Licence

Choose a licence compatible with all third-party components (note any non-commercial or copyleft dependencies) and state it here. The manga page images belong to their rights-holders and are used only for this coursework task; do not redistribute them.
