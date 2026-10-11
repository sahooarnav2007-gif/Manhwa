"""Phase P0 tests: data loader, oracle score, and proxy vs official scorer agreement on 20 random perturbations."""
from __future__ import annotations

import copy
import random
from pathlib import Path

from src.data.loader import load_score_module, load_sequences
from src.eval.proxy_score import proxy_evaluate


def test_loader_counts_and_images_exist() -> None:
    dev_seqs = load_sequences("dataset", split="development")
    test_seqs = load_sequences("dataset", split="test")
    assert len(dev_seqs) == 80
    assert len(test_seqs) == 15
    for seq in dev_seqs + test_seqs:
        assert len(seq.image_paths) == 3
        for p in seq.image_paths:
            assert Path(p).exists(), f"Missing image {p}"


def test_oracle_scores_one() -> None:
    score_mod = load_score_module("dataset")
    refs = score_mod.load_jsonl("dataset/development/labels.jsonl", reference=True, pages=3)
    # Build oracle prediction (replace None gold speakers with a dummy label since gold UNKNOWN tokens are ignored)
    oracle_preds = {}
    for sid, pages in refs.items():
        pred_pages = []
        for page in pages:
            pred_page = []
            for row in page:
                spk = row["speaker"] if row["speaker"] is not None else "char_unk_placeholder"
                pred_page.append({"speaker": spk, "text": row["text"]})
            pred_pages.append(pred_page)
        oracle_preds[sid] = pred_pages

    official_rep = score_mod.evaluate(refs, oracle_preds, pages=3)
    proxy_rep = proxy_evaluate(refs, oracle_preds, pages=3)
    assert abs(official_rep["macro"]["text_order_score"] - 1.0) < 1e-9
    assert abs(official_rep["macro"]["balanced_joint_f1"] - 1.0) < 1e-9
    assert abs(proxy_rep["macro"]["text_order_score"] - 1.0) < 1e-9
    assert abs(proxy_rep["macro"]["balanced_joint_f1"] - 1.0) < 1e-9


def _perturb_predictions(refs: dict, rng: random.Random, mode: int) -> dict:
    """Create diverse perturbed predictions from gold references."""
    preds = {}
    subset_ids = sorted(refs.keys())[:8]  # 8 sequences per random trial so 20 official-scorer runs finish in <3s
    sub_refs = {sid: refs[sid] for sid in subset_ids}
    for sid, pages in sub_refs.items():
        pred_pages = []
        line_counter = 0
        for page in pages:
            new_page = []
            for row in page:
                line_counter += 1
                if rng.random() < 0.15 and mode % 3 == 0:
                    continue  # drop line
                text = row["text"]
                if rng.random() < 0.25 and len(text) > 5:
                    idx = rng.randint(0, len(text) - 2)
                    text = text[:idx] + "X" + text[idx + 1 :]
                spk = row["speaker"] or "char1"
                if mode % 5 == 1:
                    spk = "char1" if spk != "NARRATION" else "NARRATION"
                elif mode % 5 == 2:
                    spk = f"line_{line_counter}" if spk != "NARRATION" else "NARRATION"
                elif rng.random() < 0.3:
                    spk = f"char_{rng.randint(1, 4)}"
                new_page.append({"speaker": spk, "text": text})
            if rng.random() < 0.25 and len(new_page) > 1:
                rng.shuffle(new_page)
            pred_pages.append(new_page)
        preds[sid] = pred_pages
    return sub_refs, preds


def test_proxy_agrees_with_official_on_20_perturbations() -> None:
    score_mod = load_score_module("dataset")
    refs = score_mod.load_jsonl("dataset/development/labels.jsonl", reference=True, pages=3)
    rng = random.Random(12345)
    for trial in range(20):
        sub_refs, preds = _perturb_predictions(copy.deepcopy(refs), rng, trial)
        off = score_mod.evaluate(sub_refs, preds, pages=3)["macro"]
        prx = proxy_evaluate(sub_refs, preds, pages=3)["macro"]
        for metric in ("text_order_score", "balanced_joint_f1", "joint_f1", "matched_token_coverage", "speaker_accuracy_on_matched"):
            diff = abs(off[metric] - prx[metric])
            assert diff < 0.01, f"Trial {trial} metric {metric} differed by {diff}: official={off[metric]} proxy={prx[metric]}"
