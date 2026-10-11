#!/usr/bin/env python3
"""Build a self-contained static site in `vercel_dist/` ready for Vercel deployment."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.serve_dashboard import INDEX_HTML, STATE
from src.learn.order_rank import ReadingOrderRanker
from src.learn.speaker_pair import SpeakerPairModel
from src.learn.story_clf import StoryTextClassifier
from src.pipeline.run_sequence import MangaTranscriberPipeline


def main() -> None:
    out_dir = ROOT / "vercel_dist"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "api" / "sequences").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "development").mkdir(parents=True, exist_ok=True)

    full_dir = ROOT / "artifacts/models/full_dev"
    pipe: MangaTranscriberPipeline = STATE["pipeline"]
    if (full_dir / "story_clf.joblib").exists():
        pipe.story_clf = StoryTextClassifier.load(full_dir / "story_clf.joblib")
        pipe.story_clf.story_prob_threshold = float(STATE["cfg"]["story_classifier"]["story_prob_threshold"])
        pipe.story_clf.narration_prob_threshold = float(STATE["cfg"]["story_classifier"]["narration_prob_threshold"])
        pipe.order_ranker = ReadingOrderRanker.load(full_dir / "order_ranker.joblib")
        pipe.speaker_pair_model = SpeakerPairModel.load(full_dir / "speaker_pair.joblib")

    # 1. Write /api/summary.json
    summary = {
        "test_ids": list(STATE["test_seqs"].keys()),
        "dev_ids": list(STATE["dev_seqs"].keys()),
        "ablations": STATE["ablations"],
        "weak_labels": STATE["manifest"].get("weak_labels", {}),
        "macro_scores": STATE["scores"].get("macro", {}),
    }
    (out_dir / "api" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    # 2. Export all 15 test sequences + 80 dev sequences and compressed WebP/JPEG images
    all_seqs = list(STATE["test_seqs"].values()) + list(STATE["dev_seqs"].values())
    for seq in all_seqs:
        seq_id = seq.id
        pred_pages = (
            STATE["test_preds"].get(seq_id, [[], [], []])
            if seq.split == "test"
            else STATE["dev_preds"].get(seq_id, [[], [], []])
        )
        perc_pages = []
        try:
            perc = pipe.extract_or_load_perception(seq, use_cache=True)
            _, p_boxes = pipe.predict_sequence_from_perception(perc)
            for idx, p_data in enumerate(perc["pages"]):
                perc_pages.append(
                    {
                        "width": p_data["width"],
                        "height": p_data["height"],
                        "panels": p_data.get("panels", []),
                        "boxes": p_boxes[idx] if idx < len(p_boxes) else p_data.get("boxes", []),
                    }
                )
        except Exception:
            perc_pages = []

        seq_payload = {
            "sequence_id": seq.id,
            "split": seq.split,
            "predicted_pages": pred_pages,
            "gold_pages": seq.gold_pages,
            "perception_pages": perc_pages,
            "score": STATE["scores"].get("sequences", {}).get(seq_id),
        }
        (out_dir / "api" / "sequences" / f"{seq_id}.json").write_text(
            json.dumps(seq_payload, ensure_ascii=False), encoding="utf-8"
        )

        # Save compressed JPEG page images for fast Vercel CDN delivery
        for page_num, img_path in enumerate(seq.image_paths, start=1):
            dest_img = out_dir / "images" / seq.split / f"{seq_id}_{page_num}.jpg"
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(dest_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

    # 3. Copy submission.jsonl
    shutil.copy2(ROOT / "submission.jsonl", out_dir / "submission.jsonl")

    # 4. Adapt INDEX_HTML fetch URLs to static files
    static_html = (
        INDEX_HTML.replace("fetch('/api/summary')", "fetch('/api/summary.json')")
        .replace(
            "fetch(`/api/sequence?id=${encodeURIComponent(seqId)}`)",
            "fetch(`/api/sequences/${encodeURIComponent(seqId)}.json`)",
        )
        .replace(
            "/api/image?split=${encodeURIComponent(data.split)}&id=${encodeURIComponent(data.sequence_id)}&page=${pageNum}",
            "/images/${encodeURIComponent(data.split)}/${encodeURIComponent(data.sequence_id)}_${pageNum}.jpg",
        )
        .replace('href="/api/download_submission"', 'href="/submission.jsonl" download="submission.jsonl"')
    )
    (out_dir / "index.html").write_text(static_html, encoding="utf-8")

    # 5. Write vercel.json
    vercel_cfg = {
        "version": 2,
        "cleanUrls": True,
    }
    (out_dir / "vercel.json").write_text(json.dumps(vercel_cfg, indent=2), encoding="utf-8")
    print(f"Built static Vercel bundle in {out_dir}")


if __name__ == "__main__":
    main()
