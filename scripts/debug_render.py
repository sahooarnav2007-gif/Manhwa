#!/usr/bin/env python3
"""Render visual overlays of detected boxes, weak labels, order numbers, and speaker clusters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import Sequence, load_config, load_sequences
from src.vision.preprocess import Preprocessor


def render_page_overlay(
    image_path: Path,
    boxes: List[Dict[str, Any]],
    panels: Optional[List[Dict[str, Any]]] = None,
    weak_boxes: Optional[Dict[str, Any]] = None,
    out_path: Optional[Path] = None,
) -> None:
    prep = Preprocessor()
    p_img = prep.run(image_path, page_num=1)
    canvas = cv2.cvtColor(p_img.rgb, cv2.COLOR_RGB2BGR)

    # Draw panels in subtle blue dashed-like rectangles
    if panels:
        for pan in panels:
            px0, py0, px1, py1 = pan["bbox"]
            cv2.rectangle(canvas, (px0, py0), (px1, py1), (200, 140, 40), 1)
            cv2.putText(
                canvas,
                f"P{pan.get('rank', 1)}",
                (px0 + 4, py0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 140, 40),
                1,
            )

    for b in boxes:
        bx0, by0, bx1, by1 = b["bbox"]
        kind = b.get("kind", "speech")
        if weak_boxes and b["id"] in weak_boxes:
            wb = weak_boxes[b["id"]]
            is_story = wb.get("is_story", 0) == 1
            kind = wb.get("kind", "speech" if is_story else "exclude")

        if kind == "exclude":
            color = (60, 60, 220)  # Red for excluded
        elif kind == "narration":
            color = (20, 180, 220)  # Amber/gold for narration
        else:
            color = (40, 190, 40)  # Green for included speech

        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), color, 2)
        rank = b.get("order_rank", "")
        spk = b.get("cluster") or (weak_boxes.get(b["id"], {}).get("gold_speaker") if weak_boxes else "") or kind
        label = f"#{rank} {spk}" if rank and rank != 999 else f"{b['id']} ({kind})"
        cv2.putText(
            canvas,
            str(label)[:24],
            (bx0, max(12, by0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render debug overlays for visual spot-check of >=20 pages.")
    parser.add_argument("--num-pages", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures/weak_label_audit"))
    args = parser.parse_args()

    cfg = load_config("configs/tier2.yaml")
    seqs = load_sequences(cfg["dataset_dir"], split="development")
    cache_dir = Path(cfg["cache_dir"]) / "perception" / "hybrid_ensemble"
    weak_dir = Path(cfg["weak_labels_dir"])

    rendered = 0
    for seq in seqs:
        perc_file = cache_dir / seq.id / "sequence_perception.json"
        weak_file = weak_dir / f"{seq.id}.json"
        if not perc_file.exists():
            continue
        perc = json.loads(perc_file.read_text(encoding="utf-8"))
        weak = json.loads(weak_file.read_text(encoding="utf-8"))["boxes"] if weak_file.exists() else None
        for p_idx, p_data in enumerate(perc["pages"]):
            out_file = args.out_dir / f"{seq.id}_p{p_idx + 1}.png"
            render_page_overlay(
                seq.image_paths[p_idx],
                p_data["boxes"],
                panels=p_data.get("panels"),
                weak_boxes=weak,
                out_path=out_file,
            )
            rendered += 1
            if rendered >= args.num_pages:
                break
        if rendered >= args.num_pages:
            break
    print(f"Rendered {rendered} page overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
