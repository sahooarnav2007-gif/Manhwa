"""Wrapper around official dataset/score.py with fold-level aggregation and CSV export."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.data.loader import load_score_module
from src.data.splits import load_folds

MACRO_KEYS = (
    "text_order_score",
    "balanced_joint_f1",
    "joint_f1",
    "matched_token_coverage",
    "speaker_accuracy_on_matched",
)


def run_official_evaluation(
    references_path: str | Path,
    predictions_path: str | Path,
    output_json_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    folds_path: Optional[str | Path] = "configs/folds.json",
) -> Dict[str, Any]:
    """Run the official dataset/score.py evaluation and compute fold-wise mean and std."""
    score_mod = load_score_module()
    references = score_mod.load_jsonl(references_path, reference=True, pages=3)
    predictions = score_mod.load_jsonl(predictions_path, pages=3)
    report = score_mod.evaluate(references, predictions, pages=3)

    # Compute 5-fold mean and std if folds are available
    try:
        fold_map = load_folds(folds_path) if folds_path else {}
    except Exception:
        fold_map = {}

    if fold_map:
        fold_macros: Dict[int, Dict[str, float]] = {}
        num_folds = max(fold_map.values()) + 1
        for f in range(num_folds):
            fold_rows = [
                srow for sid, srow in report["sequences"].items() if fold_map.get(sid) == f
            ]
            if fold_rows:
                f_macro = {}
                for k in MACRO_KEYS:
                    elig = fold_rows
                    if k in ("text_order_score", "matched_token_coverage"):
                        act = [r for r in elig if r["active_text_pages"] or r["missing_sequence"]]
                        elig = act or elig
                    vals = [r[k] for r in elig if r[k] is not None]
                    f_macro[k] = sum(vals) / len(vals) if vals else None
                fold_macros[f] = f_macro

        fold_summary = {}
        for k in MACRO_KEYS:
            vals = [fold_macros[f][k] for f in sorted(fold_macros) if fold_macros[f].get(k) is not None]
            if vals:
                fold_summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "per_fold": [float(v) for v in vals],
                }
        report["fold_summary"] = fold_summary

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if output_csv_path:
        csv_p = Path(output_csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        with csv_p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sequence_id",
                    "fold",
                    "text_order_score",
                    "balanced_joint_f1",
                    "joint_f1",
                    "matched_token_coverage",
                    "speaker_accuracy_on_matched",
                    "gold_tokens",
                    "predicted_tokens",
                ]
            )
            for sid, srow in sorted(report["sequences"].items()):
                writer.writerow(
                    [
                        sid,
                        fold_map.get(sid, -1),
                        srow.get("text_order_score"),
                        srow.get("balanced_joint_f1"),
                        srow.get("joint_f1"),
                        srow.get("matched_token_coverage"),
                        srow.get("speaker_accuracy_on_matched"),
                        srow.get("gold_tokens"),
                        srow.get("predicted_tokens"),
                    ]
                )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official score.py with fold statistics.")
    parser.add_argument("--references", type=Path, default=Path("dataset/development/labels.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/scores_per_sequence.csv"))
    args = parser.parse_args()
    report = run_official_evaluation(
        references_path=args.references,
        predictions_path=args.predictions,
        output_json_path=args.output,
        output_csv_path=args.csv,
    )
    print(json.dumps(report["macro"], indent=2))


if __name__ == "__main__":
    main()
