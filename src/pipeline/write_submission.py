"""M10b — Validated JSONL writer and official format checker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.loader import load_score_module


def write_and_validate_submission(
    predictions: Dict[str, List[List[Dict[str, str]]]],
    output_path: str | Path,
    expected_sequence_ids: Optional[List[str]] = None,
    dataset_dir: str | Path = "dataset",
) -> Path:
    """Write predictions to JSONL and strictly validate with official score.py::load_jsonl."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    order_ids = expected_sequence_ids if expected_sequence_ids is not None else list(predictions.keys())
    lines = []
    for sid in order_ids:
        pages = predictions.get(sid, [[], [], []])
        if len(pages) != 3:
            pages = (list(pages) + [[], [], []])[:3]
        obj = {"sequence_id": sid, "pages": pages}
        lines.append(json.dumps(obj, ensure_ascii=False))

    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_submission_file(out_p, expected_sequence_ids=order_ids, dataset_dir=dataset_dir)
    return out_p


def validate_submission_file(
    submission_path: str | Path,
    expected_sequence_ids: Optional[List[str]] = None,
    dataset_dir: str | Path = "dataset",
) -> Dict[str, Any]:
    """Validate a JSONL file using official score.py::load_jsonl and check ID completeness and no UNKNOWN speaker."""
    score_mod = load_score_module(dataset_dir)
    loaded = score_mod.load_jsonl(submission_path, reference=False, pages=3)

    # Ensure no literal UNKNOWN speaker is ever emitted
    for sid, pages in loaded.items():
        for p_idx, page in enumerate(pages, 1):
            for l_idx, item in enumerate(page, 1):
                if item["speaker"] == "UNKNOWN":
                    raise ValueError(f"{submission_path}: {sid} page {p_idx} line {l_idx} emitted forbidden speaker 'UNKNOWN'")

    if expected_sequence_ids is not None:
        expected_set = set(expected_sequence_ids)
        loaded_set = set(loaded.keys())
        if loaded_set != expected_set:
            missing = sorted(expected_set - loaded_set)
            extra = sorted(loaded_set - expected_set)
            raise ValueError(f"Sequence ID mismatch: missing={missing}, extra={extra}")
        # Also verify exact sequence order
        raw_lines = [
            json.loads(line)["sequence_id"]
            for line in Path(submission_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if raw_lines != list(expected_sequence_ids):
            raise ValueError("Sequence ID order does not match expected order")

    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission.jsonl against official score.py rules and sample_submission.jsonl.")
    parser.add_argument("--submission", type=Path, default=Path("submission.jsonl"))
    parser.add_argument("--sample", type=Path, default=Path("dataset/sample_submission.jsonl"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    score_mod = load_score_module(args.dataset_dir)
    sample_data = score_mod.load_jsonl(args.sample, reference=False, pages=3)
    expected_ids = list(sample_data.keys())
    loaded = validate_submission_file(args.submission, expected_sequence_ids=expected_ids, dataset_dir=args.dataset_dir)
    total_lines = sum(len(p) for pages in loaded.values() for p in pages)
    print(f"VALID: {args.submission} passed official score.py::load_jsonl validation ({len(loaded)} sequences, {total_lines} total lines).")


if __name__ == "__main__":
    main()
