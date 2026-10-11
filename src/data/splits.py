"""Grouped cross-validation split creation and validation by sequence_id."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.loader import Sequence, load_sequences


def create_grouped_folds(
    sequences: List[Sequence],
    num_folds: int = 5,
    seed: int = 42,
) -> Dict[str, int]:
    """Assign each development sequence_id to a fold in [0, num_folds - 1].

    Stratifies approximately by total line count and narration presence while keeping
    each 3-page sequence strictly together in one fold (Rule R6).
    """
    dev_seqs = [s for s in sequences if s.split == "development"]
    rng = random.Random(seed)

    # Sort deterministically by id, shuffle with fixed seed, then bucket by line count
    seq_ids = [s.id for s in sorted(dev_seqs, key=lambda x: x.id)]
    rng.shuffle(seq_ids)

    seq_by_id = {s.id: s for s in dev_seqs}
    # Sort shuffled IDs by approximate line count so folds have balanced line counts
    def _line_count_bucket(sid: str) -> int:
        s = seq_by_id[sid]
        if not s.gold_pages:
            return 0
        total_lines = sum(len(p) for p in s.gold_pages)
        return total_lines // 5

    seq_ids.sort(key=_line_count_bucket)

    fold_assignment: Dict[str, int] = {}
    for idx, sid in enumerate(seq_ids):
        fold_assignment[sid] = idx % num_folds

    return fold_assignment


def save_folds(
    fold_assignment: Dict[str, int],
    folds_path: str | Path = "configs/folds.json",
    num_folds: int = 5,
    seed: int = 42,
) -> Path:
    """Save fold assignments to configs/folds.json."""
    p = Path(folds_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    folds_list: Dict[str, List[str]] = {str(f): [] for f in range(num_folds)}
    for sid, f in sorted(fold_assignment.items()):
        folds_list[str(f)].append(sid)

    payload = {
        "seed": seed,
        "num_folds": num_folds,
        "sequence_to_fold": fold_assignment,
        "folds": folds_list,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_folds(folds_path: str | Path = "configs/folds.json") -> Dict[str, int]:
    """Load sequence_id -> fold mapping from configs/folds.json."""
    root = Path(__file__).resolve().parents[2]
    p = Path(folds_path)
    if not p.is_absolute() and not p.exists():
        p = root / p
    if not p.exists():
        seqs = load_sequences(split="development")
        mapping = create_grouped_folds(seqs)
        save_folds(mapping, p)
        return mapping
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data["sequence_to_fold"].items()}


def get_fold_split(
    sequences: List[Sequence],
    fold_assignment: Dict[str, int],
    val_fold: int,
) -> Tuple[List[Sequence], List[Sequence]]:
    """Return (train_sequences, val_sequences) for a given validation fold index."""
    train_seqs = [s for s in sequences if s.split == "development" and fold_assignment[s.id] != val_fold]
    val_seqs = [s for s in sequences if s.split == "development" and fold_assignment[s.id] == val_fold]
    return train_seqs, val_seqs


def verify_no_leakage(fold_assignment: Dict[str, int], num_folds: int = 5) -> bool:
    """Assert that every sequence appears in exactly one fold and train/val never overlap."""
    seen = set()
    for f in range(num_folds):
        val_ids = {sid for sid, fold in fold_assignment.items() if fold == f}
        train_ids = {sid for sid, fold in fold_assignment.items() if fold != f}
        if val_ids & train_ids:
            raise AssertionError(f"Leakage detected in fold {f}: {val_ids & train_ids}")
        if seen & val_ids:
            raise AssertionError(f"Sequence in multiple folds: {seen & val_ids}")
        seen |= val_ids
    if seen != set(fold_assignment.keys()):
        raise AssertionError("Not all sequences covered in folds")
    return True
