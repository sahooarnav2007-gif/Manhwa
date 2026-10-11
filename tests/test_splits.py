"""Test sequence-grouped 5-fold split integrity and zero split leakage (Rule R6)."""
from __future__ import annotations

from src.data.loader import load_sequences
from src.data.splits import create_grouped_folds, get_fold_split, load_folds, save_folds, verify_no_leakage


def test_grouped_folds_no_leakage(tmp_path) -> None:
    seqs = load_sequences("dataset", split="development")
    folds = create_grouped_folds(seqs, num_folds=5, seed=42)
    assert len(folds) == 80
    assert verify_no_leakage(folds, num_folds=5)

    # Check each fold has 16 sequences
    for f in range(5):
        train_seqs, val_seqs = get_fold_split(seqs, folds, f)
        assert len(val_seqs) == 16
        assert len(train_seqs) == 64
        train_ids = {s.id for s in train_seqs}
        val_ids = {s.id for s in val_seqs}
        assert not (train_ids & val_ids)

    # Test save and load
    fold_file = tmp_path / "folds.json"
    save_folds(folds, fold_file, num_folds=5, seed=42)
    loaded = load_folds(fold_file)
    assert loaded == folds
