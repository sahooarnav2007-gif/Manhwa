#!/usr/bin/env python3
"""Generate deterministic 5-fold sequence splits and save to configs/folds.json."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, load_sequences
from src.data.splits import create_grouped_folds, save_folds, verify_no_leakage


def main() -> None:
    cfg = load_config()
    seqs = load_sequences(cfg["dataset_dir"], split="development")
    folds = create_grouped_folds(seqs, num_folds=cfg.get("num_folds", 5), seed=cfg.get("seed", 42))
    verify_no_leakage(folds, num_folds=cfg.get("num_folds", 5))
    out_path = save_folds(folds, cfg.get("folds_file", "configs/folds.json"), num_folds=cfg.get("num_folds", 5), seed=cfg.get("seed", 42))
    print(f"Created {len(folds)} sequence fold assignments in {out_path}")


if __name__ == "__main__":
    main()
