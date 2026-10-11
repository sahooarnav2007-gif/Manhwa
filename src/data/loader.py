"""Data loader, configuration management, and run manifest utilities."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import yaml


@dataclass
class Sequence:
    """Represents a 3-page manga sequence."""

    id: str
    split: str
    image_paths: List[Path]
    gold_pages: Optional[List[List[Dict[str, Optional[str]]]]] = None


def load_score_module(dataset_dir: str | Path = "dataset"):
    """Dynamically import the official dataset/score.py module."""
    score_path = Path(dataset_dir) / "score.py"
    if not score_path.exists():
        # Try relative to project root
        root = Path(__file__).resolve().parents[2]
        score_path = root / "dataset" / "score.py"
    spec = importlib.util.spec_from_file_location("official_score", score_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official score.py from {score_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(tier_config_path: Optional[str | Path] = None, base_config_path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    """Load base configuration and merge optional tier configuration."""
    root = Path(__file__).resolve().parents[2]
    base_p = Path(base_config_path)
    if not base_p.is_absolute() and not base_p.exists():
        base_p = root / base_p
    cfg = yaml.safe_load(base_p.read_text(encoding="utf-8"))
    if tier_config_path is not None:
        tier_p = Path(tier_config_path)
        if not tier_p.is_absolute() and not tier_p.exists():
            tier_p = root / tier_p
        if tier_p.exists():
            tier_cfg = yaml.safe_load(tier_p.read_text(encoding="utf-8")) or {}
            cfg = deep_merge(cfg, tier_cfg)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Compute a deterministic short SHA-256 hash of a configuration dictionary."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_sequences(
    dataset_dir: str | Path = "dataset",
    split: Optional[str] = None,
) -> List[Sequence]:
    """Load sequences from sequences.json and development/labels.jsonl."""
    root = Path(__file__).resolve().parents[2]
    ds_dir = Path(dataset_dir)
    if not ds_dir.is_absolute() and not ds_dir.exists():
        ds_dir = root / ds_dir

    seq_file = ds_dir / "sequences.json"
    raw_sequences = json.loads(seq_file.read_text(encoding="utf-8"))

    labels_file = ds_dir / "development" / "labels.jsonl"
    gold_map: Dict[str, List[List[Dict[str, Optional[str]]]]] = {}
    if labels_file.exists():
        score_mod = load_score_module(ds_dir)
        gold_map = score_mod.load_jsonl(labels_file, reference=True, pages=3)

    sequences: List[Sequence] = []
    for item in raw_sequences:
        seq_split = item["split"]
        if split is not None and seq_split != split:
            continue
        seq_id = item["sequence_id"]
        img_paths = [ds_dir / rel_path for rel_path in item["images"]]
        gold_pages = copy.deepcopy(gold_map.get(seq_id)) if seq_split == "development" else None
        sequences.append(
            Sequence(
                id=seq_id,
                split=seq_split,
                image_paths=img_paths,
                gold_pages=gold_pages,
            )
        )
    return sequences


def iter_sequences(
    dataset_dir: str | Path = "dataset",
    split: Optional[str] = None,
) -> Iterator[Sequence]:
    """Yield Sequence objects from the dataset."""
    yield from load_sequences(dataset_dir=dataset_dir, split=split)


def detect_hardware_profile() -> Dict[str, Any]:
    """Detect available hardware (CPU/GPU) and select execution profile."""
    gpu_available = False
    gpu_name = None
    vram_gb = 0.0
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            gpu_available = True
            gpu_name = "CUDAExecutionProvider"
    except Exception:
        providers = ["CPUExecutionProvider"]

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "recommended_profile": "gpu" if gpu_available else "cpu_cached",
    }


def update_run_manifest(
    manifest_path: str | Path = "artifacts/run_manifest.json",
    extra_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or update artifacts/run_manifest.json with hardware, package versions, and timings."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    if p.exists():
        try:
            manifest = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    import cv2
    import lightgbm
    import numpy
    import PIL
    import rapidfuzz
    import sklearn

    manifest["hardware"] = detect_hardware_profile()
    manifest["packages"] = {
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "Pillow": PIL.__version__,
        "rapidfuzz": rapidfuzz.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    if extra_data:
        manifest = deep_merge(manifest, extra_data)

    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
