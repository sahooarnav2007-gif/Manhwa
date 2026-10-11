#!/usr/bin/env python3
"""Verify local open-weight models, compute SHA-256 hashes, and record in artifacts/run_manifest.json."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import update_run_manifest
from src.vision.detect_text import get_rapidocr


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    engine = get_rapidocr()
    import rapidocr_onnxruntime

    pkg_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent
    onnx_files = sorted(pkg_dir.rglob("*.onnx"))
    models_info = {}
    for f in onnx_files:
        models_info[f.name] = {
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "sha256": sha256_file(f),
            "license": "Apache-2.0",
        }

    manifest = update_run_manifest(
        "artifacts/run_manifest.json",
        extra_data={"local_models": models_info},
    )
    print(f"Verified {len(models_info)} local open-weight ONNX models in artifacts/run_manifest.json.")


if __name__ == "__main__":
    main()
