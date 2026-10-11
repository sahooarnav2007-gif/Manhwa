"""Rule R1 & R2 compliance tests: static grep check for hosted APIs + full offline inference with sockets disabled."""
from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from src.data.loader import load_config, load_sequences
from src.pipeline.run_sequence import MangaTranscriberPipeline

FORBIDDEN_PATTERNS = [
    re.compile(r"\bimport\s+(?:openai|anthropic|cohere|replicate)\b"),
    re.compile(r"\bfrom\s+(?:openai|anthropic|cohere|replicate|google\.generativeai)\b"),
    re.compile(r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com"),
]


def test_no_hosted_api_imports_in_codebase() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "scripts"):
        for py_file in (root / folder).rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for pat in FORBIDDEN_PATTERNS:
                assert not pat.search(content), f"Forbidden hosted API pattern {pat.pattern} found in {py_file}"


def test_inference_runs_with_network_sockets_disabled(monkeypatch) -> None:
    """Run the full pipeline inference path on a sequence with all network socket connections blocked."""

    def _blocked_socket(*args, **kwargs):
        raise RuntimeError("Network access is strictly forbidden during inference (Rule R1/R2)")

    def _blocked_connect(*args, **kwargs):
        raise RuntimeError("Socket connect is strictly forbidden during inference (Rule R1/R2)")

    monkeypatch.setattr(socket, "create_connection", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect)

    cfg = load_config("configs/tier2.yaml")
    dev_seqs = load_sequences(cfg["dataset_dir"], split="development")
    pipe = MangaTranscriberPipeline(cfg)
    pred_pages = pipe.run_sequence(dev_seqs[0], use_cache=True)

    assert isinstance(pred_pages, list)
    assert len(pred_pages) == 3
    for page in pred_pages:
        for line in page:
            assert set(line.keys()) == {"speaker", "text"}
            assert line["speaker"] != "UNKNOWN"
            assert len(line["text"].strip()) > 0
