"""Tests for M9 cross-page speaker clustering consistency and M10 official submission validation."""
from __future__ import annotations

import pytest

from src.data.loader import load_score_module
from src.pipeline.write_submission import validate_submission_file, write_and_validate_submission
from src.speakers.cluster import SequenceSpeakerClusterer


def test_cross_page_speaker_consistency_synthetic() -> None:
    """Assert that a character appearing on pages 1 and 3 receives the same cluster label."""
    emb_hero = [1.0, 0.0, 0.0, 0.0] + [0.0] * 60
    emb_rival = [0.0, 1.0, 0.0, 0.0] + [0.0] * 60

    page_chars = [
        [{"char_id": "p1_c1", "page": 1, "bbox": [100, 100, 200, 250], "embedding": emb_hero}],
        [{"char_id": "p2_c1", "page": 2, "bbox": [400, 200, 500, 350], "embedding": emb_rival}],
        [{"char_id": "p3_c1", "page": 3, "bbox": [120, 110, 220, 260], "embedding": emb_hero}],
    ]
    page_boxes = [
        [
            {
                "id": "p1_b01",
                "page": 1,
                "bbox": [90, 40, 210, 95],
                "panel_id": "p1_panel1",
                "balloon_id": "p1_bal1",
                "order_rank": 1,
                "kind": "speech",
                "attrib": [{"char_id": "p1_c1", "score": 0.92, "source": "tail+vlm"}],
                "addressed_names": ["rival"],
                "text_raw": "HEY RIVAL, WAIT UP!",
            },
            {
                "id": "p1_b02",
                "page": 1,
                "bbox": [20, 10, 300, 35],
                "panel_id": "p1_panel1",
                "balloon_id": "p1_bal2",
                "order_rank": 2,
                "kind": "narration",
                "attrib": [],
                "addressed_names": [],
                "text_raw": "THE FINALS BEGAN.",
            },
        ],
        [
            {
                "id": "p2_b01",
                "page": 2,
                "bbox": [390, 120, 510, 180],
                "panel_id": "p2_panel1",
                "balloon_id": "p2_bal1",
                "order_rank": 1,
                "kind": "speech",
                "attrib": [{"char_id": "p2_c1", "score": 0.90, "source": "tail+vlm"}],
                "addressed_names": [],
                "text_raw": "YOU'RE TOO SLOW!",
            }
        ],
        [
            {
                "id": "p3_b01",
                "page": 3,
                "bbox": [110, 50, 230, 100],
                "panel_id": "p3_panel1",
                "balloon_id": "p3_bal1",
                "order_rank": 1,
                "kind": "speech",
                "attrib": [{"char_id": "p3_c1", "score": 0.94, "source": "tail+vlm"}],
                "addressed_names": ["rival"],
                "text_raw": "RIVAL, I WON'T LOSE!",
            }
        ],
    ]

    clusterer = SequenceSpeakerClusterer(mode="heuristic", distance_threshold=0.52, attach_threshold=0.40, max_clusters=4)
    clustered = clusterer.run(page_boxes, page_chars)

    spk_p1 = clustered[0][0]["cluster"]
    spk_nar = clustered[0][1]["cluster"]
    spk_p2 = clustered[1][0]["cluster"]
    spk_p3 = clustered[2][0]["cluster"]

    assert spk_nar == "NARRATION"
    assert spk_p1 == "char1"
    assert spk_p3 == spk_p1, f"Expected same speaker on page 1 and page 3, got {spk_p1} vs {spk_p3}"
    assert spk_p2 != spk_p1


def test_submission_writer_and_validator(tmp_path) -> None:
    score_mod = load_score_module("dataset")
    sample = score_mod.load_jsonl("dataset/sample_submission.jsonl", reference=False, pages=3)
    expected_ids = list(sample.keys())
    assert len(expected_ids) == 15

    preds = {
        sid: [
            [{"speaker": "char1", "text": "Hello there!"}, {"speaker": "NARRATION", "text": "Meanwhile..."}],
            [{"speaker": "char2", "text": "..."}],
            [],
        ]
        for sid in expected_ids
    }
    sub_file = tmp_path / "submission.jsonl"
    write_and_validate_submission(preds, sub_file, expected_sequence_ids=expected_ids)
    loaded = validate_submission_file(sub_file, expected_sequence_ids=expected_ids)
    assert len(loaded) == 15

    # Reject UNKNOWN speaker
    bad_preds = dict(preds)
    bad_preds[expected_ids[0]] = [[{"speaker": "UNKNOWN", "text": "Invalid speaker"}], [], []]
    bad_file = tmp_path / "bad.jsonl"
    with pytest.raises(ValueError, match="UNKNOWN"):
        write_and_validate_submission(bad_preds, bad_file, expected_sequence_ids=expected_ids)
