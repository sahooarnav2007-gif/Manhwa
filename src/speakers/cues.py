"""M8 Signal 3 — Discourse, vocative, and turn-taking cues from transcribed text."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from src.data.loader import load_score_module
from src.learn.features import NARRATION_CUES

_SCORE = load_score_module()
norm = _SCORE.norm
TOKEN = _SCORE.TOKEN

HONORIFIC_RE = re.compile(r"\b([a-z]{3,12})-(?:san|kun|chan|senpai|sama|sensei)\b")
VOCATIVE_PATTERNS = [
    re.compile(r"^(?:hey|oh|um|well|wait|look|right|good work|good luck|thanks|thank you|sorry|c'mon|come on)[,\s]+([a-z]{3,12})\b"),
    re.compile(r"\b([a-z]{3,12})(?:,\s+(?:you|we're|you're|we'd|did you|do you|are you|can you|have some))\b"),
    re.compile(r"\b(?:what's wrong|counting on you|invite you|flee|tell me that)[,\s]+([a-z]{3,12})\b"),
]

STOP_VOCATIVES = {
    "you", "guys", "everyone", "everybody", "someone", "anyone", "nobody",
    "that", "this", "there", "here", "what", "where", "when", "why", "how",
    "right", "well", "really", "still", "even", "just", "okay", "sure",
    "little", "mean", "think", "know", "look", "wait", "sorry", "thanks",
}


def extract_addressed_names(text: str) -> Set[str]:
    """Extract character names/titles directly addressed (vocatives) in a line of dialogue."""
    t_norm = norm(text)
    names: Set[str] = set()

    for m in HONORIFIC_RE.finditer(t_norm):
        w = m.group(1)
        if w not in STOP_VOCATIVES:
            names.add(w)

    for pat in VOCATIVE_PATTERNS:
        for m in pat.finditer(t_norm):
            w = m.group(1)
            if w not in STOP_VOCATIVES:
                names.add(w)

    if "president" in t_norm and any(
        p in t_norm for p in ("president,", "president!", "president?", "president.")
    ):
        names.add("president")

    return names


class DiscourseCueAnalyzer:
    """Extracts discourse-level cues (vocatives, Q&A turn-taking, continuation, caption narration) per box."""

    def run(self, boxes: List[Dict[str, Any]]) -> None:
        included = [b for b in boxes if b.get("kind") != "exclude"]
        for idx, b in enumerate(included):
            txt = str(b.get("text_raw") or b.get("pre_ocr_text") or "")
            t_norm = norm(txt)
            vocs = sorted(extract_addressed_names(txt))
            b["addressed_names"] = vocs
            b["is_question"] = "?" in txt
            b["is_exclam"] = "!" in txt
            b["starts_with_ellipsis"] = t_norm.startswith("...")
            b["ends_with_ellipsis"] = t_norm.endswith("...")
            b["has_first_person"] = bool(
                re.search(r"\b(i|i'm|i'll|i've|i'd|my|me|mine)\b", t_norm)
            )
            b["has_second_person"] = bool(
                re.search(r"\b(you|you're|you'll|you've|your|yours|yourself)\b", t_norm)
            )

            # Promote obvious rectangular third-person narrator captions to narration
            if (
                b.get("is_caption_box", False)
                and not b["has_first_person"]
                and not b["has_second_person"]
                and (any(c in t_norm for c in NARRATION_CUES) or b.get("narration_prob", 0.0) >= 0.38)
            ):
                b["kind"] = "narration"

            # Combine tail and vlm_marks candidates into unified 'attrib' list
            tail_c = {c["char_id"]: c["score"] for c in b.get("tail_candidates", [])}
            vlm_c = {c["char_id"]: c["score"] for c in b.get("vlm_candidates", [])}
            all_cids = set(tail_c.keys()) | set(vlm_c.keys())
            merged = []
            for cid in all_cids:
                s_tail = tail_c.get(cid, 0.0)
                s_vlm = vlm_c.get(cid, 0.0)
                if s_tail > 0 and s_vlm > 0:
                    comb = 0.55 * s_tail + 0.45 * s_vlm + 0.08  # agreement bonus
                    src = "tail+vlm"
                elif s_tail > 0:
                    comb = s_tail
                    src = "tail"
                else:
                    comb = s_vlm
                    src = "vlm"
                merged.append({"char_id": cid, "score": round(float(min(0.99, comb)), 4), "source": src})
            merged.sort(key=lambda x: x["score"], reverse=True)
            b["attrib"] = merged[:3]
