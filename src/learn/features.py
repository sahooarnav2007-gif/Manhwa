"""Feature extraction for story-text classification (M6), reading-order ranking (M7), and speaker linking (M9)."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List

import numpy as np

from src.data.loader import load_score_module

_SCORE = load_score_module()
norm = _SCORE.norm
TOKEN = _SCORE.TOKEN

# Common English function & conversational words in manga dialogue
COMMON_DIALOGUE_WORDS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "myself", "yourself",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "don't", "doesn't", "didn't", "can", "can't", "cannot", "could",
    "Couldn't", "will", "won't", "would", "wouldn't", "shall", "should", "shouldn't",
    "may", "might", "must", "need", "ought", "i'm", "you're", "he's", "she's", "it's",
    "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "i'll",
    "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "what", "what's", "who", "who's", "where", "where's",
    "when", "why", "how", "that", "that's", "this", "these", "those", "there", "there's",
    "here", "here's", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "now", "then", "well", "oh", "ah", "uh", "um", "er", "hey", "huh", "yeah", "yes", "right",
    "okay", "ok", "sure", "sorry", "thanks", "thank", "please", "wait", "look", "see", "think",
    "know", "want", "mean", "guess", "going", "go", "come", "get", "got", "make", "made",
    "take", "let", "let's", "say", "said", "tell", "told", "really", "still", "even", "back",
    "way", "good", "great", "little", "much", "never", "always", "something", "anything",
    "nothing", "someone", "everyone", "anyone", "nobody", "time", "day", "one", "two",
    "first", "next", "last", "man", "girl", "guy", "guys", "people", "president", "senpai",
    "club", "school", "home", "work", "fish", "movie", "love", "blood", "money", "devil",
    "about", "after", "again", "against", "already", "also", "although", "always", "among",
    "around", "because", "before", "behind", "between", "but", "by", "down", "during",
    "enough", "ever", "every", "for", "from", "if", "in", "into", "like", "off", "on",
    "out", "over", "Since", "through", "to", "under", "until", "up", "with", "without",
}

# Typical visual SFX / onomatopoeia words excluded by task rules
SFX_WORDS = {
    "snap", "bam", "wham", "thud", "thump", "clang", "clack", "click", "boom", "kaboom",
    "slam", "slap", "smack", "crack", "crunch", "crush", "rustle", "shiver", "tremble",
    "twitch", "stare", "glare", "glance", "nod", "blush", "flush", "sweat", "gulp", "gasp",
    "pant", "wheeze", "sigh", "sob", "sniff", "snicker", "chuckle", "grin", "smirk", "beam",
    "sparkle", "glitter", "shine", "glow", "flash", "zoom", "whoosh", "swish", "swoosh",
    "slash", "slice", "stab", "rip", "tear", "drip", "drop", "splash", "splat", "splatter",
    "sizzle", "rumble", "grumble", "groan", "moan", "roar", "screech", "shriek", "squeak",
    "squeal", "beep", "buzz", "ring", "ding", "dong", "tick", "tock", "tap", "pat", "step",
    "stomp", "dash", "leap", "jump", "slide", "slip", "grab", "grip", "squeeze", "poke",
    "prod", "nudge", "shake", "rattle", "roll", "spin", "twirl", "flip", "flop", "plop",
    "throb", "ba-dump", "badump", "dokidoki", "doki", "zawa", "menacing", "silence",
    "tmp", "fwoosh", "fwip", "shing", "ching", "klang", "krack", "kree", "skree", "zzt",
}

CREDIT_META_WORDS = {
    "chapter", "volume", "vol", "page", "translated", "translation", "typeset", "typesetting",
    "cleaning", "redraw", "scanlation", "scans", "credits", "author", "artist", "story",
    "art", "manga", "weekly", "magazine", "copyright", "reserved", "http", "www", "discord",
}

NARRATION_CUES = (
    "was the kind of",
    "in matters of",
    "in other words",
    "on the verge of",
    "the next day",
    "days-long",
    "at last",
    "meanwhile",
    "originally established",
    "modern times",
    "child companies",
    "zaibatsu",
    "home economics",
    "thought for an instant",
    "used the skill",
    "passed down from",
    "thrown into disarray",
    "stubborn pride",
    "chaos theory",
    "the result",
    "day finally arrived",
)


def extract_box_features(
    box: Dict[str, Any],
    page_boxes: List[Dict[str, Any]],
    page_w: int,
    page_h: int,
) -> np.ndarray:
    """Compute a 32-D numeric feature vector for a single candidate box (for M6 StoryTextClassifier)."""
    bx0, by0, bx1, by1 = box["bbox"]
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    area = float(bw * bh)
    page_area = float(max(1, page_w * page_h))

    cx = 0.5 * (bx0 + bx1) / float(max(1, page_w))
    cy = 0.5 * (by0 + by1) / float(max(1, page_h))
    rel_w = bw / float(max(1, page_w))
    rel_h = bh / float(max(1, page_h))
    rel_area = area / page_area
    aspect = bw / float(bh)

    line_cnt = max(1, int(box.get("line_count", 1)))
    line_h = bh / float(line_cnt)

    # Compare line height to page median line height
    all_line_hs = [
        max(1, b["bbox"][3] - b["bbox"][1]) / float(max(1, b.get("line_count", 1)))
        for b in page_boxes
    ]
    med_line_h = float(np.median(all_line_hs)) if all_line_hs else 20.0
    rel_line_h = line_h / max(1.0, med_line_h)

    text = str(box.get("text_raw") or box.get("pre_ocr_text") or "").strip()
    text_n = norm(text)
    toks = TOKEN.findall(text_n)
    word_toks = [t for t in toks if any(c.isalpha() for c in t)]
    num_toks = len(toks)
    num_words = len(word_toks)
    char_len = len(text_n)

    alpha_chars = [c for c in text if c.isalpha()]
    upper_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / float(len(alpha_chars))
        if alpha_chars
        else 0.0
    )
    digit_ratio = (
        sum(1 for c in text if c.isdigit()) / float(max(1, len(text)))
    )
    is_digit_only = float(bool(text_n) and all(c.isdigit() or c in ".,-/#" for c in text_n))
    is_punct_only = float(bool(box.get("is_punct_only", False)) or (bool(text_n) and all(not c.isalnum() for c in text_n)))

    common_word_hits = sum(1 for w in word_toks if w in COMMON_DIALOGUE_WORDS)
    common_word_ratio = common_word_hits / float(max(1, num_words))

    sfx_hits = sum(1 for w in word_toks if w in SFX_WORDS)
    sfx_ratio = sfx_hits / float(max(1, num_words))
    # Single repeated-character SFX pattern like "SHHHHH", "ZZZZ", "BAMMM" vs spoken scream
    is_single_sfx_word = float(num_words == 1 and sfx_hits == 1)

    credit_hits = sum(1 for w in word_toks if w in CREDIT_META_WORDS)
    has_credit_kw = float(credit_hits > 0)

    has_narration_cue = float(any(cue in text_n for cue in NARRATION_CUES))
    has_question = float("?" in text)
    has_exclam = float("!" in text)
    has_ellipsis = float("..." in text_n)
    has_quotes = float('"' in text_n or "“" in text or "”" in text)
    has_first_person = float(any(w in {"i", "i'm", "i'll", "i've", "i'd", "my", "me", "mine", "we", "we're", "us", "our"} for w in word_toks))

    inside_bubble = float(bool(box.get("inside_bubble", False)))
    is_caption = float(bool(box.get("is_caption_box", False)))
    det_conf = float(box.get("det_conf", 0.5))
    ocr_conf = float(box.get("ocr_conf", det_conf))
    ocr_disagree = float(box.get("ocr_disagreement", 0.0))
    bg_white = float(box.get("bg_white_ratio", 0.5))
    stroke_dark = float(box.get("stroke_dark_ratio", 0.15))
    circularity = float(box.get("circularity", 0.5))
    rect_fill = float(box.get("rect_fill", 0.5))
    solidity = float(box.get("solidity", 0.7))
    tail_prom = float(box.get("tail_prominence", 1.0))

    # Margin / corner indicator (page numbers, headers, footers)
    near_edge = float(cx < 0.06 or cx > 0.94 or cy < 0.035 or cy > 0.965)

    return np.array(
        [
            cx,
            cy,
            rel_w,
            rel_h,
            rel_area,
            aspect,
             float(line_cnt),
            rel_line_h,
            float(char_len),
            float(num_toks),
            float(num_words),
            upper_ratio,
            digit_ratio,
            is_digit_only,
            is_punct_only,
            common_word_ratio,
            sfx_ratio,
            is_single_sfx_word,
            has_credit_kw,
            has_narration_cue,
            has_question,
            has_exclam,
            has_ellipsis,
            has_quotes,
            has_first_person,
            inside_bubble,
            is_caption,
            det_conf,
            ocr_conf,
            ocr_disagree,
            bg_white,
            stroke_dark,
            circularity,
            rect_fill,
            solidity,
            tail_prom,
            near_edge,
        ],
        dtype=np.float32,
    )


def extract_order_pair_features(
    box_a: Dict[str, Any],
    box_b: Dict[str, Any],
    page_w: int,
    page_h: int,
) -> np.ndarray:
    """Compute pairwise features for P(box_a before box_b) in manga reading order (M7)."""
    ax0, ay0, ax1, ay1 = box_a["bbox"]
    bx0, by0, bx1, by1 = box_b["bbox"]

    acx = 0.5 * (ax0 + ax1) / float(max(1, page_w))
    acy = 0.5 * (ay0 + ay1) / float(max(1, page_h))
    bcx = 0.5 * (bx0 + bx1) / float(max(1, page_w))
    bcy = 0.5 * (by0 + by1) / float(max(1, page_h))

    dx = acx - bcx  # In RTL manga, positive dx (A is to the right of B) means A comes before B
    dy = acy - bcy  # Negative dy (A is above B) means A comes before B

    ah = max(1, ay1 - ay0)
    bh = max(1, by1 - by0)
    aw = max(1, ax1 - ax0)
    bw = max(1, bx1 - bx0)

    y_overlap = (min(ay1, by1) - max(ay0, by0)) / float(min(ah, bh))
    x_overlap = (min(ax1, bx1) - max(ax0, bx0)) / float(min(aw, bw))

    pa = int(box_a.get("panel_rank", 1))
    pb = int(box_b.get("panel_rank", 1))
    same_panel = float(pa == pb)
    panel_diff = float(pa - pb)

    same_balloon = float(
        box_a.get("balloon_id") is not None
        and box_a.get("balloon_id") == box_b.get("balloon_id")
    )

    # Baseline RTL rule score:
    # 1. Earlier panel comes first
    # 2. Within same panel, if vertically separated (y_overlap < 0.25), higher box (smaller y) comes first;
    #    if in same horizontal band, rightmost box (larger x) comes first!
    if pa != pb:
        rule_a_before_b = 1.0 if pa < pb else -1.0
    elif same_balloon:
        rule_a_before_b = 1.0 if (ay0, ax0) < (by0, bx0) else -1.0
    else:
        # Check if A is clearly above B vs to the right of B
        # Manga intra-panel rule: top-right to bottom-left, where vertical separation dominates unless horizontally far
        rtl_score_a = acy - 0.38 * acx
        rtl_score_b = bcy - 0.38 * bcx
        rule_a_before_b = 1.0 if rtl_score_a < rtl_score_b else -1.0

    return np.array(
        [
            dx,
            dy,
            ax0 / float(max(1, page_w)) - bx0 / float(max(1, page_w)),
            ay0 / float(max(1, page_h)) - by0 / float(max(1, page_h)),
            ax1 / float(max(1, page_w)) - bx1 / float(max(1, page_w)),
            ay1 / float(max(1, page_h)) - by1 / float(max(1, page_h)),
            y_overlap,
            x_overlap,
            same_panel,
            panel_diff,
            same_balloon,
            rule_a_before_b,
            float(box_a.get("is_caption_box", False)) - float(box_b.get("is_caption_box", False)),
            math.log(max(1.0, aw * ah) / max(1.0, bw * bh)),
        ],
        dtype=np.float32,
    )
