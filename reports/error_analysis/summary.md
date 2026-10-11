# Error Analysis — 15 Lowest-Scoring Development Sequences

Generated from 5-fold out-of-fold cross-validation predictions evaluated with the official `dataset/score.py`.

| Rank | Sequence ID | `text_order_score` | `balanced_joint_f1` | `joint_f1` | `matched_token_coverage` | Gold Tokens | Pred Tokens | Primary Failure Mode |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `seq_b3247b4b3a2f591b` | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 14 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 2 | `seq_720a7023d4f2ffcb` | 0.312 | 0.159 | 0.527 | 0.648 | 76 | 59 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 3 | `seq_91119fd54cd62066` | 0.366 | 0.159 | 0.346 | 0.585 | 144 | 110 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 4 | `seq_2022b8c0e1f2d3c0` | 0.383 | 0.229 | 0.512 | 0.725 | 100 | 103 | Irregular multi-column panel / balloon reading order inversion |
| 5 | `seq_8f7b98bae85cd5e5` | 0.421 | 0.316 | 0.371 | 0.476 | 124 | 86 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 6 | `seq_71033fec4de494fc` | 0.652 | 0.138 | 0.277 | 0.688 | 433 | 392 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 7 | `seq_cd727990f22c777b` | 0.662 | 0.179 | 0.383 | 0.618 | 241 | 203 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 8 | `seq_a3d73a93577ffd94` | 0.476 | 0.379 | 0.678 | 0.638 | 42 | 34 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 9 | `seq_2032620aa4e4ac7f` | 0.634 | 0.244 | 0.510 | 0.924 | 98 | 124 | Irregular multi-column panel / balloon reading order inversion |
| 10 | `seq_d8bc3c6ac1bb6017` | 0.619 | 0.296 | 0.564 | 0.627 | 56 | 45 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 11 | `seq_98544ce821c3f74b` | 0.710 | 0.211 | 0.396 | 0.642 | 274 | 242 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 12 | `seq_ac5793f8168efdd1` | 0.804 | 0.118 | 0.285 | 0.699 | 300 | 258 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 13 | `seq_5f407d0064bb4a43` | 0.763 | 0.166 | 0.578 | 0.699 | 141 | 118 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 14 | `seq_864aaca43a3580bb` | 0.805 | 0.124 | 0.367 | 0.684 | 334 | 301 | Missed stylized/small balloons or vertical lettering (low token recall) |
| 15 | `seq_98d98dac2a89a339` | 0.739 | 0.204 | 0.493 | 0.745 | 270 | 250 | Speaker cluster merge/split on off-panel or crowd dialogue |

## Recurring Failure Modes Observed

1. **Crowd / Off-Panel Dialogue (`speaker` merge/split):** Sequences with 5–8 speakers or unmarked crowd reactions where speech balloons lack clear tails pointing to visible faces suffer speaker cluster merging or over-splitting.
2. **Stylised / Tiny Punctuation & Scream Balloons:** Extremely small or heavily stylized screams (`GHA-AAA-HH!!`, `?`, `...`) occasionally suffer partial OCR recognition or low detector confidence.
3. **Dense Narration + Dialogue Interleaving:** Pages mixing borderless floating narrator text with character thoughts can cause narration/speech boundary ambiguity.
4. **Irregular Panel Layouts:** Double-column inset panels where a narrow vertical panel spans two rows can occasionally invert pairwise reading order.
