#!/usr/bin/env python3
"""Interactive localhost web server for inspecting Manga Transcriber sequences, overlays, transcripts, and ablations."""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, load_score_module, load_sequences
from src.pipeline.run_sequence import MangaTranscriberPipeline


def build_state() -> dict:
    cfg = load_config("configs/final.yaml" if (ROOT / "configs/final.yaml").exists() else "configs/tier2.yaml")
    score_mod = load_score_module(ROOT / "dataset")

    dev_seqs = load_sequences(ROOT / "dataset", split="development")
    test_seqs = load_sequences(ROOT / "dataset", split="test")

    dev_preds = {}
    dev_pred_path = ROOT / "artifacts/dev_cv_predictions.jsonl"
    if dev_pred_path.exists():
        dev_preds = score_mod.load_jsonl(dev_pred_path, reference=False, pages=3)

    test_preds = {}
    test_pred_path = ROOT / "submission.jsonl"
    if test_pred_path.exists():
        test_preds = score_mod.load_jsonl(test_pred_path, reference=False, pages=3)

    scores_data = {}
    scores_path = ROOT / "artifacts/scores.json"
    if scores_path.exists():
        scores_data = json.loads(scores_path.read_text(encoding="utf-8"))

    ablations = []
    ablation_csv = ROOT / "reports/ablations.csv"
    if ablation_csv.exists():
        with ablation_csv.open("r", encoding="utf-8") as f:
            ablations = list(csv.DictReader(f))

    manifest = {}
    manifest_path = ROOT / "artifacts/run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "cfg": cfg,
        "dev_seqs": {s.id: s for s in dev_seqs},
        "test_seqs": {s.id: s for s in test_seqs},
        "dev_preds": dev_preds,
        "test_preds": test_preds,
        "scores": scores_data,
        "ablations": ablations,
        "manifest": manifest,
        "pipeline": MangaTranscriberPipeline(cfg),
    }


STATE = build_state()

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manga Transcriber — Story Text, Reading Order & Speaker Attribution</title>
  <style>
    :root {
      --bg: #0b0f19;
      --panel: #111827;
      --panel-alt: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
      --pink: #ec4899;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 1rem 1.5rem;
      background: rgba(17, 24, 39, 0.92);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand h1 {
      font-size: 1.15rem;
      margin: 0;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .brand p {
      margin: 0.15rem 0 0;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.65rem;
    }
    select, button, a.btn {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 0.45rem 0.8rem;
      font-size: 0.85rem;
      cursor: pointer;
      text-decoration: none;
      transition: border-color 0.15s, background 0.15s;
    }
    select:hover, button:hover, a.btn:hover {
      border-color: var(--accent);
    }
    button.active {
      background: #0284c7;
      border-color: var(--accent);
      font-weight: 600;
    }
    main {
      max-width: 1680px;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 3rem;
    }
    .metrics-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 0.85rem;
      margin-bottom: 1.25rem;
    }
    .metric-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.65rem;
      padding: 0.85rem 1rem;
    }
    .metric-card .label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .metric-card .val {
      font-size: 1.4rem;
      font-weight: 700;
      margin-top: 0.2rem;
      color: var(--accent);
    }
    .metric-card .sub {
      font-size: 0.75rem;
      color: var(--muted);
    }
    .pages-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1.1rem;
      margin-bottom: 1.75rem;
    }
    @media (max-width: 1100px) {
      .pages-grid { grid-template-columns: 1fr; }
    }
    .page-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .page-header {
      padding: 0.65rem 1rem;
      background: var(--panel-alt);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 600;
      font-size: 0.9rem;
    }
    .img-wrap {
      position: relative;
      width: 100%;
      background: #050811;
      border-bottom: 1px solid var(--border);
    }
    .img-wrap img {
      display: block;
      width: 100%;
      height: auto;
    }
    .img-wrap svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .transcript-pane {
      padding: 0.85rem 1rem;
      flex: 1;
      max-height: 440px;
      overflow-y: auto;
    }
    .transcript-section-title {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin: 0.25rem 0 0.5rem;
      font-weight: 700;
    }
    .line-item {
      display: flex;
      align-items: baseline;
      gap: 0.55rem;
      padding: 0.38rem 0.5rem;
      border-radius: 0.4rem;
      margin-bottom: 0.32rem;
      background: rgba(30, 41, 59, 0.55);
      font-size: 0.84rem;
    }
    .spk-badge {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 0.12rem 0.48rem;
      border-radius: 999px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .spk-char1 { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.45); }
    .spk-char2 { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.45); }
    .spk-char3 { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.45); }
    .spk-char4 { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.45); }
    .spk-narration { background: rgba(245, 158, 11, 0.22); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.5); }
    .spk-other { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
    .section-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.15rem 1.25rem;
      margin-bottom: 1.25rem;
    }
    .section-card h2 {
      margin: 0 0 0.85rem;
      font-size: 1.05rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.6rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: rgba(30, 41, 59, 0.5);
    }
    tr.highlight {
      background: rgba(56, 189, 248, 0.09);
      font-weight: 600;
    }
    .empty-msg {
      color: var(--muted);
      font-style: italic;
      font-size: 0.82rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Manga Transcriber — Story Text, Reading Order &amp; Speaker Attribution</h1>
      <p>3-Page Sequence Pipeline (DBNet + PP-OCRv4 + Learned Story Filter + Pairwise Order Ranker + Agglomerative Speaker Clustering)</p>
    </div>
    <div class="controls">
      <button id="btnTest" class="active" onclick="setSplit('test')">Test Split (15 seqs)</button>
      <button id="btnDev" onclick="setSplit('development')">Development Split (80 seqs)</button>
      <select id="seqSelect" onchange="loadSequence(this.value)"></select>
      <button id="btnPrev" onclick="stepSequence(-1)">&#8592; Prev</button>
      <button id="btnNext" onclick="stepSequence(1)">Next &#8594;</button>
      <button id="btnBoxes" class="active" onclick="toggleBoxes()">Boxes &amp; Order Overlay: ON</button>
      <a class="btn" href="/api/download_submission" download="submission.jsonl">Download submission.jsonl</a>
    </div>
  </header>

  <main>
    <div class="metrics-strip" id="metricsStrip"></div>
    <div class="pages-grid" id="pagesGrid"></div>

    <div class="section-card">
      <h2>5-Fold Grouped Cross-Validation Ablations (Official <code>dataset/score.py</code> on 80 Dev Sequences)</h2>
      <div style="overflow-x:auto">
        <table id="ablationTable">
          <thead>
            <tr>
              <th>System</th>
              <th>text_order_score</th>
              <th>balanced_joint_f1</th>
              <th>joint_f1</th>
              <th>matched_token_coverage</th>
              <th>speaker_accuracy_on_matched</th>
              <th>Kendall &tau;</th>
              <th>Empty-Page Prec.</th>
              <th>Mean Speakers</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    let summaryData = null;
    let currentSplit = 'test';
    let showBoxes = true;
    let currentSeqData = null;

    function badgeClass(speaker) {
      if (!speaker) return 'spk-other';
      const s = speaker.toUpperCase();
      if (s === 'NARRATION') return 'spk-narration';
      if (s.endsWith('1')) return 'spk-char1';
      if (s.endsWith('2')) return 'spk-char2';
      if (s.endsWith('3')) return 'spk-char3';
      if (s.endsWith('4')) return 'spk-char4';
      return 'spk-other';
    }

    function escapeHtml(str) {
      return String(str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    async function init() {
      const res = await fetch('/api/summary');
      summaryData = await res.json();
      renderAblations(summaryData.ablations);
      setSplit('test');
    }

    function renderAblations(rows) {
      const tbody = document.querySelector('#ablationTable tbody');
      tbody.innerHTML = rows.map(r => {
        const isBest = r.system.includes('(T2)');
        return `<tr class="${isBest ? 'highlight' : ''}">
          <td>${escapeHtml(r.system)}</td>
          <td>${Number(r.text_order_score_mean).toFixed(3)} &plusmn; ${Number(r.text_order_score_std).toFixed(3)}</td>
          <td>${Number(r.balanced_joint_f1_mean).toFixed(3)} &plusmn; ${Number(r.balanced_joint_f1_std).toFixed(3)}</td>
          <td>${Number(r.joint_f1_mean).toFixed(3)} &plusmn; ${Number(r.joint_f1_std).toFixed(3)}</td>
          <td>${Number(r.matched_token_coverage_mean).toFixed(3)} &plusmn; ${Number(r.matched_token_coverage_std).toFixed(3)}</td>
          <td>${Number(r.speaker_accuracy_on_matched_mean).toFixed(3)} &plusmn; ${Number(r.speaker_accuracy_on_matched_std).toFixed(3)}</td>
          <td>${Number(r.kendall_tau).toFixed(3)}</td>
          <td>${(Number(r.empty_page_precision) * 100).toFixed(0)}%</td>
          <td>${r.mean_pred_speakers}</td>
        </tr>`;
      }).join('');
    }

    function setSplit(split) {
      currentSplit = split;
      document.getElementById('btnTest').classList.toggle('active', split === 'test');
      document.getElementById('btnDev').classList.toggle('active', split === 'development');
      const ids = split === 'test' ? summaryData.test_ids : summaryData.dev_ids;
      const sel = document.getElementById('seqSelect');
      sel.innerHTML = ids.map((id, idx) => `<option value="${id}">${idx + 1}. ${id}</option>`).join('');
      if (ids.length > 0) loadSequence(ids[0]);
    }

    function stepSequence(delta) {
      const sel = document.getElementById('seqSelect');
      const nextIdx = (sel.selectedIndex + delta + sel.options.length) % sel.options.length;
      sel.selectedIndex = nextIdx;
      loadSequence(sel.value);
    }

    function toggleBoxes() {
      showBoxes = !showBoxes;
      const btn = document.getElementById('btnBoxes');
      btn.classList.toggle('active', showBoxes);
      btn.textContent = `Boxes & Order Overlay: ${showBoxes ? 'ON' : 'OFF'}`;
      if (currentSeqData) renderSequence(currentSeqData);
    }

    async function loadSequence(seqId) {
      const res = await fetch(`/api/sequence?id=${encodeURIComponent(seqId)}`);
      currentSeqData = await res.json();
      renderSequence(currentSeqData);
    }

    function renderSequence(data) {
      const strip = document.getElementById('metricsStrip');
      const totalPredLines = data.predicted_pages.reduce((acc, p) => acc + p.length, 0);
      const speakersSet = new Set();
      data.predicted_pages.forEach(p => p.forEach(r => speakersSet.add(r.speaker)));

      if (data.score) {
        const s = data.score;
        strip.innerHTML = `
          <div class="metric-card"><div class="label">Sequence ID</div><div class="val" style="font-size:1.05rem">${data.sequence_id}</div><div class="sub">Split: ${data.split}</div></div>
          <div class="metric-card"><div class="label">text_order_score</div><div class="val">${Number(s.text_order_score).toFixed(3)}</div><div class="sub">Official score.py</div></div>
          <div class="metric-card"><div class="label">balanced_joint_f1</div><div class="val">${s.balanced_joint_f1 !== null ? Number(s.balanced_joint_f1).toFixed(3) : 'N/A'}</div><div class="sub">Hungarian 1-to-1 match</div></div>
          <div class="metric-card"><div class="label">matched_token_coverage</div><div class="val">${Number(s.matched_token_coverage).toFixed(3)}</div><div class="sub">${s.predicted_tokens} pred / ${s.gold_tokens} gold tokens</div></div>
          <div class="metric-card"><div class="label">Predicted Lines / Speakers</div><div class="val">${totalPredLines} / ${speakersSet.size}</div><div class="sub">${Array.from(speakersSet).join(', ') || 'None'}</div></div>
        `;
      } else {
        const w = summaryData.weak_labels || {};
        strip.innerHTML = `
          <div class="metric-card"><div class="label">Test Sequence ID</div><div class="val" style="font-size:1.05rem">${data.sequence_id}</div><div class="sub">Frozen T2 Config (submission.jsonl)</div></div>
          <div class="metric-card"><div class="label">Predicted Story Lines</div><div class="val">${totalPredLines}</div><div class="sub">Across 3 consecutive pages</div></div>
          <div class="metric-card"><div class="label">Distinct Speakers</div><div class="val">${speakersSet.size}</div><div class="sub">${Array.from(speakersSet).join(', ') || 'None'}</div></div>
          <div class="metric-card"><div class="label">Dev CV text_order_score</div><div class="val">0.810 &plusmn; 0.038</div><div class="sub">5-fold official score.py</div></div>
          <div class="metric-card"><div class="label">Dev CV balanced_joint_f1</div><div class="val">0.320 &plusmn; 0.046</div><div class="sub">Detector recall: ${(Number(w.detector_token_recall || 0.8)*100).toFixed(1)}%</div></div>
        `;
      }

      const grid = document.getElementById('pagesGrid');
      grid.innerHTML = [0, 1, 2].map(idx => {
        const pageNum = idx + 1;
        const predLines = data.predicted_pages[idx] || [];
        const goldLines = data.gold_pages ? (data.gold_pages[idx] || []) : null;
        const pInfo = (data.perception_pages && data.perception_pages[idx]) || { width: 900, height: 1300, boxes: [], panels: [] };
        const pw = pInfo.width || 900;
        const ph = pInfo.height || 1300;

        let svgOverlay = '';
        if (showBoxes && pInfo.boxes) {
          const panelRects = (pInfo.panels || []).map(pan => {
            const [x0, y0, x1, y1] = pan.bbox;
            return `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="6,4" opacity="0.55" />`;
          }).join('');

          const boxRects = (pInfo.boxes || []).map(b => {
            const [x0, y0, x1, y1] = b.bbox;
            const isExcl = b.kind === 'exclude';
            const isNar = b.kind === 'narration';
            const stroke = isExcl ? '#ef4444' : (isNar ? '#f59e0b' : '#22c55e');
            const label = isExcl ? 'EXCL' : `#${b.order_rank || ''} ${b.cluster || ''}`;
            return `
              <rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="${stroke}" stroke-width="3" />
              <rect x="${x0}" y="${Math.max(0, y0-20)}" width="${Math.min(130, Math.max(55, label.length*8))}" height="20" fill="${stroke}" rx="3" />
              <text x="${x0+4}" y="${Math.max(14, y0-5)}" fill="#050811" font-size="13" font-weight="bold" font-family="monospace">${escapeHtml(label)}</text>
            `;
          }).join('');

          svgOverlay = `<svg viewBox="0 0 ${pw} ${ph}" preserveAspectRatio="none">${panelRects}${boxRects}</svg>`;
        }

        const predHtml = predLines.length
          ? predLines.map((r, lIdx) => `
              <div class="line-item">
                <span style="color:var(--muted);font-size:0.75rem;min-width:1.4rem">#${lIdx+1}</span>
                <span class="spk-badge ${badgeClass(r.speaker)}">${escapeHtml(r.speaker)}</span>
                <span>${escapeHtml(r.text)}</span>
              </div>`).join('')
          : `<div class="empty-msg">[] (Empty page — no story text)</div>`;

        const goldHtml = goldLines !== null
          ? `<div class="transcript-section-title" style="margin-top:0.9rem">Gold Reference (${goldLines.length} lines)</div>` +
            (goldLines.length
              ? goldLines.map((r, lIdx) => `
                  <div class="line-item" style="opacity:0.82">
                    <span style="color:var(--muted);font-size:0.75rem;min-width:1.4rem">#${lIdx+1}</span>
                    <span class="spk-badge ${badgeClass(r.speaker || 'UNKNOWN')}">${escapeHtml(r.speaker || 'UNKNOWN')}</span>
                    <span>${escapeHtml(r.text)}</span>
                  </div>`).join('')
              : `<div class="empty-msg">[] (Gold empty page)</div>`)
          : '';

        const pageScoreBadge = (data.score && data.score.page_text_scores && data.score.page_text_scores[String(pageNum)] !== undefined)
          ? `<span style="color:var(--accent);font-size:0.8rem">Page Text Score: ${Number(data.score.page_text_scores[String(pageNum)]).toFixed(3)}</span>`
          : `<span style="color:var(--muted);font-size:0.8rem">${predLines.length} lines</span>`;

        return `
          <div class="page-card">
            <div class="page-header">
              <span>Page 0${pageNum}.png</span>
              ${pageScoreBadge}
            </div>
            <div class="img-wrap">
              <img src="/api/image?split=${encodeURIComponent(data.split)}&id=${encodeURIComponent(data.sequence_id)}&page=${pageNum}" alt="Page ${pageNum}" loading="lazy" />
              ${svgOverlay}
            </div>
            <div class="transcript-pane">
              <div class="transcript-section-title">Predicted Story Transcript (${predLines.length} lines)</div>
              ${predHtml}
              ${goldHtml}
            </div>
          </div>
        `;
      }).join('');
    }

    init();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/summary":
            self._send_json(
                {
                    "test_ids": list(STATE["test_seqs"].keys()),
                    "dev_ids": list(STATE["dev_seqs"].keys()),
                    "ablations": STATE["ablations"],
                    "weak_labels": STATE["manifest"].get("weak_labels", {}),
                    "macro_scores": STATE["scores"].get("macro", {}),
                }
            )
            return

        if path == "/api/sequence":
            seq_id = (qs.get("id") or [""])[0]
            if seq_id in STATE["test_seqs"]:
                seq = STATE["test_seqs"][seq_id]
                pred_pages = STATE["test_preds"].get(seq_id, [[], [], []])
            elif seq_id in STATE["dev_seqs"]:
                seq = STATE["dev_seqs"][seq_id]
                pred_pages = STATE["dev_preds"].get(seq_id, [[], [], []])
            else:
                self._send_json({"error": f"Unknown sequence_id {seq_id}"}, status=404)
                return

            # Load cached perception & predicted box states if available
            pipe: MangaTranscriberPipeline = STATE["pipeline"]
            perc_pages = []
            try:
                perc = pipe.extract_or_load_perception(seq, use_cache=True)
                _, p_boxes = pipe.predict_sequence_from_perception(perc)
                for idx, p_data in enumerate(perc["pages"]):
                    perc_pages.append(
                        {
                            "width": p_data["width"],
                            "height": p_data["height"],
                            "panels": p_data.get("panels", []),
                            "boxes": p_boxes[idx] if idx < len(p_boxes) else p_data.get("boxes", []),
                        }
                    )
            except Exception:
                perc_pages = []

            seq_score = STATE["scores"].get("sequences", {}).get(seq_id)
            self._send_json(
                {
                    "sequence_id": seq.id,
                    "split": seq.split,
                    "predicted_pages": pred_pages,
                    "gold_pages": seq.gold_pages,
                    "perception_pages": perc_pages,
                    "score": seq_score,
                }
            )
            return

        if path == "/api/image":
            split = (qs.get("split") or ["test"])[0]
            seq_id = (qs.get("id") or [""])[0]
            page = int((qs.get("page") or ["1"])[0])
            seq_map = STATE["test_seqs"] if split == "test" else STATE["dev_seqs"]
            if seq_id not in seq_map or not (1 <= page <= 3):
                self.send_error(404, "Image not found")
                return
            img_path = seq_map[seq_id].image_paths[page - 1]
            if not img_path.exists():
                self.send_error(404, "Image file missing")
                return
            data = img_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(img_path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/download_submission":
            sub_path = ROOT / "submission.jsonl"
            data = sub_path.read_bytes() if sub_path.exists() else b""
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="submission.jsonl"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")


class DualStackServer(ThreadingHTTPServer):
    address_family = __import__("socket").AF_INET6

    def server_bind(self) -> None:
        import socket

        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Manga Transcriber web dashboard.")
    parser.add_argument("--host", default="::")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load full-dev trained models into the pipeline so box overlays match T2 predictions
    full_dir = ROOT / "artifacts/models/full_dev"
    pipe: MangaTranscriberPipeline = STATE["pipeline"]
    if (full_dir / "story_clf.joblib").exists():
        from src.learn.order_rank import ReadingOrderRanker
        from src.learn.speaker_pair import SpeakerPairModel
        from src.learn.story_clf import StoryTextClassifier

        pipe.story_clf = StoryTextClassifier.load(full_dir / "story_clf.joblib")
        pipe.story_clf.story_prob_threshold = float(STATE["cfg"]["story_classifier"]["story_prob_threshold"])
        pipe.story_clf.narration_prob_threshold = float(STATE["cfg"]["story_classifier"]["narration_prob_threshold"])
        pipe.order_ranker = ReadingOrderRanker.load(full_dir / "order_ranker.joblib")
        pipe.speaker_pair_model = SpeakerPairModel.load(full_dir / "speaker_pair.joblib")

    try:
        server = DualStackServer(("::", args.port), DashboardHandler)
    except Exception:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Manga Transcriber Dashboard running at http://localhost:{args.port} and http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

