#!/usr/bin/env python3
"""Generate self-contained HTML demo pages from demo data.

Reads offline/demo_data/{benchmark}.json files produced by generate_demo_data.py,
resizes and base64-encodes images, and outputs self-contained HTML files.

Usage:
    python scripts/generate_demo_html.py              # all benchmarks
    python scripts/generate_demo_html.py --benchmark library_cards

Output:
    offline/demo_{benchmark}.html  — one page per benchmark
    offline/demo_index.html        — index page linking them all

Dependencies: Pillow (for image resizing). Falls back to full-size if unavailable.
"""

import argparse
import base64
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = PROJECT_ROOT / "offline" / "demo_data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "demo"

MAX_IMAGE_WIDTH = 500  # px — images will take ~1/3 of screen

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

try:
    from PIL import Image

    def resize_and_encode(image_path: str) -> str:
        """Resize image to MAX_IMAGE_WIDTH and return base64 JPEG data URI."""
        img = Image.open(image_path)
        if img.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / img.width
            new_size = (MAX_IMAGE_WIDTH, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

except ImportError:
    logger.warning("Pillow not available — images will be embedded at full resolution")

    def resize_and_encode(image_path: str) -> str:
        """Encode image as base64 without resizing (Pillow fallback)."""
        path = Path(image_path)
        suffix = path.suffix.lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix.lstrip("."), "jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

# The CSS + JS template is benchmark-agnostic. Data is injected as inline JSON.
# The JS renders a generic recursive diff view of any JSON structure.

BENCHMARK_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; font-size: 13px; }}

  /* Header */
  .header {{ background: #16213e; padding: 10px 20px; display: flex; align-items: center; gap: 14px; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }}
  .header h1 {{ font-size: 15px; font-weight: 600; color: #e94560; white-space: nowrap; }}
  .header a {{ color: #888; text-decoration: none; font-size: 12px; }}
  .header a:hover {{ color: #e94560; }}
  .header select {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483; padding: 5px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }}
  .header .spacer {{ flex: 1; }}

  /* Score bar — shows all optimizer scores at a glance */
  .score-bar {{ display: flex; gap: 6px; padding: 8px 20px; background: #111; border-bottom: 1px solid #0f3460; flex-wrap: wrap; align-items: center; }}
  .score-chip {{ padding: 4px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; border: 2px solid transparent; transition: border-color 0.15s; }}
  .score-chip:hover {{ border-color: #533483; }}
  .score-chip.active {{ border-color: #e94560; }}
  .score-chip .chip-name {{ color: #888; margin-right: 6px; }}
  .score-chip .chip-val {{ font-weight: 700; }}

  .good {{ color: #2ecc71; }}
  .mid {{ color: #f39c12; }}
  .bad {{ color: #e94560; }}
  .chip-bg {{ background: #16213e; }}

  /* Main layout */
  .main {{ display: flex; height: calc(100vh - 90px); }}
  .image-panel {{ width: 33%; min-width: 200px; max-width: 500px; overflow: auto; border-right: 1px solid #0f3460; background: #111; }}
  .image-panel img {{ width: 100%; display: block; }}
  .image-panel .page-sep {{ text-align: center; font-size: 10px; color: #555; padding: 2px 0; background: #0a0a1a; }}
  .compare-panel {{ flex: 1; overflow: auto; padding: 14px; }}

  /* Two-column GT vs Prediction */
  .columns {{ display: flex; gap: 12px; }}
  .col {{ flex: 1; min-width: 0; }}
  .col-header {{ text-align: center; font-weight: 700; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #666; padding: 4px 8px; margin-bottom: 8px; border-bottom: 1px solid #333; }}

  /* JSON tree rendering */
  .json-block {{ background: #111; border-radius: 6px; padding: 10px; margin-bottom: 10px; border: 1px solid #1a1a3e; }}
  .json-block.has-errors {{ border-color: #e94560; }}
  .json-block.perfect {{ border-color: #2ecc71; }}
  .json-key {{ color: #888; font-size: 11px; }}
  .json-val {{ word-break: break-word; }}
  .json-field {{ padding: 3px 0; display: flex; gap: 6px; }}
  .json-field .fk {{ min-width: 120px; flex-shrink: 0; color: #666; font-size: 11px; }}
  .json-field .fv {{ flex: 1; word-break: break-word; }}
  .fv.match {{ color: #2ecc71; }}
  .fv.partial {{ color: #f39c12; }}
  .fv.wrong {{ color: #e94560; }}
  .fv.missing {{ color: #e67e22; font-style: italic; }}
  .fv.extra {{ color: #9b59b6; }}
  .fv.null {{ color: #444; font-style: italic; }}
  .section-label {{ font-weight: 700; font-size: 12px; color: #aaa; padding: 6px 0 4px; border-bottom: 1px solid #222; margin-bottom: 6px; }}
  .list-idx {{ color: #533483; font-size: 10px; font-weight: 600; }}

  /* Legend */
  .legend {{ display: flex; gap: 14px; font-size: 10px; align-items: center; padding: 6px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

  /* Summary card */
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 14px; }}
  .summary-card {{ background: #16213e; border-radius: 6px; padding: 10px; text-align: center; }}
  .summary-card .label {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .summary-card .value {{ font-size: 22px; font-weight: 700; margin: 4px 0; }}
  .summary-card .sub {{ font-size: 10px; color: #555; }}
</style>
</head>
<body>

<div class="header">
  <a href="demo_index.html">&larr; All benchmarks</a>
  <h1>{title}</h1>
  <select id="imageSelect"></select>
  <div class="spacer"></div>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div> Match (&ge;0.92)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div> Partial (0.5&ndash;0.92)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#e94560"></div> Wrong (&lt;0.5)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#e67e22"></div> Missing from prediction</div>
  <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div> Extra in prediction</div>
</div>

<div class="score-bar" id="scoreBar"></div>

<div class="main">
  <div class="image-panel" id="imagePanel"></div>
  <div class="compare-panel" id="comparePanel"></div>
</div>

<script>
// --- Injected data ---
const DATA = {data_json};
const METRIC_KEY = {metric_key_json};

// --- State ---
let currentImageIdx = 0;
let currentOptIdx = 0;

// --- Utilities ---
function scoreClass(s) {{
  if (s >= 0.92) return 'match';
  if (s >= 0.5) return 'partial';
  return 'wrong';
}}
function scoreColor(s) {{
  if (s >= 0.8) return 'good';
  if (s >= 0.4) return 'mid';
  return 'bad';
}}
function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}
function formatVal(v) {{
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'object') return JSON.stringify(v, null, 1);
  return String(v);
}}

// Simple Levenshtein
function levenshtein(a, b) {{
  const m = a.length, n = b.length;
  const dp = Array.from({{length: m+1}}, () => Array(n+1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
  return dp[m][n];
}}
function fuzzyScore(a, b) {{
  if (a === b) return 1.0;
  if (a == null || b == null) return 0.0;
  const sa = String(a), sb = String(b);
  if (sa === sb) return 1.0;
  const len = Math.max(sa.length, sb.length);
  if (len === 0) return 1.0;
  return (len - levenshtein(sa, sb)) / len;
}}

// --- Recursive JSON diff rendering ---
// Returns HTML for a field tree, comparing pred against gt
function renderFieldsCompare(pred, gt, prefix) {{
  let html = '';
  if (gt === null || gt === undefined) gt = {{}};
  if (pred === null || pred === undefined) pred = {{}};

  // Collect all keys from both
  const allKeys = new Set();
  if (typeof gt === 'object' && !Array.isArray(gt))
    Object.keys(gt).forEach(k => allKeys.add(k));
  if (typeof pred === 'object' && !Array.isArray(pred))
    Object.keys(pred).forEach(k => allKeys.add(k));

  // Handle arrays
  if (Array.isArray(gt) || Array.isArray(pred)) {{
    const gtArr = Array.isArray(gt) ? gt : [];
    const predArr = Array.isArray(pred) ? pred : [];
    const maxLen = Math.max(gtArr.length, predArr.length);
    for (let i = 0; i < maxLen; i++) {{
      const gv = i < gtArr.length ? gtArr[i] : undefined;
      const pv = i < predArr.length ? predArr[i] : undefined;
      const idxLabel = prefix ? `${{prefix}}[${{i}}]` : `[${{i}}]`;
      if (typeof gv === 'object' && gv !== null && !Array.isArray(gv)) {{
        html += `<div class="section-label"><span class="list-idx">#${{i}}</span></div>`;
        html += renderFieldsCompare(pv, gv, idxLabel);
      }} else {{
        const score = fuzzyScore(pv, gv);
        const cls = pv === undefined ? 'missing' : gv === undefined ? 'extra' : scoreClass(score);
        html += `<div class="json-field"><span class="fk">${{escHtml(idxLabel)}}</span><span class="fv ${{cls}}">${{escHtml(formatVal(pv))}}</span></div>`;
      }}
    }}
    return html;
  }}

  for (const key of allKeys) {{
    const fullKey = prefix ? `${{prefix}}.${{key}}` : key;
    const gv = gt[key];
    const pv = pred[key];

    if (typeof gv === 'object' && gv !== null) {{
      html += `<div class="section-label">${{escHtml(key)}}</div>`;
      html += renderFieldsCompare(pv, gv, fullKey);
    }} else {{
      const pvFmt = pv === undefined ? null : pv;
      const gvFmt = gv === undefined ? null : gv;
      let cls;
      if (pv === undefined && gv !== undefined && gv !== null) cls = 'missing';
      else if (gv === undefined && pv !== undefined && pv !== null) cls = 'extra';
      else if (pv === null || pv === undefined) cls = 'null';
      else {{
        const score = fuzzyScore(pv, gv);
        cls = scoreClass(score);
      }}
      html += `<div class="json-field"><span class="fk">${{escHtml(key)}}</span><span class="fv ${{cls}}">${{escHtml(formatVal(pvFmt))}}</span></div>`;
    }}
  }}
  return html;
}}

// Render GT side (no diff coloring, just neutral display)
function renderFieldsGT(obj, prefix) {{
  let html = '';
  if (obj === null || obj === undefined) return '<span class="fv null">null</span>';

  if (Array.isArray(obj)) {{
    for (let i = 0; i < obj.length; i++) {{
      const idxLabel = prefix ? `${{prefix}}[${{i}}]` : `[${{i}}]`;
      if (typeof obj[i] === 'object' && obj[i] !== null && !Array.isArray(obj[i])) {{
        html += `<div class="section-label"><span class="list-idx">#${{i}}</span></div>`;
        html += renderFieldsGT(obj[i], idxLabel);
      }} else {{
        html += `<div class="json-field"><span class="fk">${{escHtml(idxLabel)}}</span><span class="fv">${{escHtml(formatVal(obj[i]))}}</span></div>`;
      }}
    }}
    return html;
  }}

  if (typeof obj === 'object') {{
    for (const [key, val] of Object.entries(obj)) {{
      const fullKey = prefix ? `${{prefix}}.${{key}}` : key;
      if (typeof val === 'object' && val !== null) {{
        html += `<div class="section-label">${{escHtml(key)}}</div>`;
        html += renderFieldsGT(val, fullKey);
      }} else {{
        const cls = (val === null || val === undefined) ? 'null' : '';
        html += `<div class="json-field"><span class="fk">${{escHtml(key)}}</span><span class="fv ${{cls}}">${{escHtml(formatVal(val))}}</span></div>`;
      }}
    }}
    return html;
  }}

  return `<span class="fv">${{escHtml(formatVal(obj))}}</span>`;
}}

// --- Main render ---
function render() {{
  const img = DATA.images[currentImageIdx];
  const opt = img.optimizers[currentOptIdx];
  const metricVal = opt.scores[METRIC_KEY] ?? 0;

  // Images
  const imgPanel = document.getElementById('imagePanel');
  imgPanel.innerHTML = '';
  img.image_data.forEach((src, i) => {{
    const el = document.createElement('img');
    el.src = src;
    el.alt = `Page ${{i+1}}`;
    imgPanel.appendChild(el);
    if (img.image_data.length > 1) {{
      const sep = document.createElement('div');
      sep.className = 'page-sep';
      sep.textContent = `Page ${{i+1}} of ${{img.image_data.length}}`;
      imgPanel.appendChild(sep);
    }}
  }});

  // Score bar chips
  const bar = document.getElementById('scoreBar');
  bar.innerHTML = '';
  img.optimizers.forEach((o, idx) => {{
    const val = o.scores[METRIC_KEY] ?? 0;
    const chip = document.createElement('div');
    chip.className = `score-chip chip-bg${{idx === currentOptIdx ? ' active' : ''}}`;
    chip.innerHTML = `<span class="chip-name">${{escHtml(o.name)}}</span><span class="chip-val ${{scoreColor(val)}}">${{val.toFixed(4)}}</span>`;
    chip.onclick = () => {{ currentOptIdx = idx; render(); }};
    bar.appendChild(chip);
  }});

  // Compare panel
  const panel = document.getElementById('comparePanel');
  panel.innerHTML = '';

  // Summary cards
  const summaryHtml = `<div class="summary">
    <div class="summary-card">
      <div class="label">${{METRIC_KEY.replace('_', ' ')}}</div>
      <div class="value ${{scoreColor(metricVal)}}">${{metricVal.toFixed(4)}}</div>
      <div class="sub">${{opt.name}}</div>
    </div>
    ${{opt.scores.precision !== undefined ? `
    <div class="summary-card">
      <div class="label">Precision</div>
      <div class="value ${{scoreColor(opt.scores.precision)}}">${{opt.scores.precision.toFixed(3)}}</div>
    </div>
    <div class="summary-card">
      <div class="label">Recall</div>
      <div class="value ${{scoreColor(opt.scores.recall)}}">${{opt.scores.recall.toFixed(3)}}</div>
    </div>` : ''}}
  </div>`;
  panel.innerHTML += summaryHtml;

  // Two-column diff
  const gt = img.ground_truth;
  const pred = opt.prediction;

  panel.innerHTML += `
    <div class="columns">
      <div class="col">
        <div class="col-header">Ground Truth</div>
        <div class="json-block">${{renderFieldsGT(gt, '')}}</div>
      </div>
      <div class="col">
        <div class="col-header">Prediction &mdash; ${{escHtml(opt.name)}}</div>
        <div class="json-block">${{renderFieldsCompare(pred, gt, '')}}</div>
      </div>
    </div>
  `;
}}

// --- Init ---
function init() {{
  const sel = document.getElementById('imageSelect');
  DATA.images.forEach((img, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = img.id;
    sel.appendChild(opt);
  }});
  sel.onchange = () => {{
    currentImageIdx = parseInt(sel.value);
    currentOptIdx = 0;
    render();
  }};
  render();
}}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


INDEX_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HumBench Demo — Optimizer Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 40px; }}

  h1 {{ font-size: 22px; color: #e94560; margin-bottom: 8px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 32px; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}

  .card {{ background: #16213e; border-radius: 10px; padding: 20px; border: 1px solid #0f3460; transition: border-color 0.2s, transform 0.15s; cursor: pointer; text-decoration: none; color: inherit; display: block; }}
  .card:hover {{ border-color: #e94560; transform: translateY(-2px); }}
  .card h2 {{ font-size: 16px; color: #e94560; margin-bottom: 10px; }}
  .card .metric {{ font-size: 11px; color: #888; margin-bottom: 6px; }}
  .card .scores {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .card .score-pill {{ background: #0f3460; padding: 3px 8px; border-radius: 4px; font-size: 11px; }}
  .card .score-pill .name {{ color: #888; }}
  .card .score-pill .val {{ font-weight: 700; }}

  .good {{ color: #2ecc71; }}
  .mid {{ color: #f39c12; }}
  .bad {{ color: #e94560; }}

  .card .images-count {{ font-size: 11px; color: #555; margin-top: 8px; }}

  .footer {{ margin-top: 40px; font-size: 11px; color: #444; text-align: center; }}
</style>
</head>
<body>
  <h1>HumBench — Optimizer Comparison Demo</h1>
  <p class="subtitle">Interactive visualization of ground truth vs. optimizer predictions across RISE HumBench benchmarks. Select a benchmark to explore.</p>

  <div class="grid">
    {cards}
  </div>

  <div class="footer">Generated from DSPy optimization pipeline results. Model: Gemini 2.0 Flash.</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def generate_benchmark_html(benchmark: str):
    """Generate HTML for one benchmark from its demo data JSON."""
    data_path = DEMO_DATA_DIR / f"{benchmark}.json"
    if not data_path.exists():
        logger.warning(f"No demo data for {benchmark} at {data_path}, skipping")
        return None

    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"Generating HTML for {data['display_name']} ({len(data['images'])} images)")

    # Resize and base64-encode images
    for img in data["images"]:
        img["image_data"] = []
        for img_path in img["image_paths"]:
            p = Path(img_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                img["image_data"].append(resize_and_encode(str(p)))
                logger.info(f"  Encoded {p.name}")
            else:
                logger.warning(f"  Image not found: {p}")
                img["image_data"].append("")
        # Remove raw paths from embedded data (not needed in HTML)
        del img["image_paths"]

    # Build HTML
    title = f"{data['display_name']} — Optimizer Comparison"
    html = BENCHMARK_HTML_TEMPLATE.format(
        title=title,
        data_json=json.dumps(data, ensure_ascii=False),
        metric_key_json=json.dumps(data["metric_key"]),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"demo_{benchmark}.html"
    with open(out_path, "w") as f:
        f.write(html)

    logger.info(f"  Saved to {out_path}")
    return data


def generate_index_html(all_data: dict[str, dict]):
    """Generate the index page linking all benchmark pages."""
    cards_html = ""
    for benchmark, data in all_data.items():
        if data is None:
            continue

        # Compute summary: for each optimizer, average the metric across images
        metric_key = data["metric_key"]
        opt_names = [o["name"] for o in data["images"][0]["optimizers"]] if data["images"] else []
        opt_avgs = {}
        for opt_idx, name in enumerate(opt_names):
            vals = []
            for img in data["images"]:
                if opt_idx < len(img["optimizers"]):
                    vals.append(img["optimizers"][opt_idx]["scores"].get(metric_key, 0))
            opt_avgs[name] = sum(vals) / len(vals) if vals else 0

        def _color(v):
            if v >= 0.8:
                return "good"
            if v >= 0.4:
                return "mid"
            return "bad"

        pills = "".join(
            f'<div class="score-pill"><span class="name">{name}</span> '
            f'<span class="val {_color(avg)}">{avg:.3f}</span></div>'
            for name, avg in opt_avgs.items()
        )

        n_images = len(data["images"])
        cards_html += f"""
    <a class="card" href="demo_{benchmark}.html">
      <h2>{data['display_name']}</h2>
      <div class="metric">Primary metric: {metric_key}</div>
      <div class="scores">{pills}</div>
      <div class="images-count">{n_images} sample image{'s' if n_images != 1 else ''}</div>
    </a>
"""

    html = INDEX_HTML_TEMPLATE.format(cards=cards_html)
    out_path = OUTPUT_DIR / "demo_index.html"
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Saved index to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate demo HTML from demo data")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Single benchmark to generate (default: all available)")
    args = parser.parse_args()

    benchmarks = ["library_cards", "bibliographic_data", "personnel_cards", "business_letters"]
    if args.benchmark:
        benchmarks = [args.benchmark]

    all_data = {}
    for bench in benchmarks:
        data = generate_benchmark_html(bench)
        if data is not None:
            all_data[bench] = data

    if all_data:
        generate_index_html(all_data)
    else:
        logger.error("No demo data found. Run generate_demo_data.py first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
