"""
Behaviour-cloning runs on the training dashboard: ``/bc`` (page) and ``/api/bc?run=<name>`` (data).

A BC run is a directory with a ``history.json`` (one entry per epoch, written by ``l2_NN/train.py``)
under ``l2_agent/checkpoints`` or the project ``archive``; its ``train.log`` (the copy of the console,
``train.py --save-dir``) gives the live tail. A run counts as running while a ``train.py`` process
with its ``--save-dir`` is alive.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = AGENT_ROOT / "checkpoints"
ARCHIVE = AGENT_ROOT.parent / "archive"
METRICS = ("loss_total", "loss_action", "loss_pointer", "loss_skill", "acc_action", "acc_pointer", "acc_skill")


def run_dirs() -> dict[str, Path]:
    """Run name -> directory, the most recently updated first (current runs before archived ones)."""
    found: list[tuple[str, Path]] = []
    for root, prefix in ((CHECKPOINTS, ""), (ARCHIVE, "archive/")):
        if root.exists():
            runs = [p for p in root.iterdir() if (p / "history.json").exists()]
            runs.sort(key=lambda p: (p / "history.json").stat().st_mtime, reverse=True)
            found += [(prefix + p.name, p) for p in runs]
    return dict(found)


def _running_dirs() -> set[str]:
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return set()
    names = set()
    for line in out.splitlines():
        if "train.py" in line and "--save-dir" in line and "train_rl" not in line:
            names.add(Path(line.split("--save-dir", 1)[1].split()[0]).name)
    return names


def _summary(name: str, path: Path, running: set[str]) -> dict:
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    best = min(history, key=lambda h: h.get("val_loss_total", float("inf"))) if history else {}
    return {"name": name, "epochs": len(history), "best_epoch": best.get("epoch"),
            "best": {m: best.get(f"val_{m}") for m in METRICS},
            "updated": (path / "history.json").stat().st_mtime, "running": path.name in running}


def payload(run_name: str | None) -> dict:
    dirs = run_dirs()
    if not dirs:
        return {"runs": [], "error": f"no BC run with a history.json under {CHECKPOINTS} or {ARCHIVE}"}
    running = _running_dirs()
    name = run_name if run_name in dirs else next(iter(dirs))
    path = dirs[name]
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    log = path / "train.log"
    tail: list[str] = []
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [line for line in lines if line.strip()][-40:]
    return {"now": time.time(), "runs": [_summary(n, p, running) for n, p in dirs.items()], "run": name,
            "summary": _summary(name, path, running),
            "history": [{"epoch": h["epoch"], "lr": h.get("lr"),
                         **{f"{split}_{m}": h.get(f"{split}_{m}") for split in ("train", "val") for m in METRICS}}
                        for h in history],
            "log": tail}


PAGE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BC training</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#6b7385; --line:#e3e6ec; --good:#1a7f4b;
          --train:#2f6fdb; --val:#c2410c; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029; --ink:#e7eaf0; --muted:#98a0b3; --line:#2b3240;
          --good:#3fbf7f; --train:#6ea2ff; --val:#ff9a5c; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:14px 20px; border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:2; }
  header h1 { font-size:17px; margin:0 12px 0 0; }
  header a { color:var(--muted); }
  select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--ink); max-width:100%; }
  .badge { padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid var(--line); color:var(--muted); }
  .badge.on { color:var(--good); border-color:var(--good); }
  main { padding:16px 20px 40px; max-width:1500px; margin:0 auto; }
  .tiles { display:flex; gap:8px; margin-bottom:16px; }
  .tile { flex:1 1 0; min-width:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 10px; }
  .tile .v { font-size:clamp(15px, 1.5vw, 22px); font-weight:650; font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tile .k { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(420px, 1fr)); gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; min-width:0; }
  .card h2 { font-size:14px; margin:0 0 6px; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted); margin-bottom:4px; }
  .legend i { display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:5px; }
  .plot { position:relative; }
  svg { width:100%; height:200px; display:block; }
  svg text { fill:var(--muted); font-size:10px; font-variant-numeric:tabular-nums; }
  .tip { position:absolute; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:5px 8px; font-size:12px;
         box-shadow:0 2px 8px rgba(0,0,0,.12); white-space:nowrap; display:none; z-index:1; }
  .tip b { font-variant-numeric:tabular-nums; }
  pre { margin:0; font:12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; white-space:pre-wrap; word-break:break-word; color:var(--ink); max-height:420px; overflow:auto; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { padding:4px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  tr.sel td { font-weight:650; }
  .scroll { overflow-x:auto; }
  .muted { color:var(--muted); font-size:12px; }
  @media (max-width:900px) { .tiles { flex-wrap:wrap; } .tile { flex:1 1 130px; } }
  @media (max-width:520px) { .grid { grid-template-columns:1fr; } header, main { padding-left:16px; padding-right:16px; } }
</style>
</head>
<body>
<header>
  <h1>BC training</h1>
  <label>run <select id="run"></select></label>
  <span id="status" class="badge">stopped</span>
  <span id="updated" class="muted"></span>
  <a href="/">← RL dashboard</a>
</header>
<main>
  <div class="tiles" id="tiles"></div>
  <div class="grid">
    <div class="card"><h2>Loss (total)</h2><div class="legend"><span><i style="background:var(--train)"></i>train</span><span><i style="background:var(--val)"></i>validation</span></div><div class="plot"><svg id="ch-loss"></svg><div class="tip"></div></div></div>
    <div class="card"><h2>Accuracy: action type</h2><div class="legend"><span><i style="background:var(--train)"></i>train</span><span><i style="background:var(--val)"></i>validation</span></div><div class="plot"><svg id="ch-act"></svg><div class="tip"></div></div></div>
    <div class="card"><h2>Accuracy: target monster (pointer)</h2><div class="legend"><span><i style="background:var(--train)"></i>train</span><span><i style="background:var(--val)"></i>validation</span></div><div class="plot"><svg id="ch-ptr"></svg><div class="tip"></div></div></div>
    <div class="card"><h2>Accuracy: skill slot</h2><div class="legend"><span><i style="background:var(--train)"></i>train</span><span><i style="background:var(--val)"></i>validation</span></div><div class="plot"><svg id="ch-skl"></svg><div class="tip"></div></div></div>
    <div class="card"><h2>All BC runs</h2><div class="scroll"><table id="runs"></table></div></div>
    <div class="card"><h2>train.log (last lines)</h2><pre id="log"></pre></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pct = v => v === null || v === undefined ? "–" : (100 * v).toFixed(1) + "%";
const num = (v, d = 3) => v === null || v === undefined ? "–" : Number(v).toFixed(d);
const params = new URLSearchParams(location.search);
let current = params.get("run");

function chart(id, rows, key, fmt) {
  const svg = $(id), tip = svg.parentElement.querySelector(".tip");
  const W = 600, H = 200, L = 46, R = 12, T = 10, B = 22;
  const series = [{k: "train_" + key, c: css("--train"), n: "train"}, {k: "val_" + key, c: css("--val"), n: "validation"}];
  const pts = rows.filter(r => series.some(s => r[s.k] !== null && r[s.k] !== undefined));
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!pts.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">no epochs yet</text>`; return; }
  const vals = pts.flatMap(r => series.map(s => r[s.k])).filter(v => v !== null && v !== undefined);
  let lo = Math.min(...vals), hi = Math.max(...vals); if (hi === lo) { hi += 0.01; lo -= 0.01; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const e0 = pts[0].epoch, e1 = pts[pts.length - 1].epoch;
  const x = e => L + (W - L - R) * (e - e0) / Math.max(e1 - e0, 1), y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, yy = y(v);
    g += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="${css('--line')}"/><text x="${L - 6}" y="${yy + 3}" text-anchor="end">${fmt(v)}</text>`;
  }
  const ticks = Math.min(6, e1 - e0 + 1);
  for (let i = 0; i < ticks; i++) { const e = Math.round(e0 + (e1 - e0) * i / Math.max(ticks - 1, 1)); g += `<text x="${x(e)}" y="${H - 6}" text-anchor="middle">${e}</text>`; }
  for (const s of series) {
    let d = "", pen = false;
    for (const r of pts) { const v = r[s.k]; if (v === null || v === undefined) { pen = false; continue; } d += `${pen ? "L" : "M"}${x(r.epoch).toFixed(1)},${y(v).toFixed(1)}`; pen = true; }
    g += `<path d="${d}" fill="none" stroke="${s.c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }
  g += `<line id="${id}-x" y1="${T}" y2="${H - B}" stroke="${css('--muted')}" stroke-dasharray="3 3" visibility="hidden"/>`;
  for (const s of series) g += `<circle class="dot" r="4" fill="${s.c}" stroke="${css('--card')}" stroke-width="2" visibility="hidden"/>`;
  g += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
  svg.innerHTML = g;
  const cross = svg.querySelector(`#${id}-x`), dots = svg.querySelectorAll(".dot"), hit = svg.querySelector("rect");
  hit.onmousemove = ev => {
    const box = svg.getBoundingClientRect(), mx = (ev.clientX - box.left) * W / box.width;
    const e = Math.round(e0 + (mx - L) / (W - L - R) * (e1 - e0));
    const r = pts.reduce((a, b) => Math.abs(b.epoch - e) < Math.abs(a.epoch - e) ? b : a);
    cross.setAttribute("x1", x(r.epoch)); cross.setAttribute("x2", x(r.epoch)); cross.setAttribute("visibility", "visible");
    series.forEach((s, i) => { const v = r[s.k]; dots[i].setAttribute("visibility", v === null || v === undefined ? "hidden" : "visible");
      if (v !== null && v !== undefined) { dots[i].setAttribute("cx", x(r.epoch)); dots[i].setAttribute("cy", y(v)); } });
    tip.innerHTML = `epoch <b>${r.epoch}</b>` + series.map(s => `<br><span style="color:${s.c}">■</span> ${s.n} <b>${fmt(r[s.k])}</b>`).join("");
    tip.style.display = "block";
    const px = (x(r.epoch) / W) * box.width, left = px + 12 + tip.offsetWidth > box.width ? px - tip.offsetWidth - 12 : px + 12;
    tip.style.left = left + "px"; tip.style.top = "8px";
  };
  hit.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tip.style.display = "none"; };
}

function tile(k, v) { return `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`; }

async function refresh() {
  let d;
  try { d = await (await fetch("/api/bc" + (current ? "?run=" + encodeURIComponent(current) : ""))).json(); }
  catch (e) { $("updated").textContent = "server not reachable"; return; }
  if (d.error) { $("updated").textContent = d.error; return; }
  current = d.run; params.set("run", current); history.replaceState(null, "", "?" + params);
  $("run").innerHTML = d.runs.map(r => `<option value="${esc(r.name)}"${r.name === d.run ? " selected" : ""}>${esc(r.name)}${r.running ? " (training)" : ""}</option>`).join("");
  const s = d.summary, last = d.history[d.history.length - 1] || {};
  $("status").textContent = s.running ? "training" : "stopped"; $("status").className = "badge" + (s.running ? " on" : "");
  const ago = Math.round((d.now - s.updated) / 60);
  $("updated").textContent = `last epoch ${ago < 1 ? "just now" : ago < 60 ? ago + " min ago" : Math.floor(ago / 60) + " h " + String(ago % 60).padStart(2, "0") + " min ago"}`;
  $("tiles").innerHTML = [
    tile("epochs", s.epochs), tile("best epoch", s.best_epoch ?? "–"), tile("best val loss", num(s.best.loss_total)),
    tile("val acc action (best)", pct(s.best.acc_action)), tile("val acc pointer (best)", pct(s.best.acc_pointer)),
    tile("val acc skill (best)", pct(s.best.acc_skill)), tile("learning rate", last.lr ? Number(last.lr).toExponential(2) : "–"),
  ].join("");
  chart("ch-loss", d.history, "loss_total", v => num(v, 2));
  chart("ch-act", d.history, "acc_action", pct);
  chart("ch-ptr", d.history, "acc_pointer", pct);
  chart("ch-skl", d.history, "acc_skill", pct);
  $("runs").innerHTML = "<tr><th>run</th><th>epochs</th><th>best epoch</th><th>val loss</th><th>action</th><th>pointer</th><th>skill</th></tr>"
    + d.runs.map(r => `<tr class="${r.name === d.run ? "sel" : ""}"><td>${esc(r.name)}${r.running ? " ●" : ""}</td><td>${r.epochs}</td><td>${r.best_epoch ?? "–"}</td>`
      + `<td>${num(r.best.loss_total)}</td><td>${pct(r.best.acc_action)}</td><td>${pct(r.best.acc_pointer)}</td><td>${pct(r.best.acc_skill)}</td></tr>`).join("");
  $("log").textContent = d.log.length ? d.log.join("\n") : "no train.log in this run";
}
$("run").onchange = e => { current = e.target.value; refresh(); };
refresh(); setInterval(refresh, 15000);
</script>
</body>
</html>
"""
