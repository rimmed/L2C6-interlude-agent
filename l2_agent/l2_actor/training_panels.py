"""
The skill panels the model is trained on: the fixed panel of every class (``class_panels.json``,
``l2_client/tools/data/class_panels.py``), slot by slot, the way BC and RL training and the live
client feed it to the network.

Served by the training dashboard as ``/panels`` (page) and ``/api/panels?class=<id>`` (data).
"""

from __future__ import annotations

from l2_client.tools.data import class_panels


def classes() -> list[dict]:
    """Every class with a panel: id, name, profession tier (0-3) and panel size."""
    out = [{"id": p.class_id, "name": p.name, "tier": min(len(p.chain) - 1, 3), "size": p.size}
           for p in class_panels.load().values()]
    return sorted(out, key=lambda c: (c["tier"], c["name"]))


def panel(class_id: int, icons: set[int]) -> dict:
    """One class's panel: its chain and every slot."""
    p = class_panels.panel(class_id)
    names = class_panels.load()
    return {"class": p.class_id, "name": p.name, "chain": [names[c].name for c in reversed(p.chain) if c in names],
            "slots": [{"id": s.skill_id, "name": s.name, "category": s.category, "pet": s.pet,
                       "learn": s.learn_level, "until": s.outgrown_at, "needs": list(s.needs),
                       "icon": s.skill_id in icons}
                      for s in p.slots]}


def payload(class_id: str | None, icons: set[int]) -> dict:
    listed = classes()
    wanted = int(class_id) if class_id and class_id.isdigit() and class_panels.panel(int(class_id)) else listed[0]["id"]
    return {"hash": class_panels.panels_hash(), "max_slots": class_panels.MAX_SLOTS, "classes": listed,
            "panel": panel(wanted, icons)}


PAGE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training panels</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#6b7385; --line:#e3e6ec; --slot:#2a2f3a; --empty:#e9ebf0;
          --accent:#2f6fdb; --attack:#d9534f; --debuff:#9a6dd7; --heal:#1a7f4b; --buff:#e08a1e; --summon:#2f6fdb; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029; --ink:#e7eaf0; --muted:#98a0b3; --line:#2b3240;
          --slot:#0d1015; --empty:#232935; --accent:#6ea2ff;
          --attack:#ff7b72; --debuff:#b892ff; --heal:#3fbf7f; --buff:#ffb44d; --summon:#6ea2ff; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; padding:12px 20px; border-bottom:1px solid var(--line);
           background:var(--card); position:sticky; top:0; z-index:2; }
  header h1 { font-size:17px; margin:0 8px 0 0; }
  header a { color:var(--muted); }
  select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--ink); max-width:100%; }
  .lvl { display:flex; align-items:center; gap:8px; }
  .lvl input { width:180px; accent-color:var(--accent); }
  .lvl b { font-variant-numeric:tabular-nums; min-width:2.2em; }
  main { padding:16px 20px 40px; max-width:1200px; margin:0 auto; }
  .note { color:var(--muted); margin:0 0 6px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12px; color:var(--muted); margin:10px 0 14px; }
  .legend i { display:inline-block; width:12px; height:4px; margin-right:5px; vertical-align:middle; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .card h3 { margin:0 0 10px; font-size:14px; display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }
  .muted { color:var(--muted); font-size:12px; font-weight:400; }
  .bar { display:grid; grid-template-columns:repeat(12, 64px); gap:5px; }
  .slot { position:relative; width:64px; height:84px; border-radius:6px; background:var(--slot); border-bottom:3px solid var(--c);
          display:flex; flex-direction:column; align-items:center; padding-top:5px; color:#e7eaf0; transition:opacity .15s; }
  .slot.empty { background:var(--empty); border-bottom-color:transparent; }
  .slot.pet { outline:2px dashed var(--summon); outline-offset:-2px; }
  .slot.locked { opacity:.28; }
  .slot.outgrown { opacity:.28; }
  .slot.outgrown img, .slot.outgrown .noico { filter:grayscale(1); }
  .slot.outgrown::after { content:""; position:absolute; left:6px; right:6px; top:50%; border-top:2px solid #d9534f; transform:rotate(-20deg); }
  .slot img, .slot .noico { width:36px; height:36px; border-radius:3px; }
  .slot .noico { background:#4a5160; display:flex; align-items:center; justify-content:center; font-size:10px; }
  .slot .n { position:absolute; top:1px; left:4px; font-size:9px; color:#9aa3b5; }
  .slot .name { font-size:9px; line-height:1.1; text-align:center; padding:2px 3px 0; max-height:2.2em; overflow:hidden; color:#cfd5e1; }
  .slot .tag { position:absolute; top:1px; right:3px; font-size:9px; color:#ffcf70; }
  .slot .lv { font-size:10px; color:#9aa3b5; margin-top:auto; padding-bottom:2px; }
  @media (max-width:860px) { .bar { grid-template-columns:repeat(6, 64px); } }
  @media (max-width:520px) { header, main { padding-left:16px; padding-right:16px; } .bar { grid-template-columns:repeat(4, 64px); } }
</style>
</head>
<body>
<header>
  <h1>Training panels</h1>
  <label>class <select id="cls"></select></label>
  <label class="lvl">level <input id="lv" type="range" min="1" max="80" value="80"><b id="lvn">80</b></label>
  <a href="/">← dashboard</a>
</header>
<main>
  <p class="note" id="chain"></p>
  <p class="note" id="info"></p>
  <div class="legend">
    <span><i style="background:var(--attack)"></i>attack</span><span><i style="background:var(--debuff)"></i>debuff</span>
    <span><i style="background:var(--heal)"></i>heal</span><span><i style="background:var(--buff)"></i>buff</span>
    <span><i style="background:var(--summon)"></i>summon</span>
    <span><i style="outline:2px dashed var(--summon);height:10px"></i>servitor's skill (valid while that servitor is out)</span>
    <span>faded - not learned yet at the chosen level</span>
    <span>crossed out - outgrown: an attack, debuff or summon learned before level 20 that stops growing before 40 leaves at 40 (buffs and heals stay)</span>
    <span><b style="color:#d9a440">BOW</b> - weapon / armor the skill needs (invalid with other equipment)</span>
  </div>
  <div class="card"><h3 id="title"></h3><div class="bar" id="bar"></div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const params = new URLSearchParams(location.search);
let data = null;

function render() {
  const level = +$("lv").value;
  $("lvn").textContent = level;
  const p = data.panel;
  const active = p.slots.filter(s => !s.pet && s.learn <= level && !(s.until && level >= s.until)).length;
  $("title").innerHTML = `${esc(p.name)} <span class="muted">${p.slots.length} of ${data.max_slots} slots · ${active} own skills on the panel at level ${level}</span>`;
  const pad = Math.max(12, Math.ceil(p.slots.length / 12) * 12);
  let html = p.slots.map((s, i) => {
    const locked = !s.pet && s.learn > level;
    const outgrown = !s.pet && s.until && level >= s.until;
    const needs = s.needs.length ? `<span class="tag">${esc(s.needs.join("/"))}</span>` : "";
    return `<div class="slot${s.pet ? " pet" : ""}${locked ? " locked" : ""}${outgrown ? " outgrown" : ""}" style="--c:var(--${s.category})"`
      + ` title="${esc(s.name)} · ${s.category}${s.pet ? " · servitor" : " · on the panel at levels " + s.learn + (s.until ? "–" + (s.until - 1) : "+")}${s.needs.length ? " · needs " + s.needs.join(" or ") : ""} · id ${s.id}">`
      + `<span class="n">${i}</span>${needs}`
      + (s.icon ? `<img src="/icons/${s.id}.png" alt="">` : `<span class="noico">${s.id}</span>`)
      + `<span class="name">${esc(s.name)}</span><span class="lv">${s.pet ? "pet" : s.until ? s.learn + "–" + (s.until - 1) : s.learn + "+"}</span></div>`;
  }).join("");
  for (let i = p.slots.length; i < pad; i++) html += `<div class="slot empty"><span class="n">${i}</span></div>`;
  $("bar").innerHTML = html;
}

async function load(classId) {
  const r = await fetch("/api/panels" + (classId ? "?class=" + classId : ""));
  data = await r.json();
  if (data.error) { $("info").textContent = data.error; return; }
  if (!$("cls").options.length) {
    $("cls").innerHTML = data.classes.map(c => `<option value="${c.id}">${c.tier} · ${esc(c.name)} (${c.size})</option>`).join("");
  }
  $("cls").value = data.panel.class;
  params.set("class", data.panel.class);
  history.replaceState(null, "", "?" + params);
  $("chain").textContent = data.panel.chain.join(" → ");
  $("info").textContent = `The fixed panel of the class (class_panels.json, ${data.hash}): BC, RL and the live agents feed the model exactly these slots. `
    + `A slot is valid only at the levels shown under it, `
    + `while the character knows the skill and its equipment allows it; otherwise it is an empty row. Move the level slider to see the panel grow and shed skills.`;
  render();
}
$("cls").onchange = e => load(e.target.value);
$("lv").oninput = () => { params.set("level", $("lv").value); history.replaceState(null, "", "?" + params); render(); };
if (params.get("level")) $("lv").value = params.get("level");
load(params.get("class"));
</script>
</body>
</html>
"""
