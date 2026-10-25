"""
The buffs the server gives the RL agents (``RLBuffs.java``), every possible variant per class and level.
The agent's summon (servitor or pet) gets the same set as its owner.

Mirrors the server: every life an agent gets one buff set picked with ``RLBuffSetWeights`` (``support``
= one of the ``RLBuffSupport.<class>`` lists, each equally likely; ``prophet`` = the Prophet's list); a buff marked ``:mage`` /
``:fighter`` goes to mage / fighter builds only (support classes pick a build with ``RLBuilds.<family>``,
undead hunting only from ``RLUndeadMinLevel``); a buff is given at the highest level whose magic level
the agent has reached and left out below its first one; a buff with the effect of one of the agent's
own active buffs is left out (it casts that itself); one buff per effect; at most ``MaxBuffAmount``
buffs and ``MaxDanceAmount`` dances / songs; an empty set (newbie buffs outside their levels) falls
back to the first other set. The agents also drink the shop buff potions of ``RLBuffPotions``.

Read from the repository's ``RLAgentEnvironment.ini`` / ``Player.ini`` (deployed as they are) and the
datapack. Served by the dashboard as ``/buffs`` and ``/api/buffs?class=<id>&level=<n>``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from l2_client.tools.data import class_panels

GAME = class_panels.DATAPACK.parent
INI = GAME / "config" / "Custom" / "RLAgentEnvironment.ini"
PLAYER_INI = GAME / "config" / "Player.ini"
#: build families of the support classes (``RLAgentManager.buildFamily``): family -> root classes
FAMILY_ROOTS = {"Prophet": (17,), "Elder": (15, 29, 42), "Shaman": (49,)}
BUILD_MAGE = {"undead_mage": True, "caster": True, "melee_heavy": False, "melee_light": False}
CONTINUOUS = {"A2", "A4", "DA2", "A3"}


def _ini(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _entries(value: str) -> list[dict]:
    """``id[:min-max][:mage|fighter]`` entries (``RLAgentEnvironmentConfig.parseBuffs``)."""
    out = []
    for part in filter(None, (p.strip() for p in value.split(","))):
        fields = part.split(":")
        entry = {"id": int(fields[0]), "min": 1, "max": 80, "only": None}
        for field in fields[1:]:
            if re.fullmatch(r"\d+-\d+", field):
                entry["min"], entry["max"] = map(int, field.split("-"))
            elif field in ("mage", "fighter"):
                entry["only"] = field
        out.append(entry)
    return out


@lru_cache(maxsize=1)
def _skills() -> dict[int, dict]:
    """skill id -> name, magic level per skill level, effect (abnormal type), operate type, debuff and dance flags."""
    out = {}
    for path in sorted((class_panels.DATAPACK / "stats" / "skills").glob("*.xml")):
        for skill in ET.parse(path).getroot().iter("skill"):
            levels = int(skill.get("levels", "1"))
            tables = {t.get("name"): (t.text or "").split() for t in skill.findall("table")}

            def value(tag: str, level: int, default: str = "") -> str:
                raw = (skill.findtext(tag) or default).strip()
                if raw.startswith("#"):
                    values = tables.get(raw, [])
                    return values[min(level, len(values)) - 1] if values else default
                return raw

            out[int(skill.get("id"))] = {
                "name": skill.get("name", ""), "levels": levels,
                "magic": [int(float(value("magicLevel", lv, "0") or 0)) for lv in range(1, levels + 1)],
                "abnormal": value("abnormalType", 1, "NONE") or "NONE", "operate": value("operateType", 1),
                "debuff": value("isDebuff", 1, "false") == "true", "dance": value("isMagic", 1, "0") == "3",
            }
    return out


@lru_cache(maxsize=1)
def config() -> dict:
    ini, player = _ini(INI), _ini(PLAYER_INI)
    weights = dict((k, int(v)) for k, v in (p.split(":") for p in ini.get("RLBuffSetWeights", "").split(";") if ":" in p))
    builds = {fam: dict((k, int(v)) for k, v in (p.split(":") for p in ini.get(f"RLBuilds.{fam}", "").split(";") if ":" in p))
              for fam in FAMILY_ROOTS}
    return {"weights": weights,
            "newbie": _entries(ini.get("RLBuffNewbie", "")),
            "support": {k[len("RLBuffSupport."):]: _entries(v) for k, v in ini.items() if k.startswith("RLBuffSupport.")},
            "dances_songs": _entries(ini.get("RLBuffDancesSongs", "")), "full": _entries(ini.get("RLBuffFull", "")),
            "potions": _entries(ini.get("RLBuffPotions", "")), "builds": builds,
            "undead_min": int(ini.get("RLUndeadMinLevel", "30")),
            "buff_slots": int(player.get("MaxBuffAmount", "20")), "dance_slots": int(player.get("MaxDanceAmount", "12"))}


@lru_cache(maxsize=1)
def _trees() -> dict:
    return class_panels._read_trees(class_panels.DATAPACK / "stats" / "players" / "skillTrees")


def _family(class_id: int) -> str | None:
    chain = class_panels._chain(_trees(), class_id)
    if 17 in chain:  # Prophet is checked first, it is a child of the Cleric
        return "Prophet"
    for family, roots in FAMILY_ROOTS.items():
        if family != "Prophet" and any(root in chain for root in roots):
            return family
    return None


def _own_effects(class_id: int, level: int) -> set[str]:
    """Effects of the agent's own active buffs at ``level`` (it learns every class skill it can)."""
    skills, trees, effects = _skills(), _trees(), set()
    for c in class_panels._chain(trees, class_id):
        for sid, (first, _) in trees[c][1].items():
            s = skills.get(sid)
            if s and first <= level and s["operate"] in CONTINUOUS and not s["debuff"] and s["abnormal"] != "NONE":
                effects.add(s["abnormal"])
    return effects


def _level_for(sid: int, level: int) -> int:
    s = _skills().get(sid)
    return max((i + 1 for i, magic in enumerate(s["magic"]) if magic <= level), default=0) if s else 0


def _build(entries: list[dict], level: int, mage: bool, own: set[str]) -> list[dict]:
    cfg, skills = config(), _skills()
    taken, out = set(own), []
    buffs, dances = cfg["buff_slots"], cfg["dance_slots"]
    for e in entries:
        if not (e["min"] <= level <= e["max"]) or (e["only"] == "mage" and not mage) or (e["only"] == "fighter" and mage):
            continue
        skill_level = _level_for(e["id"], level)
        s = skills.get(e["id"])
        if not skill_level or not s or s["abnormal"] in taken:
            continue
        taken.add(s["abnormal"])
        if s["dance"]:
            dances -= 1
            if dances < 0:
                continue
        else:
            buffs -= 1
            if buffs < 0:
                continue
        out.append({"id": e["id"], "name": s["name"], "level": skill_level, "dance": s["dance"]})
    return out


def variants(class_id: int, level: int) -> list[dict]:
    """Every buff set an agent of the class can get at ``level``, with its probability."""
    cfg = config()
    family = _family(class_id)
    if family:
        builds = {b: w for b, w in cfg["builds"][family].items() if b != "undead_mage" or level >= cfg["undead_min"]}
    else:
        mystic = 10 in class_panels._chain(_trees(), class_id) or any(r in class_panels._chain(_trees(), class_id) for r in (25, 38, 49))
        builds = {"mage" if mystic else "fighter": 1}
    total_build = sum(builds.values()) or 1
    total_set = sum(cfg["weights"].values()) or 1
    own = _own_effects(class_id, level)
    merged: dict[tuple, dict] = {}
    for build, bw in builds.items():
        mage = BUILD_MAGE.get(build, build == "mage")
        for set_name, sw in cfg["weights"].items():
            options = (list(cfg["support"].items()) if set_name == "support"
                       else [(None, cfg["support"].get("Prophet", []))] if set_name == "prophet"
                       else [(None, cfg.get(set_name, []))])
            for support, entries in options:
                p = (bw / total_build) * (sw / total_set) / len(options)
                name, buffs = set_name, _build(entries, level, mage, own) if set_name != "none" else []
                if not buffs and set_name not in ("none",):
                    # the fallback: the first set after none / newbie, i.e. a random support class
                    fallback = [k for k in cfg["weights"] if k not in ("none", "newbie")][0]
                    sub = list(cfg["support"].items()) if fallback == "support" else [(None, cfg.get(fallback, []))]
                    for fs, fe in sub:
                        fb = _build(fe, level, mage, own)
                        key = (f"{set_name} → {fallback}" + (f" ({fs})" if fs else ""), "mage" if mage else "fighter", tuple(b["id"] for b in fb))
                        item = merged.setdefault(key, {"set": key[0], "build": key[1], "p": 0.0, "buffs": fb})
                        item["p"] += p / len(sub)
                    continue
                label = name + (f" ({support})" if support else "")
                key = (label, "mage" if mage else "fighter", tuple(b["id"] for b in buffs))
                item = merged.setdefault(key, {"set": label, "build": key[1], "p": 0.0, "buffs": buffs})
                item["p"] += p
    out = sorted(merged.values(), key=lambda v: -v["p"])
    for v in out:
        v["p"] = round(v["p"], 4)
        v["buff_count"] = sum(not b["dance"] for b in v["buffs"])
        v["dance_count"] = sum(b["dance"] for b in v["buffs"])
    return out


def potions(class_id: int, level: int) -> list[dict]:
    """Shop buff potions the agent drinks (``RLBuffPotions``) for its mage / fighter build(s)."""
    names = {}
    for path in (class_panels.DATAPACK / "stats" / "items").glob("*.xml"):
        for item in ET.parse(path).getroot().iter("item"):
            names[int(item.get("id"))] = item.get("name")
    out = []
    for e in config()["potions"]:
        if e["min"] <= level <= e["max"]:
            out.append({"id": e["id"], "name": names.get(e["id"], str(e["id"])), "only": e["only"]})
    return out


def payload(class_id: str | None, level: str | None, icons: set[int]) -> dict:
    panels = class_panels.load()
    classes = sorted(({"id": c, "name": p.name, "tier": min(len(p.chain) - 1, 3)} for c, p in panels.items()),
                     key=lambda c: (c["tier"], c["name"]))
    cid = int(class_id) if class_id and class_id.isdigit() and int(class_id) in panels else classes[0]["id"]
    lv = min(max(int(level), 1), 80) if level and level.isdigit() else 40
    vs = variants(cid, lv)
    for v in vs:
        for b in v["buffs"]:
            b["icon"] = b["id"] in icons
    cfg = config()
    return {"classes": classes, "class": cid, "name": panels[cid].name, "level": lv, "family": _family(cid),
            "variants": vs, "potions": potions(cid, lv), "own": sorted(_own_effects(cid, lv)),
            "weights": cfg["weights"], "slots": {"buffs": cfg["buff_slots"], "dances": cfg["dance_slots"]}}


PAGE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent buffs</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#6b7385; --line:#e3e6ec; --accent:#2f6fdb; --dance:#9a6dd7; --bar:#2f6fdb; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029; --ink:#e7eaf0; --muted:#98a0b3; --line:#2b3240; --accent:#6ea2ff; --dance:#b892ff; --bar:#6ea2ff; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; padding:12px 20px; border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:2; }
  header h1 { font-size:17px; margin:0 8px 0 0; }
  header a { color:var(--muted); }
  select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--ink); max-width:100%; }
  .lvl { display:flex; align-items:center; gap:8px; } .lvl input { width:200px; accent-color:var(--accent); } .lvl b { min-width:2em; font-variant-numeric:tabular-nums; }
  main { padding:16px 20px 40px; max-width:1400px; margin:0 auto; }
  .note { color:var(--muted); margin:0 0 10px; font-size:13px; }
  .cols { display:grid; grid-template-columns:minmax(0, 2fr) minmax(280px, 1fr); gap:14px; align-items:start; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin-bottom:10px; }
  .card h3 { margin:0 0 8px; font-size:14px; display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }
  .card h3 .p { font-variant-numeric:tabular-nums; color:var(--accent); }
  .muted { color:var(--muted); font-size:12px; font-weight:400; }
  .chips { display:flex; flex-wrap:wrap; gap:5px 8px; }
  .chip { display:inline-flex; align-items:center; gap:5px; padding:2px 8px 2px 2px; border-radius:6px; border:1px solid var(--line); font-size:12px; }
  .chip.dance { border-color:var(--dance); }
  .chip img, .chip .noico { width:22px; height:22px; border-radius:3px; } .chip .noico { background:var(--line); }
  .chip .lv { color:var(--muted); }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  td, th { padding:4px 6px; border-bottom:1px solid var(--line); text-align:left; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .bar { height:8px; border-radius:4px; background:var(--bar); }
  h2 { font-size:14px; margin:0 0 8px; }
  @media (max-width:900px) { .cols { grid-template-columns:1fr; } }
  @media (max-width:520px) { header, main { padding-left:16px; padding-right:16px; } .lvl input { width:140px; } }
</style>
</head>
<body>
<header>
  <h1>Agent buffs</h1>
  <label>class <select id="cls"></select></label>
  <label class="lvl">level <input id="lv" type="range" min="1" max="80" value="40"><b id="lvn">40</b></label>
  <a href="/">← dashboard</a>
</header>
<main>
  <p class="note" id="info"></p>
  <div class="cols">
    <div id="variants"></div>
    <div>
      <div class="card"><h2>Chance to have each buff at this level</h2><table id="per-buff"></table></div>
      <div class="card"><h2>Buff potions the agent drinks</h2><div class="chips" id="potions"></div></div>
      <div class="card"><h2>Own buff effects (left out of the sets)</h2><div class="muted" id="own"></div></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pct = p => (100 * p).toFixed(p < 0.01 ? 1 : 0) + "%";
const params = new URLSearchParams(location.search);
let timer = null;
function chip(b) {
  return `<span class="chip${b.dance ? " dance" : ""}" title="${esc(b.name)} Lv ${b.level} · id ${b.id}${b.dance ? " · dance/song" : ""}">`
    + (b.icon ? `<img src="/icons/${b.id}.png" alt="">` : `<span class="noico"></span>`) + `${esc(b.name)} <span class="lv">${b.level}</span></span>`;
}
async function load() {
  const q = new URLSearchParams({level: $("lv").value}); if ($("cls").value) q.set("class", $("cls").value); else if (params.get("class")) q.set("class", params.get("class"));
  const d = await (await fetch("/api/buffs?" + q)).json();
  if (d.error) { $("info").textContent = d.error; return; }
  if (!$("cls").options.length) $("cls").innerHTML = d.classes.map(c => `<option value="${c.id}">${c.tier} · ${esc(c.name)}</option>`).join("");
  $("cls").value = d.class; params.set("class", d.class); params.set("level", d.level); history.replaceState(null, "", "?" + params);
  $("info").textContent = `${d.name}, level ${d.level}${d.family ? ` (${d.family} family: the build decides mage or fighter buffs)` : ""}. `
    + `One set per life, picked with ${Object.entries(d.weights).map(([k, w]) => `${k} ${w}`).join(", ")}; at most ${d.slots.buffs} buffs and ${d.slots.dances} dances / songs. `
    + `${d.variants.length} variants at this level. The agent's summon (servitor or pet) gets the same set as the agent.`;
  $("variants").innerHTML = d.variants.map(v => `<div class="card"><h3><span class="p">${pct(v.p)}</span> ${esc(v.set)} <span class="muted">${v.build} · ${v.buff_count} buffs · ${v.dance_count} dances / songs</span></h3>`
    + `<div class="chips">${v.buffs.map(chip).join("") || '<span class="muted">no buffs</span>'}</div></div>`).join("");
  const per = {};
  for (const v of d.variants) for (const b of v.buffs) { per[b.id] = per[b.id] || {b, p: 0}; per[b.id].p += v.p; }
  const rows = Object.values(per).sort((a, b) => b.p - a.p);
  $("per-buff").innerHTML = "<tr><th>buff</th><th>chance</th><th style='width:40%'></th></tr>" + rows.map(r => `<tr><td>${chip(r.b)}</td><td class="num">${pct(r.p)}</td><td><div class="bar" style="width:${(100 * r.p).toFixed(1)}%"></div></td></tr>`).join("");
  $("potions").innerHTML = d.potions.map(p => `<span class="chip"><span class="noico"></span>${esc(p.name)}${p.only ? ` <span class="lv">${p.only}</span>` : ""}</span>`).join("") || '<span class="muted">none</span>';
  $("own").textContent = d.own.length ? d.own.join(", ") : "none";
}
$("cls").onchange = load;
$("lv").oninput = () => { $("lvn").textContent = $("lv").value; clearTimeout(timer); timer = setTimeout(load, 150); };
if (params.get("level")) { $("lv").value = params.get("level"); $("lvn").textContent = params.get("level"); }
load();
</script>
</body>
</html>
"""
