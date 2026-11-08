"""
Skill panels per class and level: what the combat panel of an agent of a class holds at every level.

The panel is built the way the agents build it (``skill_data.combat_panel``): the class's learnable
skills of the attack / debuff / heal / buff / summon categories at the highest level the character
level allows; the passive skills of that level are listed below it, from the player skill trees of the datapack along the class's profession chain (the
trees' parent classes - ``classList.xml`` has a few wrong ones). Every skill carries the weapon /
armor it needs (``<using kind>``), which the agents check against their equipment.

Served by the training dashboard as ``/skills`` (page) and ``/api/skillpanels?class=<id>`` (data).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .report import CLASS_LIST, DATAPACK_SKILLS

SKILL_TREES = DATAPACK_SKILLS.parent / "players" / "skillTrees"
MAX_LEVEL = 80


class SkillPanels:
    """Class skill trees of the datapack, loaded once."""

    def __init__(self, icons: set[int]):
        from l2_client.tools.data import skill_data

        if DATAPACK_SKILLS.exists():
            skill_data.load_skill_data(str(DATAPACK_SKILLS))
        self.skill_data = skill_data
        self.icons = icons
        #: class id -> (parent class id, [(skill id, skill level, learn level)])
        self.trees: dict[int, tuple[int | None, list[tuple[int, int, int]]]] = {}
        for path in sorted(SKILL_TREES.glob("**/*.xml")):
            for tree in ET.parse(path).getroot().iter("skillTree"):
                if tree.get("type") != "classSkillTree" or tree.get("classId") is None:
                    continue
                parent = int(tree.get("parentClassId")) if tree.get("parentClassId") else None
                skills = [(int(s.get("skillId")), int(s.get("skillLevel")), int(s.get("getLevel")))
                          for s in tree.iter("skill")]
                self.trees[int(tree.get("classId"))] = (parent, skills)
        self.names = {int(c.get("classId")): c.get("name") for c in ET.parse(CLASS_LIST).getroot().iter("class")} \
            if CLASS_LIST.exists() else {}

    @staticmethod
    def chain_of(trees: dict, class_id: int) -> list[int]:
        """The class and its parents (by the skill trees), the starting class last."""
        out, current = [], class_id
        while current is not None and current in trees and current not in out:
            out.append(current)
            current = trees[current][0]
        return out

    def chain(self, class_id: int) -> list[int]:
        return self.chain_of(self.trees, class_id)

    def tier(self, class_id: int) -> int:
        return min(len(self.chain(class_id)) - 1, 3)

    def classes(self) -> list[dict]:
        out = []
        for class_id in self.trees:
            out.append({"id": class_id, "name": self.names.get(class_id, f"class {class_id}"), "tier": self.tier(class_id)})
        return sorted(out, key=lambda c: (c["tier"], c["name"]))

    def panels(self, class_id: int) -> dict:
        """The panel at every level where it changes, with the skills that are new there."""
        learned = [s for c in self.chain(class_id) for s in self.trees[c][1]]
        rows, previous = [], None
        for level in range(1, MAX_LEVEL + 1):
            best: dict[int, int] = {}
            for skill_id, skill_level, learn_level in learned:
                if learn_level <= level:
                    best[skill_id] = max(best.get(skill_id, 0), skill_level)
            panel = self.skill_data.combat_panel(best.items(), 64)
            passives = [info for skill_id, skill_level in sorted(best.items())
                        if self.is_passive(skill_id) and (info := self.skill_data.get_skill_info(skill_id, skill_level))]
            key = (tuple((info.skill_id, info.level) for info in panel),
                   tuple((info.skill_id, info.level) for info in passives))
            if key == previous:
                rows[-1]["to"] = level
                continue
            before = {} if previous is None else dict(previous[0])
            before_passive = {} if previous is None else dict(previous[1])
            rows.append({"from": level, "to": level, "skills": [self.describe(info, before) for info in panel],
                         "passives": [self.describe(info, before_passive, "passive") for info in passives]})
            previous = key
        return {"class": class_id, "name": self.names.get(class_id, f"class {class_id}"),
                "chain": [self.names.get(c, str(c)) for c in reversed(self.chain(class_id))], "rows": rows}

    def is_passive(self, skill_id: int) -> bool:
        return self.skill_data._SKILL_RAW.get(skill_id, {}).get("operate_type") == "P"

    def describe(self, info, before: dict[int, int], category: str | None = None) -> dict:
        raw = self.skill_data._SKILL_RAW.get(info.skill_id, {})
        needs = sorted({kind for kinds in raw.get("using", ()) for kind in kinds})
        return {"id": info.skill_id, "name": info.name, "level": info.level, "category": category or info.category,
                "needs": needs, "icon": info.skill_id in self.icons,
                "new": info.skill_id not in before, "up": info.skill_id in before and before[info.skill_id] != info.level}


PAGE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skill panels</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2433; --muted:#6b7385; --line:#e3e6ec; --new:#1a7f4b; --up:#2f6fdb;
          --attack:#d9534f; --debuff:#9a6dd7; --heal:#1a7f4b; --buff:#e08a1e; --summon:#2f6fdb; --passive:#8a93a6; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --card:#1b2029; --ink:#e7eaf0; --muted:#98a0b3; --line:#2b3240;
          --new:#3fbf7f; --up:#6ea2ff; --attack:#ff7b72; --debuff:#b892ff; --heal:#3fbf7f; --buff:#ffb44d; --summon:#6ea2ff; --passive:#7c8599; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:14px 20px; border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:1; }
  header h1 { font-size:17px; margin:0 12px 0 0; }
  header a { color:var(--muted); }
  select { font:inherit; padding:4px 8px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--ink); }
  main { padding:16px 20px 40px; max-width:1500px; margin:0 auto; }
  .chain { color:var(--muted); margin:0 0 12px; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--muted); margin-bottom:12px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  td { padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  td.lv { white-space:nowrap; font-weight:650; font-variant-numeric:tabular-nums; width:90px; }
  .skills { display:flex; flex-wrap:wrap; gap:6px 10px; }
  .skill { display:inline-flex; align-items:center; gap:6px; padding:3px 8px 3px 3px; border-radius:6px; border:1px solid var(--line); border-left:4px solid var(--c); }
  .skill.new { outline:2px solid var(--new); }
  .skill.up .lvl { color:var(--up); font-weight:650; }
  .ico, .noico { width:28px; height:28px; border-radius:4px; flex:none; }
  .noico { background:var(--line); }
  .lvl, .needs { font-size:11px; color:var(--muted); }
  .needs { border:1px solid var(--line); border-radius:4px; padding:0 4px; }
  .passives { margin-top:8px; padding-top:8px; border-top:1px dashed var(--line); }
  .passives .skill { background:var(--bg); }
  .tag { font-size:11px; color:var(--muted); margin-right:4px; align-self:center; }
  @media (max-width:520px) { header, main { padding-left:16px; padding-right:16px; } td.lv { width:60px; } }
</style>
</head>
<body>
<header>
  <h1>Skill panels</h1>
  <label>class <select id="cls"></select></label>
  <a href="/">← training dashboard</a>
</header>
<main>
  <p class="chain" id="chain"></p>
  <div class="legend">
    <span><i style="background:var(--attack)"></i>attack</span><span><i style="background:var(--debuff)"></i>debuff</span>
    <span><i style="background:var(--heal)"></i>heal</span><span><i style="background:var(--buff)"></i>buff</span>
    <span><i style="background:var(--summon)"></i>summon</span><span><i style="background:var(--passive)"></i>passive (below the panel)</span>
    <span><i style="outline:2px solid var(--new)"></i>new at this level</span>
    <span><b style="color:var(--up)">Lv</b> skill level went up</span>
    <span><span class="needs">BOW</span> weapon / armor the skill needs</span>
  </div>
  <table id="panels"></table>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
async function load(classId) {
  const r = await fetch("/api/skillpanels" + (classId ? "?class=" + classId : ""));
  const d = await r.json();
  if (!$("cls").options.length) {
    $("cls").innerHTML = d.classes.map(c => `<option value="${c.id}">${c.tier} · ${esc(c.name)}</option>`).join("");
  }
  $("cls").value = d.panel.class;
  history.replaceState(null, "", "?class=" + d.panel.class);
  $("chain").textContent = d.panel.chain.join(" → ");
  $("panels").innerHTML = d.panel.rows.map(row => {
    const lv = row.from === row.to ? row.from : `${row.from}–${row.to}`;
    const chip = s => `<span class="skill${s.new ? " new" : ""}${s.up ? " up" : ""}" style="--c:var(--${s.category})" title="${esc(s.name)} (id ${s.id})">`
      + (s.icon ? `<img class="ico" src="/icons/${s.id}.png" alt="">` : `<span class="noico"></span>`)
      + `${esc(s.name)} <span class="lvl">Lv ${s.level}</span>`
      + s.needs.map(n => ` <span class="needs">${esc(n)}</span>`).join("") + `</span>`;
    const skills = row.skills.map(chip).join("");
    const passives = (row.passives || []).map(chip).join("");
    return `<tr><td class="lv">${lv}</td><td><div class="skills">${skills || '<span class="lvl">no combat skills</span>'}</div>`
      + (passives ? `<div class="skills passives"><span class="tag">passive</span>${passives}</div>` : "") + `</td></tr>`;
  }).join("");
}
$("cls").onchange = e => load(e.target.value);
load(new URLSearchParams(location.search).get("class"));
</script>
</body>
</html>
"""
