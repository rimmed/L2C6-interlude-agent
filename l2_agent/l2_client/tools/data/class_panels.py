"""
Fixed skill panels per class, built once from the datapack and shared by BC and RL training and the live client.

A class's panel lists every combat skill (``SkillInfo.combat``: attack, debuff, heal, buff, summon) of
its profession chain - the player skill trees, following their ``parentClassId`` (``classList.xml``
has a few wrong parents) - in category priority, then skill id order, followed by the skills of
every servitor the class can summon. A skill keeps its slot for the whole life of the character:
what changes is only whether the slot is *valid* right now - the skill is on the panel at the
character's level (learned and not outgrown, :attr:`PanelSlot.outgrown_at`), the character knows it, its equipment allows it, for a summon's skill: that summon is out. An invalid slot is fed to the model
as an all-zero row, exactly like padding.

The panels live in ``class_panels.json`` next to this module; rebuild it after a datapack change::

    python -m l2_client.tools.data.class_panels          # writes class_panels.json, prints a summary

Checkpoints store :func:`panels_hash` so a model and the panels it was trained with stay together.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PANELS_JSON = Path(__file__).with_name("class_panels.json")
#: datapack root (``dist/game/data``) relative to the l2_agent project
DATAPACK = Path(__file__).resolve().parents[4] / "L2J_Mobius" / "L2J_Mobius_CT_0_Interlude" / "dist" / "game" / "data"
MAX_SLOTS = 64
#: Skills learned before this character level (the starting profession's) ...
EARLY_BEFORE = 20
#: ... leave the panel at this level unless they still grow from it on: an attack that stopped
#: growing misses or hits far weaker than the skills learned since.
EARLY_UNTIL = 40
#: Categories the rule spares: a buff or a heal does not miss, players keep using them at any level.
NEVER_OUTGROWN: frozenset[str] = frozenset({"buff", "heal"})
#: Skills a class line never gets on its panel: class id in the chain -> skill ids. The Artisan line
#: (Artisan, Warsmith, Maestro) keeps the Dwarf Fighter's Spoil, a Scavenger's skill it has no use for.
EXCLUDED_SKILLS: dict[int, frozenset[int]] = {56: frozenset({254})}


@dataclass(frozen=True)
class PanelSlot:
    """One fixed slot of a class panel."""

    skill_id: int
    name: str
    category: str
    #: a skill of the class's servitors (ordered through RequestActionUse), not of the character
    pet: bool
    #: lowest character level at which the class (chain) learns the skill; 0 for a servitor's skill
    learn_level: int
    #: the weapon / armor kinds the skill needs (``<using kind>``, any of them), empty = none
    needs: tuple[str, ...] = ()
    #: character level at which the class learns the skill's last level; 0 for a servitor's skill
    last_level: int = 0
    #: first character level at which the skill is off the panel (:func:`outgrown_at`); 0 = never
    until: int = 0

    @property
    def outgrown_at(self) -> int | None:
        return self.until or None

    def active_at(self, level: int) -> bool:
        """The skill is on the panel of a character of ``level``: learned and not outgrown yet."""
        return self.pet or (self.learn_level <= level and (not self.until or level < self.until))


def outgrown_at(category: str, learn_level: int, last_level: int) -> int:
    """
    First character level at which a skill leaves the panel, 0 = never: a skill of the starting
    profession that no longer grows at :data:`EARLY_UNTIL` leaves there (buffs and heals never do).
    Stored in the panel file, so its hash covers the rule.
    """
    if category in NEVER_OUTGROWN:
        return 0
    return EARLY_UNTIL if learn_level < EARLY_BEFORE and last_level < EARLY_UNTIL else 0


@dataclass(frozen=True)
class ClassPanel:
    class_id: int
    name: str
    #: the class and its parents, the starting class last
    chain: tuple[int, ...]
    slots: tuple[PanelSlot, ...]

    @property
    def size(self) -> int:
        return len(self.slots)

    def slot_of(self, skill_id: int, pet: bool = False) -> int | None:
        return _slot_index(self)[(int(skill_id), bool(pet))] if (int(skill_id), bool(pet)) in _slot_index(self) else None


@lru_cache(maxsize=None)
def _slot_index(panel: ClassPanel) -> dict[tuple[int, bool], int]:
    return {(slot.skill_id, slot.pet): i for i, slot in enumerate(panel.slots)}


# ----------------------------------------------------------------------------------------------
# Building from the datapack
# ----------------------------------------------------------------------------------------------

def _read_trees(directory: Path) -> dict[int, tuple[int | None, dict[int, tuple[int, int]]]]:
    """class id -> (parent class id, {skill id: (lowest getLevel, highest getLevel)})."""
    trees: dict[int, tuple[int | None, dict[int, tuple[int, int]]]] = {}
    for path in sorted(directory.glob("**/*.xml")):
        for tree in ET.parse(path).getroot().iter("skillTree"):
            if tree.get("type") != "classSkillTree" or tree.get("classId") is None:
                continue
            skills: dict[int, tuple[int, int]] = {}
            for skill in tree.iter("skill"):
                sid, level = int(skill.get("skillId")), int(skill.get("getLevel", "0"))
                first, last = skills.get(sid, (level, level))
                skills[sid] = (min(first, level), max(last, level))
            parent = tree.get("parentClassId")
            trees[int(tree.get("classId"))] = (int(parent) if parent else None, skills)
    return trees


def _chain(trees: dict, class_id: int) -> list[int]:
    out, current = [], class_id
    while current is not None and current in trees and current not in out:
        out.append(current)
        current = trees[current][0]
    return out


def _summon_npcs(skills_dir: Path) -> dict[int, set[int]]:
    """Skill id -> npc ids of the servitors its ``Summon`` effect brings (every level and enchant)."""
    out: dict[int, set[int]] = {}
    for path in sorted(skills_dir.glob("*.xml")):
        for skill in ET.parse(path).getroot().iter("skill"):
            for effect in skill.iter("effect"):
                if effect.get("name") != "Summon":
                    continue
                ref = (effect.findtext("npcId") or "").strip()
                if ref.startswith("#"):
                    table = next((t for t in skill.findall("table") if t.get("name") == ref), None)
                    values = (table.text or "").split() if table is not None else []
                else:
                    values = [ref]
                out.setdefault(int(skill.get("id")), set()).update(int(v) for v in values if v.isdigit())
    return out


def build(datapack: Path = DATAPACK) -> dict:
    """The panels of every class from the datapack's skill trees, skills and servitor templates."""
    from . import skill_data, summon_data

    skills_dir = datapack / "stats" / "skills"
    if not skill_data._SKILL_RAW:
        skill_data.load_skill_data(str(skills_dir))
    summon_data.load_summon_data(str(datapack / "stats" / "npcs"))
    trees = _read_trees(datapack / "stats" / "players" / "skillTrees")
    class_list = datapack / "stats" / "players" / "classList.xml"
    names = {int(c.get("classId")): c.get("name") for c in ET.parse(class_list).getroot().iter("class")} \
        if class_list.exists() else {}
    summon_npcs = _summon_npcs(skills_dir)
    order = skill_data.PANEL_CATEGORIES

    classes = {}
    for class_id in sorted(trees):
        chain = _chain(trees, class_id)
        learn: dict[int, int] = {}
        last: dict[int, int] = {}
        for c in chain:
            for sid, (first, final) in trees[c][1].items():
                learn[sid] = min(learn.get(sid, first), first)
                last[sid] = max(last.get(sid, final), final)
        excluded = set().union(*(EXCLUDED_SKILLS.get(c, ()) for c in chain))
        own = []
        for sid in learn:
            if sid in excluded:
                continue
            info = skill_data.get_skill_info(sid, 1)
            if info is not None and info.combat:
                raw = skill_data._SKILL_RAW[sid]
                needs = tuple(sorted({k for kinds in raw.get("using", ()) for k in kinds}))
                own.append({"skill_id": sid, "name": info.name, "category": info.category, "pet": False,
                            "learn_level": learn[sid], "last_level": last[sid],
                            "until": outgrown_at(info.category, learn[sid], last[sid]), "needs": needs})
        own.sort(key=lambda s: (order.index(s["category"]), s["skill_id"]))
        pets: dict[int, dict] = {}
        for slot in own:
            for npc in sorted(summon_npcs.get(slot["skill_id"], ())):
                for pet_skill in summon_data.pet_skills(npc):
                    info = skill_data.get_skill_info(pet_skill.skill_id, pet_skill.level)
                    if info is not None and pet_skill.skill_id not in pets:
                        pets[pet_skill.skill_id] = {"skill_id": pet_skill.skill_id, "name": info.name,
                                                    "category": info.category, "pet": True, "learn_level": 0,
                                                    "last_level": 0, "until": 0, "needs": ()}
        slots = own + [pets[sid] for sid in sorted(pets)]
        dropped = slots[MAX_SLOTS:]
        if dropped:  # categories later in the order lose their slots first
            print(f"[CLASS_PANELS] {names.get(class_id, class_id)}: {len(slots)} skills, "
                  f"dropped {[s['name'] for s in dropped]}")
        classes[str(class_id)] = {"name": names.get(class_id, f"class {class_id}"), "chain": chain,
                                  "tree_skills": sorted(learn), "slots": slots[:MAX_SLOTS]}
    return {"max_slots": MAX_SLOTS, "categories": list(order), "classes": classes}


def save(data: dict, path: Path = PANELS_JSON) -> None:
    # one class per line: small, and a datapack change shows as a readable diff
    classes = ",\n".join(f"  {json.dumps(cid)}: {json.dumps(c, sort_keys=True)}"
                          for cid, c in sorted(data["classes"].items(), key=lambda kv: int(kv[0])))
    head = {k: v for k, v in data.items() if k != "classes"}
    path.write_text(json.dumps(head, sort_keys=True)[:-1] + ', "classes": {\n' + classes + "\n}}\n", encoding="utf-8")
    load.cache_clear()
    _tree_sets.cache_clear()


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load(path: str = str(PANELS_JSON)) -> dict[int, ClassPanel]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(cid): ClassPanel(int(cid), c["name"], tuple(c["chain"]),
                                 tuple(PanelSlot(s["skill_id"], s["name"], s["category"], s["pet"], s["learn_level"],
                                                 tuple(s["needs"]), s.get("last_level", 0), s.get("until", 0))
                                       for s in c["slots"]))
            for cid, c in data["classes"].items()}


def panel(class_id: int | None) -> ClassPanel | None:
    """The fixed panel of a class, ``None`` for an unknown class."""
    return None if class_id is None else load().get(int(class_id))


def panels_hash(path: Path = PANELS_JSON) -> str:
    """Short fingerprint of the panel file: stored in checkpoints and checked by the live client."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _tree_sets() -> dict[int, frozenset[int]]:
    data = json.loads(PANELS_JSON.read_text(encoding="utf-8"))
    return {int(cid): frozenset(c["tree_skills"]) for cid, c in data["classes"].items()}


def infer_class(known: Iterable[int], name_hint: str = "") -> int | None:
    """
    The class of a character from its known skills (logs written before they carried the class):
    the class whose chain teaches most of them; on ties the class named in the character's name
    (``HumanMystic00`` - starting classes of several races know the same skills), then the
    shortest chain (a class rather than its later profession that adds nothing known yet).
    """
    known = set(int(k) for k in known)
    if not known:
        return None
    panels, best, best_key = load(), None, None
    hint = name_hint.replace(" ", "").lower()
    for class_id, tree in _tree_sets().items():
        name = panels[class_id].name.replace(" ", "").lower() if class_id in panels else ""
        key = (len(known & tree), bool(name) and hint.startswith(name), -len(tree))
        if best_key is None or key > best_key:
            best, best_key = class_id, key
    return best


def main() -> int:
    os.chdir(Path(__file__).resolve().parents[3])  # the l2_agent project (skill_data's relative defaults)
    data = build()
    save(data)
    sizes = sorted(((len(c["slots"]), c["name"]) for c in data["classes"].values()), reverse=True)
    print(f"{PANELS_JSON}: {len(data['classes'])} classes, hash {panels_hash()}")
    print("largest:", ", ".join(f"{name} {n}" for n, name in sizes[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
