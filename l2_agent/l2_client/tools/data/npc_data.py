"""
Lineage2 C6 Interlude -- NPC name lookup from server XML datapack.

The MIT License (MIT)

Copyright (c) 2026 Viacheslav Reutskyi <viacheslav.reutskyi@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from . import skill_data

# Path to the npc stats XML files relative to the project root
DEFAULT_NPC_XML_PATH = "../L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/npcs"

_npc_names: dict[int, str] = {}
_npc_levels: dict[int, int] = {}
_npc_hp: dict[int, float] = {}
#: npc template id -> race (``<race>``)
_npc_race: dict[int, str] = {}
#: npc template id -> type (``Monster``, ``RaidBoss``, ``GrandBoss`` ...)
_npc_type: dict[int, str] = {}
_loaded = False


def load_npc_names(path: str = DEFAULT_NPC_XML_PATH) -> dict[int, str]:
    """
    Parse all NPC XML files and return a {npc_id: name} lookup table.

    Also populates the internal ``_npc_levels`` dict with the level
    attribute (e.g. ``<npc id="20120" level="4" …>``) and ``_npc_hp`` with the
    base maximum HP (``<vitals hp="…">``).

    :param path: directory containing ``20000-20099.xml`` etc.
    :returns: dict mapping npc template id -> name
    """
    global _npc_names, _npc_levels, _npc_hp, _npc_race, _npc_type, _loaded
    if _loaded:
        return _npc_names

    if not os.path.isdir(path):
        print(f"[NPC_DATA]  !! Directory not found: {path}")
        _loaded = True
        return _npc_names

    pattern = re.compile(r'<npc\s+id="(\d+)"\s+level="(\d+)"[^>]*\sname="([^"]+)"')
    vitals = re.compile(r'<vitals\s+hp="([\d.]+)"')
    race = re.compile(r'<race>([A-Z_]+)</race>')
    count = 0
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".xml"):
            continue
        fpath = os.path.join(path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception:
            continue
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            npc_id = int(m.group(1))
            level = int(m.group(2))
            name = m.group(3)
            _npc_names[npc_id] = name
            _npc_levels[npc_id] = level
            kind_attr = re.search(r'\stype="([A-Za-z]+)"', m.group(0))
            if kind_attr:
                _npc_type[npc_id] = kind_attr.group(1)
            # The first <vitals> before the next <npc> belongs to this one.
            block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            hp = vitals.search(text, m.end(), block_end)
            if hp:
                _npc_hp[npc_id] = float(hp.group(1))
            kind = race.search(text, m.end(), block_end)
            if kind:
                _npc_race[npc_id] = kind.group(1)
            count += 1

    _loaded = True
    print(f"[NPC_DATA]  Loaded {count} NPC entries (name + level) from {path}")
    load_monster_templates(path)  # the model's monster features need them wherever the names are loaded
    return _npc_names


def get_npc_name(template_id: int) -> str:
    """Return the human-readable name for an NPC template ID, or ``""``."""
    return _npc_names.get(template_id, "")


def get_npc_level(template_id: int) -> int:
    """Return the level for an NPC template ID, or 0 if unknown."""
    return _npc_levels.get(template_id, 0)


def is_boss(template_id: int) -> bool:
    """A raid boss or grand boss: group content, never a farming target."""
    return _npc_type.get(template_id) in ("RaidBoss", "GrandBoss")


def get_npc_race(template_id: int) -> str:
    """Return the race of an NPC template (``UNDEAD``, ``ANIMAL`` ...), or ``""`` if unknown."""
    return _npc_race.get(template_id, "")


def get_npc_max_hp(template_id: int) -> float:
    """Return the base maximum HP for an NPC template ID from the datapack, or 0 if unknown."""
    return _npc_hp.get(template_id, 0.0)


@dataclass(frozen=True)
class MonsterTemplate:
    """What a monster template tells about a fight with it: base stats with its passive skills applied."""

    #: attacks a character that comes into ``aggro_range`` (``isAggressive``, the server's default is true)
    aggressive: bool
    aggro_range: int
    #: hits from afar: a bow, a long attack range or a long-range magic attack
    ranged: bool
    race: str
    max_hp: float
    p_def: float
    m_def: float
    #: trait (``BLUNT``, ``BOW``, ``DAGGER`` ...) -> ``DefenceTrait`` value: below 0 a weak point, above 0 a resistance
    defence_traits: dict[str, float] = field(default_factory=dict)


#: ``<attack range>`` from which a monster counts as ranged (melee monsters hit at 40 .. 80)
RANGED_ATTACK_RANGE = 200
_monsters: dict[int, MonsterTemplate] = {}
_monsters_loaded = False
_monsters_lock = threading.Lock()


def load_monster_templates(path: str = DEFAULT_NPC_XML_PATH) -> dict[int, MonsterTemplate]:
    """
    Parse the ``<npc>`` templates of ``path`` (``custom/`` last, as the server overrides them) into
    :class:`MonsterTemplate` s. The passive skills of a template (HP Increase, Strong P. Def., weak points
    ...) are applied through :mod:`skill_data`, which is loaded from the ``skills`` folder next to ``path``.
    """
    global _monsters_loaded
    with _monsters_lock:
        if not _monsters_loaded:
            _load_monster_templates(path)
            _monsters_loaded = True
    return _monsters


def _load_monster_templates(path: str) -> None:
    if not os.path.isdir(path):
        print(f"[NPC_DATA]  !! Directory not found: {path}")
        return
    skill_data.load_skill_data(os.path.join(os.path.dirname(os.path.abspath(path)), "skills"))

    files = []
    for root_dir, _, names in os.walk(path):
        files += [os.path.join(root_dir, name) for name in names if name.endswith(".xml")]
    for fpath in sorted(files, key=lambda f: (os.path.dirname(f) != path.rstrip("/"), f)):
        try:
            root = ET.parse(fpath).getroot()
        except ET.ParseError:
            continue
        # only the direct children: a <minions> list holds <npc id=... count=...> references too
        for npc in root.findall("npc"):
            template = _monster_template(npc)
            if template is not None:
                _monsters[int(npc.get("id"))] = template
    print(f"[NPC_DATA]  Loaded {len(_monsters)} monster templates from {path}")


def _monster_template(npc: ET.Element) -> MonsterTemplate | None:
    stats = npc.find("stats")
    if stats is None:
        return None

    def number(element: ET.Element | None, attribute: str) -> float:
        try:
            return float(element.get(attribute, 0)) if element is not None else 0.0
        except ValueError:
            return 0.0

    attack = stats.find("attack")
    ai = npc.find("ai")
    aggro_range = int(number(ai, "aggroRange"))
    long_range_magic = any("LongRange" in (skill.get("name") or "") for skill in npc.iterfind("parameters/skill"))

    max_hp = number(stats.find("vitals"), "hp")
    p_def = number(stats.find("defence"), "physical")
    m_def = number(stats.find("defence"), "magical")
    traits: dict[str, float] = {}
    for skill in npc.iterfind("skillList/skill"):
        try:
            skill_id, level = int(skill.get("id")), int(skill.get("level", 1))
        except (TypeError, ValueError):
            continue
        multipliers = skill_data.stat_multipliers(skill_id, level)
        max_hp *= multipliers.get("maxHp", 1.0)
        p_def *= multipliers.get("pDef", 1.0)
        m_def *= multipliers.get("mDef", 1.0)
        for trait, value in skill_data.defence_traits(skill_id, level).items():
            traits[trait] = traits.get(trait, 0.0) + value

    return MonsterTemplate(
        aggressive=aggro_range > 0 and (ai is None or ai.get("isAggressive", "true").lower() == "true"),
        aggro_range=aggro_range,
        ranged=(attack is not None and (attack.get("type") == "BOW" or number(attack, "range") >= RANGED_ATTACK_RANGE))
        or long_range_magic,
        race=(npc.findtext("race") or "").strip(),
        max_hp=max_hp,
        p_def=p_def,
        m_def=m_def,
        defence_traits=traits,
    )


def get_monster_template(template_id: int) -> MonsterTemplate | None:
    """The :class:`MonsterTemplate` of an NPC template id (:func:`load_monster_templates` first), ``None`` if unknown."""
    return _monsters.get(template_id)
