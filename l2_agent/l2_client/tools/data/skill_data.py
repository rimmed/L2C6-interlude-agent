"""
Lineage2 C6 Interlude -- skill data lookup from server XML datapack.

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
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

# Path to skill XML files relative to the project root.
DEFAULT_SKILL_XML_PATH = "../L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/skills"

# Maps skill ID -> name.
_SKILL_NAME: dict[int, str] = {}

# Maps skill ID -> max level count.
_SKILL_LEVELS: dict[int, int] = {}

# Maps skill ID -> reuse delay in milliseconds (level 1 value for per-level tables).
_SKILL_REUSE_MS: dict[int, int] = {}

# Maps skill ID -> raw definition used by the combat catalog (see :func:`get_skill_info`).
_SKILL_RAW: dict[int, dict] = {}

# Skill ids that appear in a player skill tree, i.e. skills a character can learn. Item skills
# (herbs, potions, keys) are not in any tree, which is how they are kept off the skill panel.
_LEARNABLE: set[int] = set()
#: skill id -> {skill level: lowest character level (``getLevel``) at which any class tree teaches it}
_LEARN_LEVEL: dict[int, dict[int, int]] = {}

# ---------------------------------------------------------------------------
# Combat catalog
# ---------------------------------------------------------------------------

#: Skill categories, in panel priority order (the first ones survive when a panel is full).
CATEGORIES = ("attack", "debuff", "heal", "buff")
#: Summoning a servitor, pet or cubic: on the panel too, after the combat categories (it has its own
#: ``cat_summon`` feature, so the original one-hot layout of :data:`CATEGORIES` stays unchanged).
SUMMON_CATEGORY = "summon"
#: Every category that gets a panel slot, in panel priority order.
PANEL_CATEGORIES = (*CATEGORIES, SUMMON_CATEGORY)
#: Effects that bring a servitor or cubic (``SummonPet`` - pet items - and ``SummonNpc`` are not panel skills).
_SUMMON_EFFECTS = frozenset(("Summon", "SummonCubic"))
#: Coarse target classes.
TARGETS = ("self", "single", "area")

_ATTACK_EFFECTS = frozenset((
    "PhysicalDamage", "MagicalDamage", "MagicalDamageMp", "HpDrain", "FatalBlow", "Lethal", "EnergyDamage",
    "DeathLink", "StaticDamage", "CpDamPercent", "PhysicalDamageHpLink",
))
_MAGIC_ATTACK_EFFECTS = frozenset(("MagicalDamage", "MagicalDamageMp", "HpDrain", "DeathLink"))
_DEBUFF_EFFECTS = frozenset((
    "Debuff", "DamOverTime", "ManaDamOverTime", "Stun", "Root", "Paralyze", "Mute", "PhysicalMute", "Fear",
    "Sleep", "Confuse", "Bluff", "Betray",
    # aggro and hate (Aggression, Aura of Hate, Lure, Charm, Confusion, Curse Discord), buff removal
    # (Mage / Warrior Bane) and Mass Fear
    "GetAgro", "TargetMe", "AddHate", "RandomizeHate", "Distrust", "DispelBySlotProbability", "RunAway",
))
_HEAL_EFFECTS = frozenset((
    "Heal", "HealOverTime", "HealPercent", "CpHeal", "ManaHeal", "ManaHealOverTime", "ManaHealByLevel",
    "ManaHealPercent", "HpByLevel",
    # cures of the character's own ailments (Cure Poison / Bleeding, Bandage, Break Duress)
    "DispelBySlot",
))
#: Buffs without a ``Buff`` effect: charges (Sonic Focus, Focused Force), stances that root the caster
#: (Snipe, Ultimate Defense, Vengeance), invincibility, max HP and the servitor's empowerment.
_BUFF_EFFECTS = frozenset(("FocusEnergy", "ImmobileBuff", "ImmobilePetBuff", "Invincible", "MaxHp"))
#: Skills whose effects do not tell their role: Provoke (aggro without an effect of its own) and
#: Cancellation (its effect is shared with the friendly Cleanse).
_CATEGORY_OVERRIDES = {286: "debuff", 1056: "debuff"}
#: A/B switch (``L2_LEGACY_PANELS=1``): the panels of 2026-10-22 - per-character panels (policy, dataloader)
#: and the categories before the aggro / cure / charge skills were added to them.
LEGACY_PANELS = os.environ.get("L2_LEGACY_PANELS") == "1"
if LEGACY_PANELS:
    _DEBUFF_EFFECTS = _DEBUFF_EFFECTS - {"GetAgro", "TargetMe", "AddHate", "RandomizeHate", "Distrust",
                                         "DispelBySlotProbability", "RunAway"}
    _HEAL_EFFECTS = _HEAL_EFFECTS - {"DispelBySlot"}
    _BUFF_EFFECTS = frozenset()
    _CATEGORY_OVERRIDES = {}
#: Non-combat skills: resting, faking death, sneaking (toggles) and corpse handling (sweeping).
_NON_COMBAT_EFFECTS = frozenset(("Relax", "FakeDeath", "SilentMove", "ChameleonRest", "Sweeper", "ConsumeBody"))
_SELF_TARGETS = frozenset(("SELF", "PARTY", "CLAN", "PARTY_CLAN", "CLAN_MEMBER", "OWNER_PET", "SERVITOR", "NONE"))
_AREA_TARGETS = frozenset(("AURA", "AREA", "FRONT_AREA", "BEHIND_AREA", "FRONT_AURA", "BEHIND_AURA", "AREA_CORPSE_MOB",
                           "AURA_CORPSE_MOB", "AREA_UNDEAD", "AURA_UNDEAD_ENEMY", "AREA_FRIENDLY", "AURA_FRIENDLY",
                           "GROUND"))


@dataclass(frozen=True)
class SkillInfo:
    """Static, level-specific description of a skill, used as model features."""

    skill_id: int
    level: int
    name: str
    #: one of :data:`PANEL_CATEGORIES`, or ``"other"`` (crafting, escape, ...)
    category: str
    #: one of :data:`TARGETS`
    target: str
    is_magic: bool
    power: float
    mp_cost: float
    cast_range: float
    hit_time_ms: float
    reuse_ms: float
    #: the skill is in a player skill tree (not an item skill)
    learnable: bool
    #: character level from which this skill level is learnable (0 if unknown)
    magic_level: int = 0

    @property
    def combat(self) -> bool:
        """Belongs on the combat skill panel: a learnable active/toggle skill of a combat category or a summon."""
        return self.learnable and self.category in PANEL_CATEGORIES

_loaded = False


def load_skill_data(path: str = DEFAULT_SKILL_XML_PATH) -> None:
    """
    Parse all skill XML files and populate name + level lookup tables.

    :param path: directory containing skill XML files (e.g. ``00100-00199.xml``).
    """
    global _loaded, _SKILL_NAME, _SKILL_LEVELS, _SKILL_REUSE_MS, _SKILL_RAW
    if _loaded:
        return

    if not os.path.isdir(path):
        print(f"[SKILL_DATA] !! Directory not found: {path}")
        _loaded = True
        return

    count = 0
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".xml"):
            continue
        fpath = os.path.join(path, fname)
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
        except Exception:
            continue

        for skill_elem in root.findall("skill"):
            try:
                skill_id = int(skill_elem.get("id", "0"))
                name = skill_elem.get("name", "")
                levels = int(skill_elem.get("levels", "1"))
            except (ValueError, TypeError):
                continue

            _SKILL_NAME[skill_id] = name
            _SKILL_LEVELS[skill_id] = levels
            reuse = _read_reuse_delay(skill_elem)
            if reuse is not None:
                _SKILL_REUSE_MS[skill_id] = reuse
            _SKILL_RAW[skill_id] = _read_raw(skill_elem, name, levels)
            count += 1

    trees = os.path.join(os.path.dirname(os.path.normpath(path)), "players", "skillTrees")
    _load_learnable(trees)
    _loaded = True
    print(f"[SKILL_DATA] Loaded {count} skills from {path} ({len(_LEARNABLE)} learnable)")


def _read_raw(skill_elem: ET.Element, name: str, levels: int) -> dict:
    """Keep what the combat catalog needs: operate/target type, effects and per-level values."""
    tables = {t.get("name"): (t.text or "").split() for t in skill_elem.findall("table") if t.get("name")}
    values = {tag: (skill_elem.findtext(tag) or "").strip()
              for tag in ("power", "mpConsume", "mpInitialConsume", "castRange", "hitTime", "reuseDelay", "isMagic",
                    "magicLevel")}
    return {
        "name": name,
        "levels": levels,
        "operate_type": (skill_elem.findtext("operateType") or "").strip(),
        "target_type": (skill_elem.findtext("targetType") or "").strip(),
        "abnormal_type": (skill_elem.findtext("abnormalType") or "").strip(),
        # <conditions><using kind="BOW" /> ...: the weapon / armor the skill needs (all of them must hold)
        "using": tuple(frozenset(k.strip() for k in u.get("kind", "").split(",") if k.strip())
                       for c in skill_elem.findall("conditions") for u in c.iter("using") if u.get("kind")),
        "effects": frozenset(e.get("name") for e in skill_elem.iter("effect") if e.get("name")),
        # <conditions><target race="UNDEAD"/>: the server refuses the skill on any other target
        "target_race": next((t.get("race") for c in skill_elem.findall("conditions") for t in c.iter("target")
                             if t.get("race")), None),
        "tables": tables,
        "values": values,
    }


def _load_learnable(directory: str) -> None:
    """Collect every ``skillId`` of the player skill trees and the character level each skill level needs."""
    _LEARNABLE.clear()
    _LEARN_LEVEL.clear()
    if not os.path.isdir(directory):
        print(f"[SKILL_DATA] !! Skill tree directory not found: {directory} (no skill is learnable)")
        return
    for root_dir, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".xml"):
                continue
            try:
                tree = ET.parse(os.path.join(root_dir, fname))
            except Exception:
                continue
            for skill in tree.getroot().iter("skill"):
                try:
                    skill_id = int(skill.get("skillId", ""))
                except ValueError:
                    continue
                _LEARNABLE.add(skill_id)
                try:
                    skill_level, get_level = int(skill.get("skillLevel", "")), int(skill.get("getLevel", ""))
                except ValueError:
                    continue
                levels = _LEARN_LEVEL.setdefault(skill_id, {})
                levels[skill_level] = min(levels.get(skill_level, get_level), get_level)


def _level_value(raw: dict, tag: str, level: int, default: float = 0.0) -> float:
    """Value of ``tag`` at ``level``: a literal, or the level's entry of a ``#table``."""
    text = raw["values"].get(tag, "")
    if text.startswith("#"):
        table = raw["tables"].get(text, [])
        if not table:
            return default
        text = table[min(max(level, 1), len(table)) - 1]
    try:
        return float(text)
    except ValueError:
        return default


def get_abnormal_type(skill_id: int) -> str:
    """Effect slot of a buff (e.g. ``SPEED_UP``): two effects of one type do not stack; ``""`` if none."""
    raw = _SKILL_RAW.get(skill_id)
    return raw["abnormal_type"] if raw else ""


#: ``<using kind>`` values that name a weapon type; ``LIGHT`` / ``HEAVY`` / ``MAGIC`` name the body armor.
WEAPON_KINDS = frozenset(("SWORD", "BLUNT", "DAGGER", "BOW", "POLE", "ETC", "FIST", "DUAL", "DUALFIST", "FISHINGROD",
                          "BIGSWORD", "BIGBLUNT", "DUALDAGGER"))
ARMOR_KINDS = frozenset(("LIGHT", "HEAVY", "MAGIC"))


def usable_with(skill_id: int, weapon: str, armor: str, shield: bool) -> bool:
    """
    The equipment meets the skill's ``<using kind>`` conditions (Power Shot needs a bow, Sonic skills
    dual swords ...). ``weapon`` is the weapon type (``""`` / ``FIST`` without one), ``armor`` the body
    armor type, ``shield`` whether something is in the left hand. A skill without such a condition is
    always usable.
    """
    raw = _SKILL_RAW.get(skill_id)
    for kinds in raw["using"] if raw else ():
        weapons, armors = kinds & WEAPON_KINDS, kinds & ARMOR_KINDS
        ok = (bool(weapons) and (weapon or "FIST") in weapons) or (bool(armors) and armor in armors) \
            or ("SHIELD" in kinds and shield)
        if not ok:
            return False
    return True


#: Summons of siege weapons (Siege Golem, Wild Hog Cannon, Swoop Cannon): no use for farming monsters.
SIEGE_WEAPON_SUMMONS: frozenset[int] = frozenset({13, 299, 448})


def target_race(skill_id: int) -> str | None:
    """The race the skill's target must be (``UNDEAD`` for Disrupt Undead ...), ``None`` for any."""
    raw = _SKILL_RAW.get(skill_id)
    return raw.get("target_race") if raw else None


def burns_mana_only(skill_id: int) -> bool:
    """The skill damages the target's MP, not its HP (Mana Burn): no use for killing a monster."""
    raw = _SKILL_RAW.get(skill_id)
    return bool(raw) and raw["effects"] == {"MagicalDamageMp"}


def needs_summon(skill_id: int) -> bool:
    """The skill targets the character's summon (Servitor Heal, Bright Servitor ...): useless without one."""
    raw = _SKILL_RAW.get(skill_id)
    return bool(raw) and raw["target_type"] in ("SERVITOR", "PET", "SUMMON")


def summons_servitor(skill_id: int) -> bool:
    """The skill brings a servitor or pet (not a cubic): the server refuses it while one is out."""
    raw = _SKILL_RAW.get(skill_id)
    return bool(raw) and "Summon" in raw["effects"]


def get_skill_info(skill_id: int, level: int = 1) -> SkillInfo | None:
    """Static description of a skill at ``level``, or ``None`` for an unknown skill."""
    raw = _SKILL_RAW.get(skill_id)
    if raw is None:
        return None
    level = min(max(level, 1), max(raw["levels"], 1))
    effects, operate = raw["effects"], raw["operate_type"]
    is_active = operate.startswith("A") or operate == "T"
    # Corpse-targeted skills need a dead monster as target, which the bots never select.
    if not is_active or effects & _NON_COMBAT_EFFECTS or "CORPSE" in raw["target_type"]:
        category = "other"
    elif skill_id in _CATEGORY_OVERRIDES:
        category = _CATEGORY_OVERRIDES[skill_id]
    elif effects & _SUMMON_EFFECTS:
        category = SUMMON_CATEGORY
    elif effects & _ATTACK_EFFECTS:
        category = "attack"
    elif effects & _DEBUFF_EFFECTS:
        category = "debuff"
    elif effects & _HEAL_EFFECTS:
        category = "heal"
    elif "Buff" in effects or effects & _BUFF_EFFECTS or operate == "T":
        category = "buff"
    else:
        category = "other"

    target_type = raw["target_type"]
    target = "self" if target_type in _SELF_TARGETS else "area" if target_type in _AREA_TARGETS else "single"
    magic_flag = _level_value(raw, "isMagic", level, default=-1.0)
    is_magic = magic_flag == 1.0 or (magic_flag < 0 and bool(effects & _MAGIC_ATTACK_EFFECTS))
    return SkillInfo(
        skill_id=skill_id,
        level=level,
        name=raw["name"],
        category=category,
        target=target,
        is_magic=is_magic,
        power=_level_value(raw, "power", level),
        mp_cost=_level_value(raw, "mpConsume", level) + _level_value(raw, "mpInitialConsume", level),
        cast_range=max(0.0, _level_value(raw, "castRange", level)),
        hit_time_ms=_level_value(raw, "hitTime", level),
        reuse_ms=_level_value(raw, "reuseDelay", level),
        learnable=skill_id in _LEARNABLE,
        magic_level=int(_level_value(raw, "magicLevel", level)),
    )


def combat_skill_ids() -> list[int]:
    """Ids of every learnable skill of a combat category (the pool of possible panel skills)."""
    return sorted(sid for sid in _SKILL_RAW if (info := get_skill_info(sid)) is not None and info.combat)


def best_level_for(skill_id: int, character_level: int) -> int:
    """
    Highest level of ``skill_id`` a character of ``character_level`` can have (0 = none yet).

    Uses the ``getLevel`` of the player skill trees (the lowest over all classes); a skill outside
    the trees falls back to its ``magicLevel``.
    """
    learn = _LEARN_LEVEL.get(skill_id)
    if learn:
        return max((lvl for lvl, need in learn.items() if need <= character_level), default=0)
    raw = _SKILL_RAW.get(skill_id)
    if raw is None:
        return 0
    best = 0
    for level in range(1, max(raw["levels"], 1) + 1):
        if _level_value(raw, "magicLevel", level) <= character_level:
            best = level
    return best


def combat_panel(skills: Iterable[tuple[int, int]], max_slots: int = 64,
                 equipment: tuple[str, str, bool] | None = None) -> list[SkillInfo]:
    """
    The combat skill panel of a character: its learnable active skills of a combat category.

    :param skills: ``(skill_id, level)`` pairs the character knows (passives are dropped here)
    :param max_slots: panel size; when there are more skills, categories earlier in
        :data:`PANEL_CATEGORIES` win (attacks first, summons last), then lower skill ids
    :param equipment: ``(weapon type, body armor type, has shield)`` of the character; skills its
        equipment does not allow (Power Shot without a bow ...) stay off the panel. ``None`` = unknown
    :return: the panel in a stable order (category priority, then skill id)
    """
    best: dict[int, SkillInfo] = {}
    for skill_id, level in skills:
        info = get_skill_info(int(skill_id), int(level))
        if info is not None and info.combat and (skill_id not in best or level > best[skill_id].level) \
                and (equipment is None or usable_with(int(skill_id), *equipment)):
            best[int(skill_id)] = info
    ranked = sorted(best.values(), key=lambda s: (PANEL_CATEGORIES.index(s.category), s.skill_id))
    return ranked[:max_slots]


def _read_reuse_delay(skill_elem: ET.Element) -> int | None:
    """
    Read ``<reuseDelay>`` of a skill element, in milliseconds.

    The value is either a literal (``13000``) or a reference to a per-level ``<table name="#x">``,
    in which case the level 1 entry is used.
    """
    text = (skill_elem.findtext("reuseDelay") or "").strip()
    if text.startswith("#"):
        table = next((t for t in skill_elem.findall("table") if t.get("name") == text), None)
        text = (table.text or "").split()[0] if table is not None and table.text else ""
    try:
        return int(float(text))
    except ValueError:
        return None


def get_skill_name(skill_id: int) -> str:
    """Return the human-readable name for a skill ID, or ``""`` if unknown."""
    return _SKILL_NAME.get(skill_id, "")


def get_skill_levels(skill_id: int) -> int:
    """Return the max level count for a skill ID, or ``1`` if unknown."""
    return _SKILL_LEVELS.get(skill_id, 1)


def get_skill_reuse_delay(skill_id: int) -> int:
    """Return the reuse delay of a skill in milliseconds, or ``0`` if unknown."""
    return _SKILL_REUSE_MS.get(skill_id, 0)
