"""
Player-action log → Parquet converter with NPC / item / skill links.

Parses the ``playeraction`` log files written by the L2 game client
(``l2_playeractions/*.log``) and writes three linked tables:

* ``l2jc6_playeractions.parquet`` — one row per action frame, capturing every
  block of the log line: character status, stats, equipment, world position,
  environment, skill cooldowns and the action payload itself.
* ``l2jc6_characters.parquet`` — the character dimension (``character_id`` +
  name + per-character aggregates). The other tables reference it by id only.
* ``l2jc6_character_skills.parquet`` — every skill observed per character, keyed
  by ``skill_id``, which is also the key used by the action table's skill,
  effect and cooldown columns.

Entities are resolved against the datapack tables produced by the sibling
converters:

* ``l2jc6_npcs.parquet``  — see ``npc_xml2parquet.py``
* ``l2jc6_items.parquet`` — see ``item_xml2parquet.py``
* skill names come from ``.../data/stats/skills/*.xml``

Two log layouts are supported, auto-detected per line:

==========================  ============================  ==============================
information                 older logs (``000xx``)        newer logs (``100xx``)
==========================  ============================  ==============================
character                   ``HumanFighter00``            ``[HumanFighter00]``
action                      ``attacked: X``               ``[attacked]: X``
status                      ``[HP:.. MP:.. CP:..]``       ``[CharStatus: HP:..]``
stats                       bare ``P.Atk:..``             ``[CharStats: P.Atk:..]``
selection                   ``[Target:42 Monster ..]``    ``[Nav: Dist:.. LoS:..]``
environment                 ``[Env:NearbyMon:10 ..]``     ``[Env: Monsters Count: 10, ..]``
target identity             name only, in the message     ``Name (NPC_ID:, Lvl:)``
==========================  ============================  ==============================

NPC ids present in the message text are authoritative; otherwise the name is
looked up in the NPC table (combat types and higher EXP win on name clashes, the
clash is flagged in ``entity_ambiguous``). Item ids observed in ``Pickup``
messages are learned and reused for the same item name elsewhere in the log.


Created with deepseek-v4-flash
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:  # run as a script from l2_NN/tools, or imported as part of the l2_NN package
    import playeraction_v2 as v2
except ImportError:  # pragma: no cover
    from l2_NN.tools import playeraction_v2 as v2

# Repository root, so the default datapack tables are found from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPC_PARQUET = _REPO_ROOT / "l2jc6_npcs.parquet"
DEFAULT_ITEM_PARQUET = _REPO_ROOT / "l2jc6_items.parquet"
DEFAULT_OUTPUT = _REPO_ROOT / "l2jc6_playeractions.parquet"
DEFAULT_CHARACTERS_OUTPUT = _REPO_ROOT / "l2jc6_characters.parquet"
DEFAULT_SKILLS_OUTPUT = _REPO_ROOT / "l2jc6_character_skills.parquet"
# Datapack skill definitions, used to give the skill tables human-readable names.
DEFAULT_SKILLS_XML = (
    _REPO_ROOT / "L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/skills"
)
# Skill observation sources recorded in ``l2jc6_character_skills.sources``.
SKILL_SOURCE_ACTIVE = "active"
SKILL_SOURCE_PASSIVE = "passive"
SKILL_SOURCE_USED = "used"
SKILL_SOURCE_LEARNED = "learned"
SKILL_SOURCE_COOLDOWN = "cooldown"
SKILL_SOURCE_EFFECT = "effect"

SLOTS = [
    "Weapon",
    "Shield",
    "Head",
    "Chest",
    "Gloves",
    "Legs",
    "Feet",
    "Neck",
    "REar",
    "LEar",
    "RFinger",
    "LFinger",
]

# --------------------------------------------------------------------------
# Line / block regular expressions (shared by both layouts where possible)
# --------------------------------------------------------------------------

LINE_START = re.compile(
    r"^(?P<timestamp>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+INFO\s+\d+\s+playeraction\s+(?P<rest>.*)$"
)

# Some entries repeat a short clock stamp before the character:
# "playeraction\t[19:19:40,823] [HumanFighter03] BUY Item: ..".
CLOCK_PREFIX = re.compile(r"^\[\d{2}:\d{2}:\d{2},\d{3}\]\s+")
BRACKETED_CHARACTER = re.compile(r"^\[(?P<character>[^\]]+)\]\s+(?P<rest>.+)$")
BARE_CHARACTER = re.compile(r"^(?P<character>[A-Za-z][A-Za-z0-9_]*)\s+(?P<rest>.+)$")

# [HP:.. MP:.. CP:.. Exp:.. Sp:.. Lvl:..]  /  [CharStatus: HP:..]
STATUS_BLOCK = re.compile(
    r"\[(?:CharStatus:\s*)?HP:(?P<hp>[\d.]+)/(?P<max_hp>[\d.]+)\s+MP:(?P<mp>[\d.]+)/(?P<max_mp>[\d.]+)"
    r"\s+CP:(?P<cp>[\d.]+)/(?P<max_cp>[\d.]+)\s+Exp:(?P<exp>\d+)\s+Sp:(?P<sp>\d+)\s+Lvl:(?P<lvl>\d+)\]"
)

# P.Atk:6 M.Atk:2 ..  /  [CharStats: P.Atk:6 M.Atk:2 ..]
STATS_BLOCK = re.compile(
    r"(?:\[CharStats:\s*)?P\.Atk:(?P<p_atk>\d+)\s*M\.Atk:(?P<m_atk>\d+)\s*P\.Def:(?P<p_def>\d+)\s*M\.Def:(?P<m_def>\d+)"
    r"\s*Accur:(?P<accur>\d+)\s*Crit\.rate:(?P<crit_rate>\d+)\s*Evasion:(?P<evasion>\d+)\s*Atk\.Speed:(?P<atk_speed>[\d.]+)"
    r"\s*Cast\.Speed:(?P<cast_speed>\d+)\s*STR:(?P<str>\d+)\s*INT:(?P<int>\d+)\s*DEX:(?P<dex>\d+)\s*CON:(?P<con>\d+)\s*WIT:(?P<wit>\d+)"
)

EQUIP_BLOCK = re.compile(r"\[(?P<equip_str>Weapon:[^\]]+)\]")

POS_BLOCK = re.compile(
    r"\[Pos:(?P<pos_x>-?\d+),(?P<pos_y>-?\d+),(?P<pos_z>-?\d+)\s+Head:(?P<head>-?\d+)"
    r"\s+Sit:(?P<sit>true|false)\s+Cast:(?P<cast>true|false)\s+Atk:(?P<atk>true|false)"
    r"\s+Dead:(?P<dead>true|false)\s+Wt:(?P<wt>[\d.]+)%\]"
)

# Older logs: [Target:<object id> Monster Dist:39.8 LoS:true HP:6.2% THead:51262]
TARGET_BLOCK = re.compile(r"\[Target:(?P<target>[^\]]*)\]")
TARGET_DETAIL = re.compile(
    r"^(?P<object_id>\d+)\s+(?P<kind>Monster|Npc)\s+Dist:(?P<distance>[\d.]+)"
    r"\s+LoS:(?P<los>true|false)\s+HP:(?P<hp_pct>[\d.]+)%\s+THead:(?P<head>\d+)$"
)

# Newer logs: [Nav: Dist:66.3 LoS:true THead:33556] / [Nav:None]
NAV_BLOCK = re.compile(r"\[Nav:(?P<nav>[^\]]*)\]")
NAV_DETAIL = re.compile(r"^Dist:(?P<distance>[\d.]+)\s+LoS:(?P<los>true|false)\s+THead:(?P<head>\d+)$")

ENV_BLOCK = re.compile(r"\[Env:\s*(?P<env>.*?)\]\s*\[CD:")
ENV_META = re.compile(r"GroundItems:\s*(?P<ground_items>\d+),?\s*InCombat:\s*(?P<in_combat>true|false)")
ENV_NEARBY_LEGACY = re.compile(r"NearbyMon:\s*(?P<nearby_count>\d+)")
ENV_NEARBY_NEW = re.compile(r"Monsters Count:\s*(?P<nearby_count>\d+)")
ENV_MONSTER = re.compile(
    r"Monster:\s*(?P<name>[^(]+?)\s*\(NPC_ID:\s*(?P<npc_id>\d+),\s*Lvl:\s*(?P<lvl>\d+),\s*Distance:\s*(?P<distance>[\d.]+)\)"
)

CD_BLOCK = re.compile(r"\[CD:(?P<cooldown>[^\]]*)\]")
SKILL_LIST_BLOCK = re.compile(r"\[Active:(?P<active>[^\]]*)\]\s*\[Passive:(?P<passive>[^\]]*)\]")
SKILL_ENTRY = re.compile(r"(?P<id>\d+)-(?P<level>\d+)")

# Inline target identity of the newer layout: "Stone Golem (NPC_ID:20016, Lvl:13) | HP:181.4/339.0 (53.5%)"
TARGET_IDENTITY = re.compile(r"\(NPC_ID:\s*(?P<npc_id>\d+),\s*Lvl:\s*(?P<lvl>\d+)\)")
TARGET_HP = re.compile(r"\|\s*HP:(?P<hp>[\d.]+)/(?P<max_hp>[\d.]+)\s*\((?P<hp_pct>[\d.]+)%\)")

MESSAGE_LABEL = re.compile(r"^\[?(?P<label>[^\]:]+?)\]?:\s*")
# A name ends before the next block, the target identity, a distance pipe or "skill:".
NAME_STOP = r"(?=\s*\(NPC_ID:|\s*\(Lvl:|\s*\[|\s+\||\s+skill:|$)"
KILLED_NAME = re.compile(r"^(?P<name>.+?)(?=\s*\(NPC_ID:|\s*\(Lvl:|\s*\[dropped\]|\s+dropped:|\s*\[|\s+\||$)")

PICKUP_REF = re.compile(r"Pickup\s*-\s*(?P<name>.+?)\[(?P<item_id>\d+)\]")
DROP_REF = re.compile(r"dropped\]?:\s*(?P<name>.+?)\[(?P<count>\d+)\]")
DAMAGE_OUT = re.compile(
    r"^(?P<damage>\d+)(?:\s*\(MISS/BLOCK\))?\s*to:\s*(?P<name>.+?)(?:\s+skill:\s*(?P<skill>.+?))?" + NAME_STOP
)
DAMAGE_IN = re.compile(r"^(?P<damage>\d+)\s*from:\s*(?P<name>.+?)(?:\s+skill:\s*(?P<skill>.+?))?" + NAME_STOP)
USED_SKILL_REF = re.compile(r"^(?P<skill_name>.+?)\[(?P<skill_id>\d+)\]\s*on\s+(?P<name>.+?)" + NAME_STOP)
EFFECT_REF = re.compile(
    r"^(?P<effect_kind>[^-]+?)\s*-\s*(?P<effect_name>.+?)\[(?P<effect_id>\d+)\]\s*duration:(?P<effect_duration>\d+)s"
)
AUTOUSE_ITEM_REF = re.compile(r"enabled auto-use for item:\s*(?P<name>.+?)\s*\(Item ID:\s*(?P<item_id>\d+)\)")
DISABLED_AUTOUSE_ITEM_REF = re.compile(r"disabled auto-use for item:\s*(?P<name>.+?)\s*\(Item ID:\s*(?P<item_id>\d+)\)")
USED_ITEM_REF = re.compile(r"^used item:\s*(?P<name>.+?)\s*\[Remaining:")
BUY_REF = re.compile(
    r"^(?P<name>.+?)\s*\(ID:\s*(?P<item_id>\d+)\),\s*Count:\s*(?P<count>\d+),\s*Price:\s*(?P<price>\d+)\s*Adena"
    r"\s*\(NPC:\s*(?P<vendor_name>.+?),\s*ID:\s*(?P<vendor_id>\d+)\)"
)
LEARNED_SKILL_REF = re.compile(r"^(?P<skill_name>.+?)\s*\(ID:\s*(?P<skill_id>\d+),\s*Level:\s*(?P<level>\d+)\)")
TELEPORT_REF = re.compile(
    r"teleported from \[(?P<from_x>-?\d+),(?P<from_y>-?\d+),(?P<from_z>-?\d+)\]"
    r"\s*to\s*\[(?P<to_x>-?\d+),(?P<to_y>-?\d+),(?P<to_z>-?\d+)\]"
)
# Messages without a "label:" prefix (older/newer clients emit a few).
PREFIX_ACTIONS = (
    ("teleported", re.compile(r"^\[?teleported from\b")),
    ("died", re.compile(r"^\[?DIED at\b")),
    ("resurrected", re.compile(r"^\[?RESURRECTED at\b")),
)
DIED_KILLER_REF = re.compile(r"Killer:\s*(?P<name>.+?)\s*\(ID:\s*(?P<object_id>\d+)\)")
# ".. via [Method: Teleport]" / "[Method: Town Respawn]" on teleport / resurrect lines.
METHOD_BLOCK = re.compile(r"\[Method:\s*(?P<method>[^\]]*)\]")
GENERIC_NAME_RE = re.compile(r"^(?P<name>.+?)" + NAME_STOP)
MOVE_COORDS = re.compile(r"\[(?P<x>-?\d+),(?P<y>-?\d+),(?P<z>-?\d+)\]")

# ``[CD:1177:2466,1184:1234]`` — comma-separated ``skillId:remaining`` pairs.
CD_ENTRY = re.compile(r"(?P<skill_id>\d+):(?P<remaining>\d+)")
# Target knowledge blocks of the newer layout.
ENTITY_TYPE_BLOCK = re.compile(r"\[Type:\s*(?P<value>[^\]]*)\]")
ENTITY_WEAK_BLOCK = re.compile(r"\[Weak:\s*(?P<value>[^\]]*)\]")
ENTITY_RESIST_BLOCK = re.compile(r"\[Resist:\s*(?P<value>[^\]]*)\]")
# ``used item: X [Remaining: 3]`` — how many are left in the inventory.
USED_ITEM_REMAINING = re.compile(r"\[Remaining:\s*(?P<remaining>\d+)\]")
# ``learned skill: X (ID: 3, Level: 5) [Cost: 370 SP]``.
LEARNED_SKILL_COST = re.compile(r"\[Cost:\s*(?P<cost>\d+)\s*SP\]")
# ``<skill id="1" levels="37" name="Triple Slash" ..>`` in the skills datapack.
SKILL_XML_ENTRY = re.compile(r"<skill id=\"(?P<id>\d+)\"[^>]*?name=\"(?P<name>[^\"]*)\"")

# Action label (normalised: lower case, separators removed) -> action_type.
ACTION_LABELS = {
    "skillsupdated": "skills_updated",
    "selectedtarget": "selected_target",
    "attacked": "attacked",
    "inflicteddamage": "inflicted_damage",
    "tookdamage": "took_damage",
    "usedskill": "used_skill",
    "performedaction": "performed_action",
    "effectapplied": "effect_applied",
    "killed": "killed",
    "movedto": "moved",
    "dropped": "dropped",
    "enabledautouseforitem": "autouse_enabled",
    "disabledautouseforitem": "autouse_disabled",
    "useditem": "used_item",
    "buyitem": "buy",
    "learnedskill": "learned_skill",
    "teleportedfrom": "teleported",
}

# NPC types that represent something a player can actually fight.
_COMBAT_NPC_TYPES = frozenset({"Monster", "RaidBoss", "FestivalMonster", "EventMonster", "FeedableBeast"})

SKILL = "N/A"


def _clean(value: str | None) -> str | None:
    """Return ``value`` stripped, mapping missing/empty strings to ``None``."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _entity_name(value: str | None) -> str | None:
    """Normalise a captured entity name; the log's ``None`` sentinel means no entity."""
    value = _clean(value)
    if value is None or value.lower() == "none":
        return None
    return value


def _to_int(value: str | None) -> int | None:
    """Convert a captured field to int, returning ``None`` when not numeric."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str | None) -> float | None:
    """Convert a captured field to float, returning ``None`` when not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: str | None) -> bool | None:
    """Convert a "true"/"false" capture to bool, ``None`` when absent."""
    if value is None:
        return None
    return value.strip().lower() == "true"


def _npc_rank(row: dict) -> tuple:
    """Ranking used to pick one NPC when a name maps to several ids."""
    return (0 if row["type"] in _COMBAT_NPC_TYPES else 1, -(row["exp"] or 0), row["id"])


def _item_rank(row: dict) -> tuple:
    """Ranking used to pick one item when a name maps to several ids."""
    return (1 if row["is_questitem"] else 0, row["id"])


class EntityResolver:
    """Resolves log references against ``l2jc6_npcs.parquet`` / ``l2jc6_items.parquet``."""

    def __init__(self, npc_parquet: str | Path, item_parquet: str | Path):
        """Load both datapack tables and build the name/id indexes."""
        self.npcs_by_id: dict[int, dict] = {}
        self.npcs_by_name: dict[str, list[int]] = {}
        self.items_by_id: dict[int, dict] = {}
        self.items_by_name: dict[str, list[int]] = {}
        # Item ids observed in "Pickup" messages, reused for the same name later.
        self.observed_items: dict[str, int] = {}
        # Character names seen in the log (used for cross-player references).
        self.characters: set[str] = set()
        # Character name -> character_id, assigned once every name is known.
        self.character_ids: dict[str, int] = {}
        # Skill id -> name, from the logs and from the skills datapack.
        self.skill_names: dict[int, str] = {}
        self.unresolved: Counter = Counter()
        self.ambiguous: Counter = Counter()

        for row in pq.read_table(npc_parquet).to_pylist():
            self.npcs_by_id[row["id"]] = row
            self.npcs_by_name.setdefault(row["name"], []).append(row["id"])

        for row in pq.read_table(item_parquet).to_pylist():
            self.items_by_id[row["id"]] = row
            self.items_by_name.setdefault(row["name"], []).append(row["id"])

    def observe_item(self, name: str | None, item_id: int) -> None:
        """Remember an explicit item id so later name lookups can reuse it."""
        name = _clean(name)
        if name and name not in self.observed_items:
            self.observed_items[name] = item_id

    def observe_character(self, name: str | None) -> None:
        """Remember a character name so cross-player references resolve as characters."""
        name = _clean(name)
        if name:
            self.characters.add(name)

    def is_character(self, name: str | None) -> bool:
        """Return ``True`` when ``name`` is a player character seen in the logs."""
        return _clean(name) in self.characters

    def assign_character_ids(self) -> None:
        """Assign dense, stable ``character_id``s ordered by character name."""
        self.character_ids = {name: index + 1 for index, name in enumerate(sorted(self.characters))}

    def character_id(self, name: str | None) -> int | None:
        """Return the ``character_id`` of a character name, or ``None``."""
        return self.character_ids.get(_clean(name) or "")

    def observe_skill(self, skill_id: int | None, name: str | None) -> None:
        """Remember a skill name seen in the logs (the first spelling wins)."""
        name = _clean(name)
        if skill_id is not None and name and skill_id not in self.skill_names:
            self.skill_names[skill_id] = name

    def skill_name(self, skill_id: int | None) -> str | None:
        """Return the known name of a skill id, or ``None``."""
        return self.skill_names.get(skill_id) if skill_id is not None else None

    def load_skill_names(self, xml_dir: str | Path) -> int:
        """Fill in missing skill names from the datapack ``stats/skills`` definitions."""
        directory = Path(xml_dir)
        if not directory.is_dir():
            return 0

        added = 0
        for xml_file in sorted(directory.rglob("*.xml")):
            try:
                text = xml_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in SKILL_XML_ENTRY.finditer(text):
                skill_id = int(match.group("id"))
                if skill_id not in self.skill_names:
                    self.skill_names[skill_id] = match.group("name")
                    added += 1
        return added

    def resolve_npc(self, name: str | None = None, npc_id: int | None = None) -> dict | None:
        """Resolve an NPC reference; explicit ids win over name lookups."""
        if npc_id is not None and npc_id in self.npcs_by_id:
            row = self.npcs_by_id[npc_id]
            return {"id": row["id"], "name": row["name"], "level": row["level"], "type": row["type"], "ambiguous": False}

        name = _clean(name)
        if not name:
            return None

        candidates = self.npcs_by_name.get(name)
        if not candidates:
            return None

        best = min(candidates, key=lambda npc_id_: _npc_rank(self.npcs_by_id[npc_id_]))
        if len(candidates) > 1:
            self.ambiguous[f"npc:{name}"] += 1
        row = self.npcs_by_id[best]
        return {
            "id": row["id"],
            "name": row["name"],
            "level": row["level"],
            "type": row["type"],
            "ambiguous": len(candidates) > 1,
        }

    def resolve_item(self, name: str | None = None, item_id: int | None = None) -> dict | None:
        """Resolve an item reference; explicit ids win over learned/table lookups."""
        name = _clean(name)

        if item_id is not None:
            row = self.items_by_id.get(item_id)
            if row is None:
                # The client knows an item the datapack table does not contain.
                return {"id": item_id, "name": name, "grade": None, "ambiguous": False}
            return {"id": row["id"], "name": row["name"], "grade": row["crystal_type"], "ambiguous": False}

        if not name:
            return None

        observed = self.observed_items.get(name)
        if observed is not None and observed in self.items_by_id:
            row = self.items_by_id[observed]
            return {"id": observed, "name": row["name"], "grade": row["crystal_type"], "ambiguous": False}

        candidates = self.items_by_name.get(name)
        if not candidates:
            return None

        best = min(candidates, key=lambda item_id_: _item_rank(self.items_by_id[item_id_]))
        if len(candidates) > 1:
            self.ambiguous[f"item:{name}"] += 1
        row = self.items_by_id[best]
        return {"id": row["id"], "name": row["name"], "grade": row["crystal_type"], "ambiguous": len(candidates) > 1}


EQUIP_SLOT_RE = re.compile(rf"(?P<slot>{'|'.join(SLOTS)}):(?P<value>.*?)(?=\s+(?:{'|'.join(SLOTS)}):|$)")


def _apply_npc(row: dict, info: dict) -> None:
    """Write resolved NPC information into the row's link columns."""
    row["entity_name"] = info["name"]
    row["entity_kind"] = "npc"
    row["entity_npc_id"] = info["id"]
    row["entity_npc_level"] = info["level"]
    row["entity_npc_type"] = info["type"]
    row["entity_ambiguous"] = info["ambiguous"]


def _apply_item(row: dict, info: dict) -> None:
    """Write resolved item information into the row's link columns."""
    row["entity_name"] = info["name"]
    row["entity_kind"] = "item"
    row["entity_item_id"] = info["id"]
    row["entity_item_name"] = info["name"]
    row["entity_item_grade"] = info["grade"]
    row["entity_ambiguous"] = info["ambiguous"]


def _link_npc(row: dict, resolver: EntityResolver, name: str | None, npc_id: int | None = None) -> None:
    """Link an NPC reference (explicit id or name)."""
    name = _entity_name(name)
    if name is None and npc_id is None:
        return

    info = resolver.resolve_npc(name=name, npc_id=npc_id)
    if info is None:
        resolver.unresolved[f"npc:{name}"] += 1
        row["entity_name"] = name
        return
    _apply_npc(row, info)


def _link_item(
    row: dict,
    resolver: EntityResolver,
    name: str | None = None,
    item_id: int | None = None,
    count: int | None = None,
) -> None:
    """Link an item reference (explicit id, learned id or name) and its quantity."""
    name = _entity_name(name)
    if name is None and item_id is None:
        return

    row["entity_item_count"] = count
    info = resolver.resolve_item(name=name, item_id=item_id)
    if info is None:
        resolver.unresolved[f"item:{name}"] += 1
        row["entity_name"] = name
        return
    _apply_item(row, info)


def _link_entity(
    row: dict,
    resolver: EntityResolver,
    name: str | None,
    character: str,
    npc_id: int | None = None,
) -> None:
    """Link a generic action argument: a player, NPC, then item."""
    name = _entity_name(name)
    if name is None and npc_id is None:
        return

    # The acting character itself, or another player (party member, PvP target,
    # killer). Player characters have no datapack row, so only the name is kept.
    if name is not None and (name == character or resolver.is_character(name)):
        row["entity_name"] = name
        row["entity_kind"] = "character"
        row["entity_ambiguous"] = False
        return

    info = resolver.resolve_npc(name=name, npc_id=npc_id)
    if info is not None:
        _apply_npc(row, info)
        return

    info = resolver.resolve_item(name=name)
    if info is not None:
        _apply_item(row, info)
        return

    resolver.unresolved[f"entity:{name}"] += 1
    row["entity_name"] = name
    row["entity_kind"] = "unknown"


def _parse_equipment(equip_str: str, resolver: EntityResolver) -> tuple[dict, dict]:
    """Split the equipment block into slot names and their resolved item ids."""
    names: dict[str, str | None] = dict.fromkeys(SLOTS)
    ids: dict[str, int | None] = dict.fromkeys(SLOTS)

    for match in EQUIP_SLOT_RE.finditer(equip_str):
        value = match.group("value").strip()
        if not value or value == "None":
            continue
        slot = match.group("slot")
        names[slot] = value
        info = resolver.resolve_item(name=value)
        ids[slot] = info["id"] if info else None

    return names, ids


def _parse_skill_list(raw: str | None) -> list[dict] | None:
    """Parse ``1322-1,194-1`` skill lists from ``skills updated`` messages."""
    entries = [{"id": int(m.group("id")), "level": int(m.group("level"))} for m in SKILL_ENTRY.finditer(raw or "")]
    return entries or None


# Every key of a parsed row, in schema order. Rows are initialised from this
# tuple so that all columns exist even when a block is missing from the line.
ROW_FIELDS = (
    # provenance
    "source_file",
    "line_no",
    "timestamp",
    "character_id",
    "action_raw",
    "action_type",
    "action_method",
    # links resolved against l2jc6_npcs.parquet / l2jc6_items.parquet
    "entity_name",
    "entity_object_id",
    "entity_kind",
    "entity_ambiguous",
    "entity_npc_id",
    "entity_npc_level",
    "entity_npc_type",
    "entity_type",
    "entity_weak",
    "entity_resist",
    "entity_item_id",
    "entity_item_name",
    "entity_item_grade",
    "entity_item_count",
    "item_remaining",
    # loot of "killed" events (second linked entity)
    "drop_item_id",
    "drop_item_name",
    "drop_item_count",
    # shop purchase (item + vendor NPC)
    "vendor_npc_id",
    "vendor_npc_name",
    "item_price",
    # action payload
    "damage",
    "is_miss",
    "skill_id",
    "skill_name",
    "skill_level",
    "skill_cost",
    "effect_kind",
    "effect_id",
    "effect_name",
    "effect_duration",
    # selected target / navigation
    "selected_object_id",
    "selected_kind",
    "selected_distance",
    "selected_los",
    "selected_head",
    "selected_hp_pct",
    "target_hp",
    "target_max_hp",
    # character status
    "hp",
    "max_hp",
    "mp",
    "max_mp",
    "cp",
    "max_cp",
    "exp",
    "sp",
    "lvl",
    "char_stats",
    "equipment",
    "equipment_ids",
    # position
    "pos_x",
    "pos_y",
    "pos_z",
    "head",
    "is_sitting",
    "is_casting",
    "is_attacking",
    "is_dead",
    "weight_pct",
    "move_x",
    "move_y",
    "move_z",
    "teleport_from_x",
    "teleport_from_y",
    "teleport_from_z",
    # environment
    "nearby_count",
    "nearby_objects",
    "ground_items",
    "in_combat",
    "cooldown_raw",
    "cooldowns",
    "active_skills",
    "passive_skills",
    "known_skills",
    # summon (format 2): its state while the character has one, the owner's command
    "pet",
    "pet_command",
    # format 2: the summon dealt the damage / the killing blow; every drop of a kill
    "by_pet",
    "drops",
)


def _parse_blocks(row: dict, message: str, action_type: str, resolver: EntityResolver) -> None:
    """Extract the status / position / environment blocks shared by both layouts."""
    status = STATUS_BLOCK.search(message)
    if status:
        row["hp"] = float(status.group("hp"))
        row["max_hp"] = float(status.group("max_hp"))
        row["mp"] = float(status.group("mp"))
        row["max_mp"] = float(status.group("max_mp"))
        row["cp"] = float(status.group("cp"))
        row["max_cp"] = float(status.group("max_cp"))
        row["exp"] = int(status.group("exp"))
        row["sp"] = int(status.group("sp"))
        row["lvl"] = int(status.group("lvl"))

    stats = STATS_BLOCK.search(message)
    if stats:
        row["char_stats"] = {
            "p_atk": int(stats.group("p_atk")),
            "m_atk": int(stats.group("m_atk")),
            "p_def": int(stats.group("p_def")),
            "m_def": int(stats.group("m_def")),
            "accur": int(stats.group("accur")),
            "crit_rate": int(stats.group("crit_rate")),
            "evasion": int(stats.group("evasion")),
            "atk_speed": float(stats.group("atk_speed")),
            "cast_speed": int(stats.group("cast_speed")),
            "str": int(stats.group("str")),
            "int": int(stats.group("int")),
            "dex": int(stats.group("dex")),
            "con": int(stats.group("con")),
            "wit": int(stats.group("wit")),
        }

    pos = POS_BLOCK.search(message)
    if pos:
        row["pos_x"] = int(pos.group("pos_x"))
        row["pos_y"] = int(pos.group("pos_y"))
        row["pos_z"] = int(pos.group("pos_z"))
        row["head"] = int(pos.group("head"))
        row["is_sitting"] = pos.group("sit").lower() == "true"
        row["is_casting"] = pos.group("cast").lower() == "true"
        row["is_attacking"] = pos.group("atk").lower() == "true"
        row["is_dead"] = pos.group("dead").lower() == "true"
        row["weight_pct"] = float(pos.group("wt"))

    equip = EQUIP_BLOCK.search(message)
    if equip:
        names, ids = _parse_equipment(equip.group("equip_str"), resolver)
        row["equipment"] = names
        row["equipment_ids"] = ids

    # Older logs carry the selection in [Target:..], newer ones in [Nav:..].
    target = TARGET_BLOCK.search(message)
    if target:
        detail = TARGET_DETAIL.match(target.group("target").strip())
        if detail:
            row["selected_object_id"] = int(detail.group("object_id"))
            row["selected_kind"] = detail.group("kind")
            row["selected_distance"] = float(detail.group("distance"))
            row["selected_los"] = detail.group("los").lower() == "true"
            row["selected_head"] = int(detail.group("head"))
            row["selected_hp_pct"] = float(detail.group("hp_pct"))
    else:
        nav = NAV_BLOCK.search(message)
        if nav:
            detail = NAV_DETAIL.match(nav.group("nav").strip())
            if detail:
                row["selected_distance"] = float(detail.group("distance"))
                row["selected_los"] = detail.group("los").lower() == "true"
                row["selected_head"] = int(detail.group("head"))

    env = ENV_BLOCK.search(message)
    if env:
        env_str = env.group("env")
        meta = ENV_META.search(env_str)
        if meta:
            row["ground_items"] = int(meta.group("ground_items"))
            row["in_combat"] = meta.group("in_combat").lower() == "true"
        nearby = ENV_NEARBY_LEGACY.search(env_str) or ENV_NEARBY_NEW.search(env_str)
        if nearby:
            row["nearby_count"] = int(nearby.group("nearby_count"))
        monsters = [
            {
                "type": "monster",
                "name": monster.group("name").strip(),
                "id": int(monster.group("npc_id")),
                "lvl": int(monster.group("lvl")),
                "distance": float(monster.group("distance")),
            }
            for monster in ENV_MONSTER.finditer(env_str)
        ]
        if monsters:
            row["nearby_objects"] = monsters

    cd = CD_BLOCK.search(message)
    if cd:
        cooldown = _clean(cd.group("cooldown"))
        row["cooldown_raw"] = None if cooldown in (None, "None") else cooldown
        # ``[CD:1177:2466,1184:1234]`` -> one entry per skill currently on cooldown.
        entries = [
            {"skill_id": int(entry.group("skill_id")), "remaining": int(entry.group("remaining"))}
            for entry in CD_ENTRY.finditer(cooldown or "")
        ]
        if entries:
            row["cooldowns"] = entries

    # Target knowledge printed by the newer client layout.
    target_type = ENTITY_TYPE_BLOCK.search(message)
    if target_type:
        row["entity_type"] = _clean(target_type.group("value"))
    target_weak = ENTITY_WEAK_BLOCK.search(message)
    if target_weak:
        row["entity_weak"] = _clean(target_weak.group("value"))
    target_resist = ENTITY_RESIST_BLOCK.search(message)
    if target_resist:
        row["entity_resist"] = _clean(target_resist.group("value"))

    target_hp = TARGET_HP.search(message)
    if target_hp:
        row["target_hp"] = float(target_hp.group("hp"))
        row["target_max_hp"] = float(target_hp.group("max_hp"))

    remaining = USED_ITEM_REMAINING.search(message)
    if remaining:
        row["item_remaining"] = int(remaining.group("remaining"))

    if action_type == "learned_skill":
        cost = LEARNED_SKILL_COST.search(message)
        if cost:
            row["skill_cost"] = int(cost.group("cost"))

    method = METHOD_BLOCK.search(message)
    if method:
        row["action_method"] = _clean(method.group("method"))

    if action_type == "skills_updated":
        skill_list = SKILL_LIST_BLOCK.search(message)
        if skill_list:
            row["active_skills"] = _parse_skill_list(skill_list.group("active"))
            row["passive_skills"] = _parse_skill_list(skill_list.group("passive"))


def _link_drop(row: dict, resolver: EntityResolver, name: str | None, count: int) -> None:
    """Link the loot item of a ``killed`` event."""
    row["drop_item_count"] = count
    info = resolver.resolve_item(name=name)
    if info is None:
        resolver.unresolved[f"drop:{_clean(name)}"] += 1
        row["drop_item_name"] = _clean(name)
        return
    row["drop_item_id"] = info["id"]
    row["drop_item_name"] = info["name"]


def _apply_payload(row: dict, body: str, action_type: str, resolver: EntityResolver, character: str) -> None:
    """Extract the action arguments and link the entity they refer to."""
    identity = TARGET_IDENTITY.search(body)
    npc_id = int(identity.group("npc_id")) if identity else None

    if action_type in ("selected_target", "attacked"):
        match = GENERIC_NAME_RE.match(body)
        _link_entity(row, resolver, name=match.group("name") if match else body, character=character, npc_id=npc_id)
        return

    if action_type in ("autouse_enabled", "autouse_disabled", "used_item"):
        if action_type == "used_item":
            match = USED_ITEM_REF.match(body)
        elif action_type == "autouse_enabled":
            match = AUTOUSE_ITEM_REF.search(body)
        else:
            match = DISABLED_AUTOUSE_ITEM_REF.search(body)
        if match:
            item_id = _to_int(match.groupdict().get("item_id"))
            if item_id is not None:
                resolver.observe_item(match.group("name"), item_id)
            _link_item(row, resolver, name=match.group("name"), item_id=item_id)
        return

    if action_type == "buy":
        match = BUY_REF.match(body)
        if match:
            item_id = int(match.group("item_id"))
            resolver.observe_item(match.group("name"), item_id)
            _link_item(row, resolver, name=match.group("name"), item_id=item_id, count=int(match.group("count")))
            row["item_price"] = int(match.group("price"))
            vendor_id = int(match.group("vendor_id"))
            vendor = resolver.resolve_npc(npc_id=vendor_id)
            row["vendor_npc_id"] = vendor_id
            row["vendor_npc_name"] = vendor["name"] if vendor else _clean(match.group("vendor_name"))
        return

    if action_type == "learned_skill":
        match = LEARNED_SKILL_REF.match(body)
        if match:
            row["skill_name"] = _clean(match.group("skill_name"))
            row["skill_id"] = int(match.group("skill_id"))
            row["skill_level"] = int(match.group("level"))
        return

    if action_type == "teleported":
        match = TELEPORT_REF.search(body)
        if match:
            # Both ends of the jump: the source stays available as well.
            row["teleport_from_x"] = int(match.group("from_x"))
            row["teleport_from_y"] = int(match.group("from_y"))
            row["teleport_from_z"] = int(match.group("from_z"))
            row["move_x"] = int(match.group("to_x"))
            row["move_y"] = int(match.group("to_y"))
            row["move_z"] = int(match.group("to_z"))
        return

    if action_type in ("died", "resurrected"):
        # Both carry the location in "at [x,y,z]" (death spot / respawn point).
        coords = MOVE_COORDS.search(body)
        if coords:
            row["move_x"] = int(coords.group("x"))
            row["move_y"] = int(coords.group("y"))
            row["move_z"] = int(coords.group("z"))

        if action_type == "died":
            # The killer is identified by an in-world object id, so link it by name only.
            match = DIED_KILLER_REF.search(body)
            if match:
                _link_entity(row, resolver, name=match.group("name"), character=character)
        return

    if action_type == "killed":
        match = KILLED_NAME.match(body)
        _link_entity(row, resolver, name=match.group("name") if match else body, character=character, npc_id=npc_id)
        drop = DROP_REF.search(body)
        if drop:
            _link_drop(row, resolver, name=drop.group("name"), count=int(drop.group("count")))
        return

    if action_type in ("inflicted_damage", "took_damage"):
        pattern = DAMAGE_OUT if action_type == "inflicted_damage" else DAMAGE_IN
        match = pattern.match(body)
        if match:
            row["damage"] = int(match.group("damage"))
            row["is_miss"] = "(MISS/BLOCK)" in match.group(0)
            skill = _clean(match.group("skill"))
            row["skill_name"] = None if skill in (None, SKILL) else skill
            _link_entity(row, resolver, name=match.group("name"), character=character, npc_id=npc_id)
        return

    if action_type == "used_skill":
        match = USED_SKILL_REF.match(body)
        if match:
            row["skill_name"] = _clean(match.group("skill_name"))
            row["skill_id"] = int(match.group("skill_id"))
            _link_entity(row, resolver, name=match.group("name"), character=character, npc_id=npc_id)
        return

    if action_type == "pickup":
        match = PICKUP_REF.search(body)
        if match:
            item_id = int(match.group("item_id"))
            resolver.observe_item(match.group("name"), item_id)
            _link_item(row, resolver, name=match.group("name"), item_id=item_id)
        return

    if action_type == "effect_applied":
        match = EFFECT_REF.match(body)
        if match:
            row["effect_kind"] = _clean(match.group("effect_kind"))
            row["effect_name"] = _clean(match.group("effect_name"))
            row["effect_id"] = int(match.group("effect_id"))
            row["effect_duration"] = int(match.group("effect_duration"))
        return

    if action_type == "moved":
        match = MOVE_COORDS.search(body)
        if match:
            row["move_x"] = int(match.group("x"))
            row["move_y"] = int(match.group("y"))
            row["move_z"] = int(match.group("z"))


def _split_character(rest: str) -> tuple[str | None, str]:
    """
    Split the optional character prefix from the message text.

    A bare name is only accepted when an uppercase identifier is followed by a
    labelled message (``HumanFighter00 moved to: ..``), so character-less lines
    such as ``used item: X`` or ``BUY Item: X`` stay intact.
    """
    rest = CLOCK_PREFIX.sub("", rest)

    bracketed = BRACKETED_CHARACTER.match(rest)
    if bracketed is not None:
        return bracketed.group("character"), bracketed.group("rest")

    bare = BARE_CHARACTER.match(rest)
    if bare is not None and bare.group("character")[:1].isupper():
        remainder = bare.group("rest")
        label = MESSAGE_LABEL.match(remainder)
        known = label is not None and re.sub(r"[^a-z]", "", label.group("label").lower()) in ACTION_LABELS
        if known or remainder.startswith("["):
            return bare.group("character"), remainder

    return None, rest


def _prefix_action(message: str) -> str | None:
    """Classify the few messages that carry no ``label:`` prefix (``teleported from ..``)."""
    for action, pattern in PREFIX_ACTIONS:
        if pattern.match(message):
            return action
    return None


# Format-2 event -> action_type (the same labels as format 1, see ACTION_LABELS).
V2_ACTIONS = {
    "TARGET": "selected_target", "MOVE": "moved", "ATTACK": "attacked", "SKILL": "used_skill",
    "DMG": "inflicted_damage", "HIT": "took_damage", "KILL": "killed", "PICKUP": "pickup",
    "ACTION": "performed_action", "ITEM": "used_item", "EFFECT": "effect_applied", "EFFECTS": "active_effects",
    "SHOT": "shot_used", "BUY": "buy", "SELL": "sell", "LEARN": "learned_skill", "TELEPORT": "teleported",
    "DIED": "died", "REVIVED": "resurrected", "SKILLS": "skills_updated", "NOTE": "other",
    "SUMMON": "summoned", "UNSUMMON": "unsummoned", "PETDIED": "pet_died", "PETCMD": "pet_command",
    "PETSKILL": "pet_used_skill", "PETHIT": "pet_took_damage",
}
V2_SLOTS = dict(zip(v2.SLOT_KEYS, SLOTS))
V2_STATS = {"patk": "p_atk", "matk": "m_atk", "pdef": "p_def", "mdef": "m_def", "acc": "accur", "crit": "crit_rate",
            "eva": "evasion", "aspd": "atk_speed", "cspd": "cast_speed", "str": "str", "int": "int", "dex": "dex",
            "con": "con", "wit": "wit"}
V2_DAMAGE = re.compile(r"^(?P<damage>miss|\d+) (?:->|<-) (?P<ref>.+) skill (?P<skill>.+?)(?P<by_pet> by pet)?$")
V2_KILL = re.compile(r"^(?P<ref>.+?)(?P<by_pet> by pet)? drop (?P<drops>.+)$")
V2_DROP = re.compile(r"^(?P<item>.+)\[(?P<count>\d+)\] (?P<how>auto|ground)$")
V2_TRADE = re.compile(r"^(?P<item>.+)\[(?P<item_id>\d+)\] x(?P<count>\d+) price (?P<price>\d+) npc (?P<npc>.+)\[(?P<npc_id>\d+)\]$")
V2_LEARN = re.compile(r"^(?P<skill>.+)\[(?P<skill_id>\d+)\] L(?P<level>\d+) cost (?P<sp>\d+)sp")
V2_POINT = re.compile(r"(-?\d+),(-?\d+),(-?\d+)")


def _v2_link_ref(row: dict, resolver: EntityResolver, ref: dict | None, character: str) -> None:
    """Link a format-2 reference: authoritative npc / item ids, the world object id and the target HP."""
    if ref is None:
        return
    if ref["kind"] == "item":
        _link_item(row, resolver, name=ref["name"], item_id=ref.get("item_id"))
    else:
        _link_entity(row, resolver, name=ref["name"], character=character, npc_id=ref.get("npc_id"))
    row["entity_object_id"] = ref.get("object_id")
    if "hp" in ref:
        row["target_hp"], row["target_max_hp"] = float(ref["hp"]), float(ref["max_hp"])


def _v2_apply_state(row: dict, state: v2.CharacterState) -> None:
    """The character's last STATS / EQUIP / SKILLS on every row, as format 1 had them on every line."""
    stats = state.stats
    if stats:
        row["lvl"] = stats.get("level")
        row["exp"] = stats.get("exp")
        row["sp"] = stats.get("sp")
        values = {old: stats.get(new) for new, old in V2_STATS.items()}
        if values["atk_speed"] is not None:
            values["atk_speed"] = float(values["atk_speed"])
        row["char_stats"] = values
    if state.equip:
        row["equipment"] = {V2_SLOTS[slot]: name for slot, (name, _) in state.equip.items()}
        row["equipment_ids"] = {V2_SLOTS[slot]: item_id for slot, (_, item_id) in state.equip.items()}


def parse_v2_line(line: str, resolver: EntityResolver, source_file: str, line_no: int,
                  states: dict[str, v2.CharacterState]) -> dict | None:
    """Parse a format-2 line (see ``playeraction_v2.py``); STATS / EQUIP lines only update the state."""
    m = v2.LINE.match(line)
    character, event, rest = m.group("character"), m.group("event"), m.group("rest") or ""
    state = states.setdefault(character, v2.CharacterState())
    if event == "STATS":
        state.stats = v2.parse_stats(rest)
        return None
    if event == "EQUIP":
        state.equip = {slot: (name, item_id if item_id is not None else (
            (resolver.resolve_item(name=name) or {}).get("id") if name else None))
            for slot, (name, item_id) in v2.parse_equip(rest).items()}
        return None

    row: dict = dict.fromkeys(ROW_FIELDS)
    row["source_file"] = source_file
    row["line_no"] = line_no
    row["timestamp"] = f"{m.group('date')} {m.group('time').replace('.', ',')}"
    row["character_id"] = resolver.character_id(character)
    row["action_raw"] = f"{event} {rest}".strip()
    action_type = V2_ACTIONS.get(event, "other")
    payload, blocks = v2.split_tail(rest)
    if event == "AUTOSHOT":
        action_type = "autouse_enabled" if payload.startswith("on ") else "autouse_disabled"
    row["action_type"] = action_type
    _v2_apply_state(row, state)

    if "me" in blocks:
        me = v2.parse_me(blocks["me"])
        for key in ("hp", "max_hp", "mp", "max_mp", "cp", "max_cp"):
            row[key] = me.get(key)
        row["lvl"] = me.get("level", row["lvl"])
        row["pos_x"], row["pos_y"], row["pos_z"] = me.get("x"), me.get("y"), me.get("z")
        row["head"] = me.get("heading")
        row["weight_pct"] = me.get("weight_pct")
        row["is_sitting"], row["is_casting"] = me["sit"], me["cast"]
        row["is_attacking"], row["is_dead"], row["in_combat"] = me["atk"], me["dead"], me["combat"]
    if "sel" in blocks:
        sel = v2.parse_sel(blocks["sel"])
        row["selected_distance"], row["selected_los"], row["selected_head"] = (
            sel.get("distance"), sel.get("los"), sel.get("heading"))
    if "env" in blocks:
        env = v2.parse_env(blocks["env"])
        row["ground_items"] = env["items"]
        monsters = []
        for mob in env["monsters"]:
            info = resolver.resolve_npc(npc_id=mob["npc_id"])
            monsters.append({"type": "monster", "name": info["name"] if info else None, "id": mob["npc_id"],
                             "lvl": mob["level"], "distance": mob["distance"], "object_id": mob["object_id"],
                             "hp_pct": mob["hp_pct"], "attackers": mob["attackers"] or None})
        row["nearby_count"] = len(monsters)
        row["nearby_objects"] = monsters or None
    if "pet" in blocks:
        row["pet"] = v2.parse_pet(blocks["pet"])
    if "cd" in blocks:
        row["cooldown_raw"] = blocks["cd"]
        row["cooldowns"] = [{"skill_id": sid, "remaining": ms} for sid, ms in v2.parse_cd(blocks["cd"])] or None

    if event in ("TARGET", "ATTACK"):
        _v2_link_ref(row, resolver, v2.parse_ref(payload), character)
    elif event == "MOVE":
        point = V2_POINT.match(payload)
        if point:
            row["move_x"], row["move_y"], row["move_z"] = (int(v) for v in point.groups())
    elif event in ("SKILL", "PETSKILL"):
        skill, _, ref = payload.partition(" -> ")
        row["skill_name"], row["skill_id"] = v2.named_id(skill)
        _v2_link_ref(row, resolver, v2.parse_ref(ref), character)
    elif event in ("DMG", "HIT", "PETHIT"):
        dmg = V2_DAMAGE.match(payload)
        if dmg:
            if dmg.group("by_pet"):
                row["action_type"] = action_type = "pet_inflicted_damage"
                row["by_pet"] = True
            row["is_miss"] = dmg.group("damage") == "miss"
            row["damage"] = 0 if row["is_miss"] else int(dmg.group("damage"))
            if dmg.group("skill") != "-":
                row["skill_name"], row["skill_id"] = v2.named_id(dmg.group("skill"))
            _v2_link_ref(row, resolver, v2.parse_ref(dmg.group("ref")), character)
    elif event in ("SUMMON", "UNSUMMON"):
        _v2_link_ref(row, resolver, v2.parse_ref(payload), character)
    elif event == "PETDIED":
        _, _, killer = payload.partition(" <- ")
        _v2_link_ref(row, resolver, v2.parse_ref(killer), character)
    elif event == "PETCMD":
        command, _, ref = payload.partition(" -> ")
        row["pet_command"] = command
        _v2_link_ref(row, resolver, v2.parse_ref(ref), character)
    elif event == "KILL":
        kill = V2_KILL.match(payload)
        if kill:
            _v2_link_ref(row, resolver, v2.parse_ref(kill.group("ref")), character)
            row["by_pet"] = bool(kill.group("by_pet"))
            # one line per kill with every drop ("A[1] ground; B[2] auto", "-" for none); older
            # lines carry a single drop, one line per dropped item
            drops = [d for d in (V2_DROP.match(part) for part in kill.group("drops").split("; ")) if d]
            if drops:
                _link_drop(row, resolver, name=drops[0].group("item"), count=int(drops[0].group("count")))
                row["drops"] = [{"name": d.group("item"), "id": (resolver.resolve_item(name=d.group("item")) or {}).get("id"),
                                 "count": int(d.group("count")), "auto": d.group("how") == "auto"} for d in drops]
    elif event in ("PICKUP", "AUTOSHOT"):
        name, item_id = v2.named_id(payload.split(" ", 1)[1] if event == "AUTOSHOT" else payload)
        if item_id is not None:
            resolver.observe_item(name, item_id)
        _link_item(row, resolver, name=name, item_id=item_id)
    elif event == "ITEM":
        item, _, left = payload.rpartition(" left ")
        name, item_id = v2.named_id(item)
        _link_item(row, resolver, name=name, item_id=item_id)
        row["item_remaining"] = _to_int(left)
    elif event == "EFFECT":
        kind, _, rest_effect = payload.partition(" ")
        skill, _, duration = rest_effect.rpartition(" ")
        row["effect_kind"] = kind.capitalize()
        row["effect_name"], row["effect_id"] = v2.named_id(skill)
        row["effect_duration"] = _to_int(duration.rstrip("s"))
    elif event in ("BUY", "SELL"):
        trade = V2_TRADE.match(payload)
        if trade:
            item_id = int(trade.group("item_id"))
            resolver.observe_item(trade.group("item"), item_id)
            _link_item(row, resolver, name=trade.group("item"), item_id=item_id, count=int(trade.group("count")))
            row["item_price"] = int(trade.group("price"))
            row["vendor_npc_id"] = int(trade.group("npc_id"))
            vendor = resolver.resolve_npc(npc_id=row["vendor_npc_id"])
            row["vendor_npc_name"] = vendor["name"] if vendor else trade.group("npc")
    elif event == "LEARN":
        learn = V2_LEARN.match(payload)
        if learn:
            row["skill_name"], row["skill_id"] = learn.group("skill"), int(learn.group("skill_id"))
            row["skill_level"], row["skill_cost"] = int(learn.group("level")), int(learn.group("sp"))
    elif event == "TELEPORT":
        points = V2_POINT.findall(payload)
        if len(points) == 2:
            row["teleport_from_x"], row["teleport_from_y"], row["teleport_from_z"] = (int(v) for v in points[0])
            row["move_x"], row["move_y"], row["move_z"] = (int(v) for v in points[1])
        row["action_method"] = payload.rpartition(" via ")[2] or None
    elif event in ("DIED", "REVIVED"):
        point = V2_POINT.search(payload)
        if point:
            row["move_x"], row["move_y"], row["move_z"] = (int(v) for v in point.groups())
        if event == "DIED":
            killer = payload.partition(" killer ")[2]
            if killer:
                _link_entity(row, resolver, name=v2.named_id(killer)[0], character=character)
        else:
            row["action_method"] = payload.rpartition(" via ")[2] or None
    elif event == "SKILLS":
        active, passive = v2.parse_skills(payload)
        state.active, state.passive = active, passive
        row["active_skills"] = [{"id": sid, "level": lvl} for sid, lvl in active] or None
        row["passive_skills"] = [{"id": sid, "level": lvl} for sid, lvl in passive] or None

    if action_type == "effect_applied":
        resolver.observe_skill(row["effect_id"], row["effect_name"])
    else:
        resolver.observe_skill(row["skill_id"], row["skill_name"])
    return row


def parse_log_line(line: str, resolver: EntityResolver, source_file: str, line_no: int,
                   states: dict | None = None) -> dict | None:
    """Parse one log line (format 1 or 2) into a flat row, or ``None`` when it is not an action line."""
    if v2.is_v2(line.rstrip("\n")):
        return parse_v2_line(line.rstrip("\n"), resolver, source_file, line_no, states if states is not None else {})
    start = LINE_START.match(line)
    if start is None:
        return None

    character, message = _split_character(start.group("rest").strip())

    row: dict = dict.fromkeys(ROW_FIELDS)
    row["source_file"] = source_file
    row["line_no"] = line_no
    row["timestamp"] = start.group("timestamp")
    # Characters live in ``l2jc6_characters.parquet``; only the id is stored here.
    row["character_id"] = resolver.character_id(character)
    row["action_raw"] = message

    # "[attacked]: Stone Golem .." -> label "attacked"; "attacked: .." -> same.
    label_match = MESSAGE_LABEL.match(message)
    label = label_match.group("label") if label_match else ""
    body = message[label_match.end():] if label_match else message

    action_type = ACTION_LABELS.get(re.sub(r"[^a-z]", "", label.lower()), "other")
    if action_type == "performed_action" and body.startswith("Pickup"):
        action_type = "pickup"
    if action_type == "other":
        action_type = _prefix_action(message) or "other"
    row["action_type"] = action_type

    _parse_blocks(row, message, action_type, resolver)
    _apply_payload(row, body, action_type, resolver, character or "")

    # Remember skill names so the skill tables can carry readable names.
    if action_type == "effect_applied":
        resolver.observe_skill(row["effect_id"], row["effect_name"])
    else:
        resolver.observe_skill(row["skill_id"], row["skill_name"])

    return row


def _build_schema() -> pa.Schema:
    """Build the explicit Arrow schema of the action table."""
    nearby_objects = pa.list_(pa.struct([
        ("type", pa.string()),
        ("name", pa.string()),
        ("id", pa.int64()),
        ("lvl", pa.int64()),
        ("distance", pa.float64()),
        # format 2 only: the monster's world object id, HP % and the other players attacking it
        ("object_id", pa.int64()),
        ("hp_pct", pa.float64()),
        ("attackers", pa.list_(pa.string())),
    ]))
    skill_list = pa.list_(pa.struct([("id", pa.int64()), ("level", pa.int64())]))
    cooldown_list = pa.list_(pa.struct([("skill_id", pa.int64()), ("remaining", pa.int64())]))
    char_stats = pa.struct([
        ("p_atk", pa.int64()),
        ("m_atk", pa.int64()),
        ("p_def", pa.int64()),
        ("m_def", pa.int64()),
        ("accur", pa.int64()),
        ("crit_rate", pa.int64()),
        ("evasion", pa.int64()),
        ("atk_speed", pa.float64()),
        ("cast_speed", pa.int64()),
        ("str", pa.int64()),
        ("int", pa.int64()),
        ("dex", pa.int64()),
        ("con", pa.int64()),
        ("wit", pa.int64()),
    ])

    fields = [
        ("source_file", pa.string()),
        ("line_no", pa.int64()),
        ("timestamp", pa.string()),
        ("character_id", pa.int64()),
        ("action_raw", pa.string()),
        ("action_type", pa.string()),
        ("action_method", pa.string()),
        ("entity_name", pa.string()),
        ("entity_object_id", pa.int64()),
        ("entity_kind", pa.string()),
        ("entity_ambiguous", pa.bool_()),
        ("entity_npc_id", pa.int64()),
        ("entity_npc_level", pa.int64()),
        ("entity_npc_type", pa.string()),
        ("entity_type", pa.string()),
        ("entity_weak", pa.string()),
        ("entity_resist", pa.string()),
        ("entity_item_id", pa.int64()),
        ("entity_item_name", pa.string()),
        ("entity_item_grade", pa.string()),
        ("entity_item_count", pa.int64()),
        ("item_remaining", pa.int64()),
        ("drop_item_id", pa.int64()),
        ("drop_item_name", pa.string()),
        ("drop_item_count", pa.int64()),
        ("vendor_npc_id", pa.int64()),
        ("vendor_npc_name", pa.string()),
        ("item_price", pa.int64()),
        ("damage", pa.int64()),
        ("is_miss", pa.bool_()),
        ("skill_id", pa.int64()),
        ("skill_name", pa.string()),
        ("skill_level", pa.int64()),
        ("skill_cost", pa.int64()),
        ("effect_kind", pa.string()),
        ("effect_id", pa.int64()),
        ("effect_name", pa.string()),
        ("effect_duration", pa.int64()),
        ("selected_object_id", pa.int64()),
        ("selected_kind", pa.string()),
        ("selected_distance", pa.float64()),
        ("selected_los", pa.bool_()),
        ("selected_head", pa.int64()),
        ("selected_hp_pct", pa.float64()),
        ("target_hp", pa.float64()),
        ("target_max_hp", pa.float64()),
        ("hp", pa.float64()),
        ("max_hp", pa.float64()),
        ("mp", pa.float64()),
        ("max_mp", pa.float64()),
        ("cp", pa.float64()),
        ("max_cp", pa.float64()),
        ("exp", pa.int64()),
        ("sp", pa.int64()),
        ("lvl", pa.int64()),
        ("char_stats", char_stats),
        ("equipment", pa.struct([(slot, pa.string()) for slot in SLOTS])),
        ("equipment_ids", pa.struct([(slot, pa.int64()) for slot in SLOTS])),
        ("pos_x", pa.int64()),
        ("pos_y", pa.int64()),
        ("pos_z", pa.int64()),
        ("head", pa.int64()),
        ("is_sitting", pa.bool_()),
        ("is_casting", pa.bool_()),
        ("is_attacking", pa.bool_()),
        ("is_dead", pa.bool_()),
        ("weight_pct", pa.float64()),
        ("move_x", pa.int64()),
        ("move_y", pa.int64()),
        ("move_z", pa.int64()),
        ("teleport_from_x", pa.int64()),
        ("teleport_from_y", pa.int64()),
        ("teleport_from_z", pa.int64()),
        ("nearby_count", pa.int64()),
        ("nearby_objects", nearby_objects),
        ("ground_items", pa.int64()),
        ("in_combat", pa.bool_()),
        ("cooldown_raw", pa.string()),
        ("cooldowns", cooldown_list),
        ("active_skills", skill_list),
        ("passive_skills", skill_list),
        ("known_skills", skill_list),
        ("pet", pa.struct([
            ("object_id", pa.int64()), ("npc_id", pa.int64()), ("name", pa.string()), ("level", pa.int64()),
            ("hp", pa.float64()), ("max_hp", pa.float64()), ("mp", pa.float64()), ("max_mp", pa.float64()),
            ("x", pa.int64()), ("y", pa.int64()), ("z", pa.int64()), ("distance", pa.float64()),
            ("target_id", pa.int64()), ("atk", pa.bool_()), ("cast", pa.bool_()), ("dead", pa.bool_()),
            ("follow", pa.bool_()),
        ])),
        ("pet_command", pa.string()),
        ("by_pet", pa.bool_()),
        ("drops", pa.list_(pa.struct([("name", pa.string()), ("id", pa.int64()), ("count", pa.int64()),
                                      ("auto", pa.bool_())]))),
    ]

    if [name for name, _ in fields] != list(ROW_FIELDS):
        raise ValueError("Schema fields and ROW_FIELDS are out of sync")

    return pa.schema(fields)


def _collect_skills(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """Aggregate every skill observation per (character_id, skill_id)."""
    stats: dict[tuple[int, int], dict] = {}

    def record(character_id: int, skill_id: int, level: int | None, source: str, timestamp: str) -> None:
        entry = stats.get((character_id, skill_id))
        if entry is None:
            entry = {
                "level": None,
                "sources": set(),
                "observations": 0,
                "first_seen": timestamp,
                "last_seen": timestamp,
            }
            stats[(character_id, skill_id)] = entry
        entry["observations"] += 1
        entry["sources"].add(source)
        if level is not None and (entry["level"] is None or level > entry["level"]):
            entry["level"] = level
        if timestamp < entry["first_seen"]:
            entry["first_seen"] = timestamp
        if timestamp > entry["last_seen"]:
            entry["last_seen"] = timestamp

    for row in rows:
        character_id = row["character_id"]
        if character_id is None:
            continue
        timestamp = row["timestamp"]

        for skill in row["active_skills"] or []:
            record(character_id, skill["id"], skill["level"], SKILL_SOURCE_ACTIVE, timestamp)
        for skill in row["passive_skills"] or []:
            record(character_id, skill["id"], skill["level"], SKILL_SOURCE_PASSIVE, timestamp)
        for cooldown in row["cooldowns"] or []:
            record(character_id, cooldown["skill_id"], None, SKILL_SOURCE_COOLDOWN, timestamp)

        # ``skill_id`` on damage rows is the skill the character attacked with, while
        # ``took_damage`` reports the *opponent's* skill and is not attributed here.
        if row["skill_id"] is not None and row["action_type"] in ("used_skill", "learned_skill", "inflicted_damage"):
            source = SKILL_SOURCE_LEARNED if row["action_type"] == "learned_skill" else SKILL_SOURCE_USED
            record(character_id, row["skill_id"], row["skill_level"], source, timestamp)

        if row["effect_id"] is not None:
            record(character_id, row["effect_id"], None, SKILL_SOURCE_EFFECT, timestamp)

    return stats


def _build_characters_table(rows: list[dict], resolver: EntityResolver) -> pa.Table:
    """Build the character dimension table (id, name, lifetime aggregates)."""
    aggregates: dict[int, dict] = {}
    for row in rows:
        character_id = row["character_id"]
        if character_id is None:
            continue

        entry = aggregates.get(character_id)
        if entry is None:
            entry = {
                "first_seen": row["timestamp"],
                "last_seen": row["timestamp"],
                "first_file": row["source_file"],
                "last_file": row["source_file"],
                "min_level": row["lvl"],
                "max_level": row["lvl"],
                "frames": 0,
                "files": set(),
            }
            aggregates[character_id] = entry

        entry["frames"] += 1
        entry["files"].add(row["source_file"])
        if row["timestamp"] < entry["first_seen"]:
            entry["first_seen"] = row["timestamp"]
            entry["first_file"] = row["source_file"]
        if row["timestamp"] > entry["last_seen"]:
            entry["last_seen"] = row["timestamp"]
            entry["last_file"] = row["source_file"]
        if row["lvl"] is not None:
            entry["min_level"] = row["lvl"] if entry["min_level"] is None else min(entry["min_level"], row["lvl"])
            entry["max_level"] = row["lvl"] if entry["max_level"] is None else max(entry["max_level"], row["lvl"])

    names_by_id = {character_id: name for name, character_id in resolver.character_ids.items()}
    records = [
        {
            "character_id": character_id,
            "name": names_by_id.get(character_id),
            "first_seen": entry["first_seen"],
            "last_seen": entry["last_seen"],
            "first_file": entry["first_file"],
            "last_file": entry["last_file"],
            "min_level": entry["min_level"],
            "max_level": entry["max_level"],
            "frames": entry["frames"],
            "source_files": sorted(entry["files"]),
        }
        for character_id, entry in sorted(aggregates.items())
    ]

    schema = pa.schema([
        ("character_id", pa.int64()),
        ("name", pa.string()),
        ("first_seen", pa.string()),
        ("last_seen", pa.string()),
        ("first_file", pa.string()),
        ("last_file", pa.string()),
        ("min_level", pa.int64()),
        ("max_level", pa.int64()),
        ("frames", pa.int64()),
        ("source_files", pa.list_(pa.string())),
    ])
    return pa.Table.from_pylist(records, schema=schema)


def _build_skills_table(skill_stats: dict[tuple[int, int], dict], resolver: EntityResolver) -> pa.Table:
    """Build the per-character skill table, keyed by ``character_id`` + ``skill_id``."""
    records = [
        {
            "character_id": character_id,
            "skill_id": skill_id,
            "skill_level": entry["level"],
            "skill_name": resolver.skill_name(skill_id),
            "sources": sorted(entry["sources"]),
            "observations": entry["observations"],
            "first_seen": entry["first_seen"],
            "last_seen": entry["last_seen"],
        }
        for (character_id, skill_id), entry in sorted(skill_stats.items())
    ]

    schema = pa.schema([
        ("character_id", pa.int64()),
        ("skill_id", pa.int64()),
        ("skill_level", pa.int64()),
        ("skill_name", pa.string()),
        ("sources", pa.list_(pa.string())),
        ("observations", pa.int64()),
        ("first_seen", pa.string()),
        ("last_seen", pa.string()),
    ])
    return pa.Table.from_pylist(records, schema=schema)


def _learn_entities(path: Path, resolver: EntityResolver) -> None:
    """
    First pass: learn the character names appearing in the log and the authoritative
    item ids carried by ``Pickup`` messages.

    Characters must be known before parsing so that references to *other* players
    (``attacked: HumanFighter01``, ``Killer: DwarfFighter00``) are linked as
    ``entity_kind = "character"`` instead of being reported as unresolved.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for match in PICKUP_REF.finditer(line):
                resolver.observe_item(match.group("name"), int(match.group("item_id")))

            start = LINE_START.match(line)
            if start is not None:
                character, _ = _split_character(start.group("rest").strip())
                resolver.observe_character(character)
                continue
            new = v2.LINE.match(line.rstrip("\n"))
            if new is not None and new.group("event") in v2.EVENTS:
                resolver.observe_character(new.group("character"))
                if new.group("event") == "PICKUP":
                    name, item_id = v2.named_id(v2.split_tail(new.group("rest") or "")[0])
                    if item_id is not None:
                        resolver.observe_item(name, item_id)


def _fill_known_skills(rows: list[dict]) -> int:
    """
    Give every frame the active skills its character knows at that moment (``known_skills``).

    Only ``skills_updated`` frames carry the skill list, and they come in bursts that fill the list
    step by step (``[Active:]``, ``[Active:1322-1]``, ...). Every frame takes the latest list of its
    character (rows are in chronological order: log files sorted by name, lines in order); frames
    before the first list take the state the first burst ended with, i.e. the list in force when the
    character's first non-``skills_updated`` frame appeared.

    :return: number of frames that got a list
    """
    current: dict = {}
    initial: dict = {}
    settled: set = set()
    for row in rows:
        character_id = row["character_id"]
        if row["action_type"] == "skills_updated":
            if row["active_skills"]:
                current[character_id] = row["active_skills"]
                if character_id not in settled:
                    initial[character_id] = row["active_skills"]
        else:
            settled.add(character_id)
        row["known_skills"] = current.get(character_id)

    filled = 0
    for row in rows:
        if row["known_skills"] is None:
            row["known_skills"] = initial.get(row["character_id"])
        filled += row["known_skills"] is not None
    return filled


def process_logs(
    input_path: str | Path,
    output_parquet: str | Path,
    npc_parquet: str | Path = DEFAULT_NPC_PARQUET,
    item_parquet: str | Path = DEFAULT_ITEM_PARQUET,
    characters_out: str | Path = DEFAULT_CHARACTERS_OUTPUT,
    skills_out: str | Path = DEFAULT_SKILLS_OUTPUT,
    skills_xml: str | Path | None = DEFAULT_SKILLS_XML,
) -> None:
    """Parse a log file (or directory of logs) into the three linked tables."""
    source = Path(input_path)
    if source.is_dir():
        log_files = sorted(source.rglob("*.log"))
    elif source.is_file():
        log_files = [source]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not log_files:
        raise ValueError(f"No *.log files found in {input_path}")

    for label, table_path in (("NPC", npc_parquet), ("item", item_parquet)):
        if not Path(table_path).exists():
            raise ValueError(f"{label} table not found: {table_path} (run the *_xml2parquet.py converters first)")

    print(f"Found {len(log_files)} log file(s)")

    resolver = EntityResolver(npc_parquet, item_parquet)
    print(f"Loaded {len(resolver.npcs_by_id)} NPCs and {len(resolver.items_by_id)} items for entity linking")

    for log_path in log_files:
        _learn_entities(log_path, resolver)
    if resolver.observed_items:
        print(f"Learned {len(resolver.observed_items)} item ids from Pickup messages")
    if resolver.characters:
        print(f"Learned {len(resolver.characters)} character names from the logs")
    resolver.assign_character_ids()

    if skills_xml is not None:
        added = resolver.load_skill_names(skills_xml)
        if added:
            print(f"Loaded {added} extra skill names from {skills_xml}")

    rows: list[dict] = []
    # Format 2 writes STATS / EQUIP / SKILLS only on change: keep them per character across the files.
    states: dict[str, v2.CharacterState] = {}
    skipped = 0
    for log_path in log_files:
        before = len(rows)
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                row = parse_log_line(line, resolver, log_path.name, line_no, states)
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
        print(f"  {log_path.name}: {len(rows) - before} action frames")

    print(f"Successfully parsed {len(rows)} action frames ({skipped} lines skipped).")
    known = _fill_known_skills(rows)
    print(f"known_skills filled for {known} of {len(rows)} frames")

    if not rows:
        print("No action frames found to export.")
        return

    counts = Counter(row["action_type"] for row in rows)
    print(f"Action types: {dict(counts.most_common())}")

    if resolver.ambiguous:
        print(f"[Info] {len(resolver.ambiguous)} names matched several datapack rows, resolved by heuristic:")
        for name, hits in resolver.ambiguous.most_common(10):
            print(f"    {name} ({hits}x)")
    if resolver.unresolved:
        print(f"[Warning] {len(resolver.unresolved)} unresolved entity references:")
        for name, hits in resolver.unresolved.most_common(10):
            print(f"    {name} ({hits}x)")

    table = pa.Table.from_pylist(rows, schema=_build_schema())
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"Actions saved to {output_parquet} ({table.num_rows} rows x {table.num_columns} columns)")

    characters = _build_characters_table(rows, resolver)
    pq.write_table(characters, characters_out, compression="snappy")
    print(f"Characters saved to {characters_out} ({characters.num_rows} rows x {characters.num_columns} columns)")

    skill_stats = _collect_skills(rows)
    skills = _build_skills_table(skill_stats, resolver)
    pq.write_table(skills, skills_out, compression="snappy")
    print(f"Character skills saved to {skills_out} ({skills.num_rows} rows x {skills.num_columns} columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L2J player-action log to Parquet converter with NPC / item / skill links."
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to a playeraction .log file or a directory of logs")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT), help=f"Action table (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--characters-out",
        type=str,
        default=str(DEFAULT_CHARACTERS_OUTPUT),
        help=f"Character dimension table (default: {DEFAULT_CHARACTERS_OUTPUT})",
    )
    parser.add_argument(
        "--skills-out",
        type=str,
        default=str(DEFAULT_SKILLS_OUTPUT),
        help=f"Per-character skills table (default: {DEFAULT_SKILLS_OUTPUT})",
    )
    parser.add_argument(
        "--skills-xml",
        type=str,
        default=str(DEFAULT_SKILLS_XML),
        help="Skills datapack directory used for skill names (empty string to disable)",
    )
    parser.add_argument("--npc-parquet", type=str, default=str(DEFAULT_NPC_PARQUET), help="NPC table used for entity links")
    parser.add_argument("--item-parquet", type=str, default=str(DEFAULT_ITEM_PARQUET), help="Item table used for entity links")
    args = parser.parse_args()

    process_logs(
        args.input,
        args.output,
        args.npc_parquet,
        args.item_parquet,
        args.characters_out,
        args.skills_out,
        args.skills_xml or None,
    )
