"""
Player action log, format 2: the grammar shared by the parser and the converter.

The game server (``PlayerActionLogger``) writes one compact line per action::

    2026.10.14 10:01:59.411 dwarf DMG 85 -> Ol Mahum Guard#268451514(20058 L22) hp 526/611 skill - | me L18 hp 711/711 mp 231/231 cp 368/368 @-49823,147087,-2792 h57909 wt28.5 atk combat | sel d14.2 los h25615 | env items 0; 20058#268451514 L22 d14 hp86 by:Bob; 20437#268451777 L21 d2226 hp100 | cd 3:4500

``<date> <time> <character> <EVENT> <payload> [| me ...] [| sel ...] [| pet ...] [| env ...] [| cd ...]``

==========  ==================================================================  =====================
event       payload                                                             old ``action_type``
==========  ==================================================================  =====================
TARGET      ``<ref>``                                                           selected_target
MOVE        ``x,y,z``                                                           moved
ATTACK      ``<ref>``                                                           attacked
SKILL       ``<skill>[id] -> <ref>``                                            used_skill
DMG         ``<damage|miss> -> <ref> skill <skill>[id]|-``                      inflicted_damage
            ``... by pet``: the character's summon dealt it                      pet_inflicted_damage
HIT         ``<damage|miss> <- <ref> skill <skill>[id]|-``                      took_damage
KILL        ``<ref> [by pet] drop <item>[count] auto|ground; ...|-``: one line  killed
            per kill (``by pet``: the summon finished it); older logs wrote one
            line per dropped item and no line for a kill without drops
PICKUP      ``<item>[itemId]``                                                  pickup
ACTION      ``<action> <details>``                                              performed_action
ITEM        ``<item>[itemId] left <count>``                                     used_item
EFFECT      ``buff|debuff <skill>[id] <seconds>s``                              effect_applied
EFFECTS     ``buffs <skill>[id]:<s>s,...|- debuffs ...|-``                      active_effects
AUTOSHOT    ``on|off <item>[itemId]``                                           autouse_enabled/…
SHOT        ``<item> left <count>``                                             shot_used
BUY / SELL  ``<item>[itemId] x<count> price <adena> npc <name>[npcId]``         buy / sell
LEARN       ``<skill>[id] L<level> cost <sp>sp [<adena>a]``                     learned_skill
TELEPORT    ``x,y,z -> x,y,z via <method>``                                     teleported
DIED        ``@x,y,z killer <name>[id]``                                        died
REVIVED     ``@x,y,z via <method>``                                             resurrected
SKILLS      ``active <id>-<lvl>,...|- passive ...|-`` (state, on change)        skills_updated
STATS       ``L<lvl> exp <n> sp <n> patk .. matk .. pdef .. mdef .. acc .. cls <class id> hpot <healing potions>``  (state only)
            ``crit .. eva .. aspd .. cspd .. str .. int .. dex .. con .. wit ..``
EQUIP       ``weapon <item>[id]|- shield .. head .. chest .. gloves .. legs``    (state only)
            ``.. feet .. neck .. rear .. lear .. rfinger .. lfinger ..``
SUMMON      ``<summon ref>``: a servitor / pet appeared                        summoned
UNSUMMON    ``<summon ref>``: dismissed (or its time ran out)                    unsummoned
PETDIED     ``<summon ref> <- <killer ref>``                                    pet_died
PETCMD      ``<command> -> <ref of the owner's target>``; command ``follow``      pet_command
            ``attack stop unsummon move`` or ``skill<actionId>``
PETSKILL    ``<skill>[id] -> <ref>``: the summon casts                          pet_used_skill
PETHIT      ``<damage> <- <ref> skill <skill>[id]|-``: the summon was hit        pet_took_damage
NOTE        ``<free text>``: an old line the converter could not map             (kept as is)
==========  ==================================================================  =====================

A reference ``<ref>`` is ``Name#objectId(npcId Llvl) hp cur/max`` (monster or NPC),
``Name#objectId(pc Llvl) hp cur/max`` (player), ``Name#objectId(summon npcId Llvl) hp cur/max``
(servitor or pet), ``Name#objectId(item itemId)`` (item on the ground) or ``-``. Old logs had no object ids: the converter writes ``#?``.

Tail blocks: ``me`` - the character (level, HP / MP / CP, position, heading, load %, flags
``sit cast atk dead combat``); ``sel`` - the selected target relative to the character
(distance, line of sight, its heading); ``pet`` - the character's summon while it has one
(``<summon ref> mp cur/max @x,y,z d<distance> tgt <objectId>|-`` and the flags
``atk cast dead follow``); ``env`` - ground items within 1000 and the 10 nearest
monsters within 3000 (``a<deg>`` the direction to it, 0 = north / 90 = east, newer logs only;
``me`` - its target is the character; ``by:`` lists the other players on a monster's aggro list); ``cd`` -
skill reuse still running in milliseconds.

``STATS`` / ``EQUIP`` / ``SKILLS`` lines are written only when they change; a reader keeps the
last ones per character (:class:`CharacterState`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE = re.compile(r"^(?P<date>\d{4}\.\d{2}\.\d{2}) (?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) (?P<character>\S+) "
                  r"(?P<event>[A-Z]+)(?: (?P<rest>.*))?$")

EVENTS = frozenset({
    "TARGET", "MOVE", "ATTACK", "SKILL", "DMG", "HIT", "KILL", "PICKUP", "ACTION", "ITEM", "EFFECT", "EFFECTS",
    "AUTOSHOT", "SHOT", "BUY", "SELL", "LEARN", "TELEPORT", "DIED", "REVIVED", "SKILLS", "STATS", "EQUIP", "NOTE",
    "SUMMON", "UNSUMMON", "PETDIED", "PETCMD", "PETSKILL", "PETHIT",
})

REF = re.compile(r"^(?P<name>.+?)#(?P<oid>\d+|\?)\((?P<inside>[^)]*)\)(?: hp (?P<hp>-?\d+)/(?P<max_hp>\d+))?$")
NAMED_ID = re.compile(r"^(?P<name>.+?)\[(?P<id>\d+)\]$")
STAT_KEYS = ("patk", "matk", "pdef", "mdef", "acc", "crit", "eva", "aspd", "cspd", "str", "int", "dex", "con", "wit",
             "cls", "hpot")
SLOT_KEYS = ("weapon", "shield", "head", "chest", "gloves", "legs", "feet", "neck", "rear", "lear", "rfinger", "lfinger")
FLAGS = ("sit", "cast", "atk", "dead", "combat")
PET_FLAGS = ("atk", "cast", "dead", "follow")
PET = re.compile(r"^(?P<ref>.+?\) hp -?\d+/\d+) mp (?P<mp>-?\d+)/(?P<max_mp>\d+) @(?P<x>-?\d+),(?P<y>-?\d+),(?P<z>-?\d+) "
                 r"d(?P<distance>\d+) tgt (?P<target>\d+|-)(?P<flags>(?: \w+)*)$")


def is_v2(line: str) -> bool:
    """A format-2 line (as opposed to the tab-separated format 1)."""
    m = LINE.match(line)
    return m is not None and m.group("event") in EVENTS


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------


def parse_ref(text: str) -> dict | None:
    """``Name#oid(npcId L22) hp 5/10`` -> ``{name, object_id, kind, npc_id|item_id, level, hp, max_hp}``."""
    text = text.strip()
    if text in ("", "-"):
        return None
    m = REF.match(text)
    if m is None:
        return {"name": text, "object_id": None, "kind": "name"}
    ref = {"name": m.group("name"), "object_id": None if m.group("oid") == "?" else int(m.group("oid"))}
    inside = m.group("inside").split()
    if inside and inside[0] == "pc":
        ref["kind"] = "player"
        ref["level"] = int(inside[1][1:]) if len(inside) > 1 else None
    elif inside and inside[0] == "summon":
        ref["kind"] = "summon"
        ref["npc_id"] = int(inside[1]) if len(inside) > 1 else None
        ref["level"] = int(inside[2][1:]) if len(inside) > 2 else None
    elif inside and inside[0] == "item":
        ref["kind"] = "item"
        ref["item_id"] = int(inside[1]) if len(inside) > 1 else None
    elif inside and inside[0].isdigit():
        ref["kind"] = "npc"
        ref["npc_id"] = int(inside[0])
        ref["level"] = int(inside[1][1:]) if len(inside) > 1 else None
    else:
        ref["kind"] = "creature"
        ref["level"] = int(inside[0][1:]) if inside and inside[0].startswith("L") else None
    if m.group("hp") is not None:
        ref["hp"], ref["max_hp"] = int(m.group("hp")), int(m.group("max_hp"))
    return ref


def format_ref(name: str | None, object_id: int | None = None, *, npc_id: int | None = None, level: int | None = None,
               player: bool = False, item_id: int | None = None, hp: float | None = None,
               max_hp: float | None = None) -> str:
    if not name:
        return "-"
    oid = "?" if object_id is None else str(object_id)
    if item_id is not None:
        inside = f"item {item_id}"
    elif player:
        inside = f"pc L{level if level is not None else 0}"
    elif npc_id is not None:
        inside = f"{npc_id} L{level if level is not None else 0}"
    else:
        inside = f"L{level if level is not None else 0}"
    text = f"{name}#{oid}({inside})"
    if hp is not None and max_hp is not None and item_id is None:
        text += f" hp {round(hp)}/{round(max_hp)}"
    return text


def named_id(text: str) -> tuple[str | None, int | None]:
    """``Wooden Arrow[17]`` -> ``("Wooden Arrow", 17)``; ``-`` (nothing) -> ``(None, None)``."""
    m = NAMED_ID.match(text.strip())
    if m:
        return (None if m.group("name") == "-" else m.group("name")), int(m.group("id"))
    return (None if text.strip() in ("", "-") else text.strip()), None


# ----------------------------------------------------------------------
# Tail blocks
# ----------------------------------------------------------------------


def split_tail(rest: str) -> tuple[str, dict[str, str]]:
    """Split ``payload | me .. | sel .. | pet .. | env .. | cd ..`` into the payload and the named blocks."""
    parts = rest.split(" | ")
    payload, blocks = parts[0], {}
    for part in parts[1:]:
        key, _, value = part.partition(" ")
        if key in ("me", "sel", "pet", "env", "cd"):
            blocks[key] = value
        else:  # a ' | ' inside the payload (e.g. an item name): glue it back
            if blocks:
                last = list(blocks)[-1]
                blocks[last] += " | " + part
            else:
                payload += " | " + part
    return payload, blocks


def parse_me(text: str) -> dict:
    tokens = text.split()
    me: dict = {flag: False for flag in FLAGS}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("hp", "mp", "cp") and i + 1 < len(tokens):
            cur, _, mx = tokens[i + 1].partition("/")
            me[tok], me["max_" + tok] = float(cur), float(mx)
            i += 2
            continue
        if tok.startswith("L") and tok[1:].isdigit():
            me["level"] = int(tok[1:])
        elif tok.startswith("@"):
            x, y, z = tok[1:].split(",")
            me["x"], me["y"], me["z"] = int(x), int(y), int(z)
        elif tok.startswith("wt"):
            me["weight_pct"] = float(tok[2:])
        elif tok.startswith("h") and tok[1:].isdigit():
            me["heading"] = int(tok[1:])
        elif tok in FLAGS:
            me[tok] = True
        i += 1
    return me


def format_me(level, hp, max_hp, mp, max_mp, cp, max_cp, x, y, z, heading, weight_pct, flags: dict) -> str:
    text = (f"me L{level} hp {round(hp)}/{round(max_hp)} mp {round(mp)}/{round(max_mp)} cp {round(cp)}/{round(max_cp)} "
            f"@{x},{y},{z} h{heading} wt{weight_pct:.1f}")
    return text + "".join(f" {flag}" for flag in FLAGS if flags.get(flag))


def parse_sel(text: str) -> dict:
    sel: dict = {}
    for tok in text.split():
        if tok.startswith("d"):
            sel["distance"] = float(tok[1:])
        elif tok in ("los", "nolos"):
            sel["los"] = tok == "los"
        elif tok.startswith("h") and tok[1:].isdigit():
            sel["heading"] = int(tok[1:])
    return sel


def parse_pet(text: str) -> dict | None:
    """``Kat the Cat#oid(summon 14111 L39) hp 900/1000 mp 50/60 @x,y,z d120 tgt 2684.. atk follow``."""
    m = PET.match(text.strip())
    if m is None:
        return None
    ref = parse_ref(m.group("ref")) or {}
    flags = set(m.group("flags").split())
    return {
        "object_id": ref.get("object_id"), "npc_id": ref.get("npc_id"), "name": ref.get("name"), "level": ref.get("level"),
        "hp": ref.get("hp"), "max_hp": ref.get("max_hp"), "mp": int(m.group("mp")), "max_mp": int(m.group("max_mp")),
        "x": int(m.group("x")), "y": int(m.group("y")), "z": int(m.group("z")), "distance": int(m.group("distance")),
        "target_id": None if m.group("target") == "-" else int(m.group("target")),
        **{flag: flag in flags for flag in PET_FLAGS},
    }


def format_sel(distance: float, los: bool, heading: int | None) -> str:
    return f"sel d{distance:.1f} {'los' if los else 'nolos'}" + ("" if heading is None else f" h{heading}")


def parse_env(text: str) -> dict:
    env: dict = {"items": None, "monsters": []}
    for part in text.split("; "):
        tokens = part.split()
        if not tokens:
            continue
        if tokens[0] == "items":
            env["items"] = int(tokens[1])
            continue
        npc, _, oid = tokens[0].partition("#")
        mob = {"npc_id": int(npc), "object_id": None if oid in ("", "?") else int(oid), "level": None,
               "distance": None, "hp_pct": None, "attackers": [], "bearing": None, "targets_me": False}
        for tok in tokens[1:]:
            if tok.startswith("L"):
                mob["level"] = int(tok[1:])
            elif tok.startswith("d"):
                mob["distance"] = float(tok[1:])
            elif tok.startswith("hp"):
                mob["hp_pct"] = float(tok[2:])
            elif tok.startswith("by:"):
                mob["attackers"] = tok[3:].split(",")
            elif tok == "me":
                mob["targets_me"] = True
            elif tok.startswith("a") and tok[1:].isdigit():
                mob["bearing"] = float(tok[1:])
        env["monsters"].append(mob)
    return env


def format_env(items: int | None, monsters: list[dict]) -> str:
    parts = [f"items {items if items is not None else 0}"]
    for mob in monsters:
        oid = "?" if mob.get("object_id") is None else mob["object_id"]
        text = f"{mob['npc_id']}#{oid} L{mob.get('level') or 0} d{round(mob.get('distance') or 0)}"
        if mob.get("hp_pct") is not None:
            text += f" hp{int(round(mob['hp_pct']))}"
        if mob.get("bearing") is not None:
            text += f" a{int(round(mob['bearing'])) % 360}"
        if mob.get("targets_me"):
            text += " me"
        if mob.get("attackers"):
            text += " by:" + ",".join(mob["attackers"])
        parts.append(text)
    return "env " + "; ".join(parts)


def parse_cd(text: str) -> list[tuple[int, int]]:
    out = []
    for entry in text.split(","):
        sid, _, ms = entry.partition(":")
        if sid.strip().isdigit() and ms.strip().lstrip("-").isdigit():
            out.append((int(sid), int(ms)))
    return out


def format_cd(cooldowns: list[tuple[int, int]]) -> str | None:
    return ("cd " + ",".join(f"{sid}:{ms}" for sid, ms in cooldowns)) if cooldowns else None


# ----------------------------------------------------------------------
# State lines
# ----------------------------------------------------------------------


def parse_stats(text: str) -> dict:
    tokens = text.split()
    stats: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("L") and tok[1:].isdigit():
            stats["level"] = int(tok[1:])
            i += 1
        elif i + 1 < len(tokens):
            value = tokens[i + 1]
            stats[tok] = float(value) if "." in value else int(value)
            i += 2
        else:
            i += 1
    return stats


def format_stats(level: int, exp: int, sp: int, values: dict) -> str:
    rest = " ".join(f"{key} {values[key]}" for key in STAT_KEYS if values.get(key) is not None)
    return f"L{level} exp {exp} sp {sp} {rest}"


def parse_equip(text: str) -> dict[str, tuple[str | None, int | None]]:
    """``weapon Tarbar[156] shield - ...`` -> ``{slot: (name, item_id)}`` (``(None, None)`` for an empty slot)."""
    pattern = re.compile(r"(?:^| )(?P<slot>" + "|".join(SLOT_KEYS) + r") (?P<value>.*?)(?= (?:" + "|".join(SLOT_KEYS)
                         + r") |$)")
    out = {}
    for m in pattern.finditer(text):
        value = m.group("value").strip()
        out[m.group("slot")] = (None, None) if value == "-" else named_id(value)
    return out


def format_equip(slots: dict[str, tuple[str | None, int | None]]) -> str:
    parts = []
    for slot in SLOT_KEYS:
        name, item_id = slots.get(slot, (None, None))
        parts.append(f"{slot} " + ("-" if not name else (f"{name}[{item_id}]" if item_id is not None else name)))
    return " ".join(parts)


def parse_skills(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    def entries(value: str) -> list[tuple[int, int]]:
        if value in ("", "-"):
            return []
        out = []
        for entry in value.split(","):
            sid, _, lvl = entry.partition("-")
            if sid.isdigit() and lvl.isdigit():
                out.append((int(sid), int(lvl)))
        return out

    m = re.match(r"^active (?P<active>\S+) passive (?P<passive>\S+)$", text.strip())
    return (entries(m.group("active")), entries(m.group("passive"))) if m else ([], [])


def format_skills(active: list[tuple[int, int]], passive: list[tuple[int, int]]) -> str:
    def entries(values):
        return ",".join(f"{sid}-{lvl}" for sid, lvl in values) or "-"

    return f"active {entries(active)} passive {entries(passive)}"


@dataclass
class CharacterState:
    """The last ``STATS`` / ``EQUIP`` / ``SKILLS`` of one character while a log is read or written."""

    stats: dict = field(default_factory=dict)
    equip: dict = field(default_factory=dict)
    active: list = field(default_factory=list)
    passive: list = field(default_factory=list)
    stats_text: str | None = None
    equip_text: str | None = None
    skills_text: str | None = None
