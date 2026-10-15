"""
Convert player action logs from format 1 (tab-separated, every block on every line) to the
compact format 2 (``playeraction_v2.py``), the format the game server writes now.

Every format-1 line is parsed with the parquet converter's own parser and written again:
``STATS`` / ``EQUIP`` / ``SKILLS`` only when they change, the environment cut to the 10 nearest
monsters within 3000, monster names and traits left out (the NPC id identifies them). Format 1
has no world object ids (written ``#?``) and no ``by:`` attackers. Lines that are already
format 2 are copied, lines no event fits are kept as ``NOTE <text>``.

Usage (from ``l2_agent/``)::

    uv run l2_NN/tools/convert_playeraction_log.py -i ../l2_playeractions -o ../l2_playeractions_v2
    uv run l2_NN/tools/convert_playeraction_log.py -i old.log -o new.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import playeraction_v2 as v2  # noqa: E402
from playeraction_log2parquet import (  # noqa: E402
    LINE_START,
    EntityResolver,
    _learn_entities,
    _split_character,
    parse_log_line,
)

# The datapack tables live in the repository root (next to l2_agent/).
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_RANGE = 3000
ENV_MAX_MONSTERS = 10
V1_SLOTS = {"Weapon": "weapon", "Shield": "shield", "Head": "head", "Chest": "chest", "Gloves": "gloves", "Legs": "legs",
            "Feet": "feet", "Neck": "neck", "REar": "rear", "LEar": "lear", "RFinger": "rfinger", "LFinger": "lfinger"}
STATS_KEYS = {"p_atk": "patk", "m_atk": "matk", "p_def": "pdef", "m_def": "mdef", "accur": "acc", "crit_rate": "crit",
              "evasion": "eva", "atk_speed": "aspd", "cast_speed": "cspd", "str": "str", "int": "int", "dex": "dex",
              "con": "con", "wit": "wit"}


def _ref(row: dict) -> str:
    kind = row.get("entity_kind")
    return v2.format_ref(
        row.get("entity_name") or row.get("entity_item_name"), row.get("entity_object_id"),
        npc_id=row.get("entity_npc_id") if kind not in ("item", "character") else None,
        level=row.get("entity_npc_level"), player=kind == "character",
        item_id=row.get("entity_item_id") if kind == "item" else None,
        hp=row.get("target_hp"), max_hp=row.get("target_max_hp"))


def _named(name: str | None, ident: int | None) -> str:
    return "-" if not name else (f"{name}[{ident}]" if ident is not None else name)


def _tail(row: dict) -> str:
    parts = []
    if row.get("hp") is not None and row.get("pos_x") is not None:
        flags = {"sit": row.get("is_sitting"), "cast": row.get("is_casting"), "atk": row.get("is_attacking"),
                 "dead": row.get("is_dead"), "combat": row.get("in_combat")}
        parts.append(v2.format_me(row.get("lvl") or 0, row["hp"], row.get("max_hp") or 0, row.get("mp") or 0,
                                  row.get("max_mp") or 0, row.get("cp") or 0, row.get("max_cp") or 0, row["pos_x"],
                                  row["pos_y"], row["pos_z"], row.get("head") or 0, row.get("weight_pct") or 0.0, flags))
    if row.get("selected_distance") is not None:
        parts.append(v2.format_sel(row["selected_distance"], bool(row.get("selected_los")), row.get("selected_head")))
    if row.get("ground_items") is not None or row.get("nearby_objects"):
        mobs = sorted((o for o in row.get("nearby_objects") or [] if (o.get("distance") or 0) <= ENV_RANGE),
                      key=lambda o: o.get("distance") or 0)[:ENV_MAX_MONSTERS]
        parts.append(v2.format_env(row.get("ground_items"), [
            {"npc_id": o["id"], "object_id": None, "level": o.get("lvl"), "distance": o.get("distance")} for o in mobs]))
    cd = v2.format_cd([(c["skill_id"], c["remaining"]) for c in row.get("cooldowns") or []])
    if cd:
        parts.append(cd)
    return "".join(" | " + p for p in parts)


def _event(row: dict, message: str) -> str:
    """The format-2 ``EVENT payload`` of a parsed format-1 row."""
    action = row["action_type"]
    damage = "miss" if row.get("is_miss") else str(row.get("damage") or 0)
    body = message.split(":", 1)[1].strip() if ":" in message else message
    if action == "selected_target":
        return "TARGET " + _ref(row)
    if action == "attacked":
        return "ATTACK " + _ref(row)
    if action == "moved":
        return f"MOVE {row['move_x']},{row['move_y']},{row['move_z']}"
    if action == "used_skill":
        return f"SKILL {_named(row.get('skill_name'), row.get('skill_id'))} -> {_ref(row)}"
    if action == "inflicted_damage":
        return f"DMG {damage} -> {_ref(row)} skill {_named(row.get('skill_name'), row.get('skill_id'))}"
    if action == "took_damage":
        return f"HIT {damage} <- {_ref(row)} skill {_named(row.get('skill_name'), row.get('skill_id'))}"
    if action == "killed":
        how = "auto" if "auto-looted" in message else "ground"
        return (f"KILL {_ref(row)} drop {row.get('drop_item_name') or '-'}[{row.get('drop_item_count') or 0}] {how}")
    if action == "pickup":
        return "PICKUP " + _named(row.get("entity_item_name") or row.get("entity_name"), row.get("entity_item_id"))
    if action == "used_item":
        return f"ITEM {_named(row.get('entity_item_name') or row.get('entity_name'), row.get('entity_item_id'))} left {row.get('item_remaining') or 0}"
    if action == "effect_applied":
        return (f"EFFECT {(row.get('effect_kind') or 'buff').lower()} {_named(row.get('effect_name'), row.get('effect_id'))} "
                f"{row.get('effect_duration') or 0}s")
    if action in ("autouse_enabled", "autouse_disabled"):
        state = "on" if action == "autouse_enabled" else "off"
        return f"AUTOSHOT {state} {_named(row.get('entity_item_name') or row.get('entity_name'), row.get('entity_item_id'))}"
    if action == "buy":
        return (f"BUY {_named(row.get('entity_item_name') or row.get('entity_name'), row.get('entity_item_id'))} "
                f"x{row.get('entity_item_count') or 0} price {row.get('item_price') or 0} "
                f"npc {_named(row.get('vendor_npc_name') or '?', row.get('vendor_npc_id') or 0)}")
    if action == "learned_skill":
        return (f"LEARN {_named(row.get('skill_name'), row.get('skill_id'))} L{row.get('skill_level') or 0} "
                f"cost {row.get('skill_cost') or 0}sp")
    if action == "teleported":
        return (f"TELEPORT {row.get('teleport_from_x')},{row.get('teleport_from_y')},{row.get('teleport_from_z')} -> "
                f"{row.get('move_x')},{row.get('move_y')},{row.get('move_z')} via {row.get('action_method') or '?'}")
    if action == "died":
        return f"DIED @{row.get('move_x')},{row.get('move_y')},{row.get('move_z')} killer {row.get('entity_name') or '?'}[0]"
    if action == "resurrected":
        return f"REVIVED @{row.get('move_x')},{row.get('move_y')},{row.get('move_z')} via {row.get('action_method') or '?'}"
    if action == "performed_action":
        return "ACTION " + body
    return "NOTE " + message


def convert_line(line: str, resolver: EntityResolver, states: dict[str, v2.CharacterState]) -> list[str]:
    """The format-2 lines of one format-1 line (state lines first), ``[]`` for a non-action line."""
    if v2.is_v2(line):
        return [line]
    start = LINE_START.match(line)
    if start is None:
        return []
    character, message = _split_character(start.group("rest").strip())
    row = parse_log_line(line, resolver, "", 0)
    if row is None or not character:
        return []

    date, time = start.group("timestamp").replace(",", ".").split(" ")
    prefix = f"{date} {time} {character} "
    state = states.setdefault(character, v2.CharacterState())
    out = []

    if row.get("char_stats") and row.get("lvl") is not None:
        values = {STATS_KEYS[k]: v for k, v in row["char_stats"].items() if k in STATS_KEYS}
        text = v2.format_stats(row["lvl"], row.get("exp") or 0, row.get("sp") or 0, values)
        if text != state.stats_text:
            state.stats_text = text
            out.append(prefix + "STATS " + text)
    if row.get("equipment"):
        ids = row.get("equipment_ids") or {}
        slots = {V1_SLOTS[slot]: (name, ids.get(slot)) for slot, name in row["equipment"].items() if slot in V1_SLOTS}
        text = v2.format_equip(slots)
        if text != state.equip_text:
            state.equip_text = text
            out.append(prefix + "EQUIP " + text)
    if row["action_type"] == "skills_updated":
        text = v2.format_skills([(s["id"], s["level"]) for s in row.get("active_skills") or []],
                                [(s["id"], s["level"]) for s in row.get("passive_skills") or []])
        if text != state.skills_text:
            state.skills_text = text
            out.append(prefix + "SKILLS " + text)
        return out

    event = _event(row, message)
    with_tail = not event.startswith(("NOTE", "AUTOSHOT", "ITEM", "BUY", "LEARN", "TELEPORT", "DIED", "REVIVED"))
    out.append(prefix + event + (_tail(row) if with_tail else ""))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Convert player action logs to format 2.")
    p.add_argument("-i", "--input", required=True, help="A .log file or a directory of them")
    p.add_argument("-o", "--output", required=True, help="Output file (for a file) or directory (for a directory)")
    p.add_argument("--npc-parquet", default=str(REPO_ROOT / "l2jc6_npcs.parquet"))
    p.add_argument("--item-parquet", default=str(REPO_ROOT / "l2jc6_items.parquet"))
    args = p.parse_args()

    src = Path(args.input)
    files = sorted(src.glob("*.log")) if src.is_dir() else [src]
    out_dir = Path(args.output) if src.is_dir() else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    resolver = EntityResolver(args.npc_parquet, args.item_parquet)
    for path in files:
        _learn_entities(path, resolver)

    states: dict[str, v2.CharacterState] = {}
    for path in files:
        target = out_dir / path.name if out_dir else Path(args.output)
        before = path.stat().st_size
        lines = 0
        with open(path, encoding="utf-8", errors="replace") as fin, open(target, "w", encoding="utf-8") as fout:
            for line in fin:
                for new in convert_line(line.rstrip("\n"), resolver, states):
                    fout.write(new + "\n")
                    lines += 1
        after = target.stat().st_size
        print(f"{path.name}: {before / 2**20:.1f} MB -> {after / 2**20:.1f} MB ({before / max(after, 1):.1f}x), {lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
