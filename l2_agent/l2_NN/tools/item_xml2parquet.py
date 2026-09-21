"""
L2J Mobius Item XML to Parquet converter.

Flattens the Lineage2 Interlude item datapack (``data/stats/items/*.xml``)
into a single Parquet table. Every ``<item>`` element becomes one row:
identity attributes, the most relevant ``<set>`` attributes, combat
``<stats>``, granted ``<skills>`` and the ``<conditions>`` required to equip
it. A ``text_representation`` column is added for RAG / vector search
indexing, mirroring the NPC converter.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# --- <set name="..."> attributes that are flattened into their own column ---
# Human readable / enum-like values.
SET_STRING_KEYS = [
    "icon",
    "default_action",
    "bodypart",
    "material",
    "crystal_type",
    "weapon_type",
    "armor_type",
    "etcitem_type",
    "damage_range",
    "handler",
    "item_type",
    "use_condition",
    "equip_condition",
    "reduced_mp_consume",
    "reduced_soulshot",
]

# Plain integer values.
SET_INT_KEYS = [
    "weight",
    "price",
    "soulshots",
    "spiritshots",
    "crystal_count",
    "recipe_id",
    "duration",
    "reuse_delay",
    "mp_consume",
    "useSkillDisTime",
    "equip_reuse_delay",
    "shared_reuse_group",
    "displayId",
]

# "true" / "false" flags.
SET_BOOL_KEYS = [
    "immediate_effect",
    "ex_immediate_effect",
    "is_stackable",
    "is_tradable",
    "is_dropable",
    "is_sellable",
    "is_depositable",
    "is_questitem",
    "is_destroyable",
    "is_oly_restricted",
    "is_magic_weapon",
    "enchant_enabled",
    "element_enabled",
    "for_npc",
    "isAttackWeapon",
    "isForceEquip",
    "useWeaponSkillsOnly",
]

KNOWN_SET_KEYS = SET_STRING_KEYS + SET_INT_KEYS + SET_BOOL_KEYS

# --- <stat type="..."> combat attributes folded into the "stats" struct ---
STAT_KEYS = [
    "pAtk",
    "mAtk",
    "pDef",
    "mDef",
    "pAtkSpd",
    "critRate",
    "accCombat",
    "randomDamage",
    "pAtkRange",
    "maxMp",
    "sDef",
    "rShld",
    "rEvas",
]

# Collects tags/attributes the datapack uses but this converter does not model,
# so that datapack changes are reported instead of silently dropped.
_UNKNOWN_SETS: set[str] = set()
_UNKNOWN_STATS: set[str] = set()


def _clean(value: str | None) -> str | None:
    """Return ``value`` stripped, mapping missing/empty strings to ``None``."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _to_int(value: str | None) -> int | None:
    """Convert an XML attribute to int, returning ``None`` when not numeric."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_float(value: str | None) -> float | None:
    """Convert element text to float, returning ``None`` when not numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_bool(value: str | None) -> bool | None:
    """Convert a "true"/"false" XML attribute to bool, ``None`` when absent."""
    if value is None:
        return None
    return value.strip().lower() == "true"


def _condition_to_text(element: ET.Element) -> str:
    """Render a ``<conditions>`` child (``<player>``, ``<and>``, ...) as text."""
    attrs = ", ".join(f"{key}={value}" for key, value in element.attrib.items())
    children = [_condition_to_text(child) for child in element]

    if children:
        body = f"{attrs}; {', '.join(children)}" if attrs else ", ".join(children)
        return f"{element.tag.upper()}({body})"

    return f"{element.tag}({attrs})" if attrs else element.tag


def parse_single_item(item_element: ET.Element) -> dict:
    """Parses a single <item> tag and returns a dictionary with extracted attributes."""
    item_id = int(item_element.attrib["id"])
    name = item_element.attrib.get("name", "")
    item_type = item_element.attrib.get("type", "")

    # <set name="..." val="..."/> children flattened into a plain dict.
    sets: dict[str, str] = {}
    for set_el in item_element.findall("set"):
        set_name = set_el.attrib.get("name")
        if set_name:
            sets[set_name] = set_el.attrib.get("val", "")

    for unknown_set in sorted(set(sets) - set(KNOWN_SET_KEYS)):
        _UNKNOWN_SETS.add(unknown_set)

    record = {
        "id": item_id,
        "name": name,
        "type": item_type,
    }

    for key in SET_STRING_KEYS:
        record[key] = _clean(sets.get(key))

    for key in SET_INT_KEYS:
        record[key] = _to_int(sets.get(key))

    for key in SET_BOOL_KEYS:
        record[key] = _to_bool(sets.get(key))

    # Combat stats, defaulting to 0.0 like the NPC converter.
    stats = dict.fromkeys(STAT_KEYS, 0.0)
    stats_el = item_element.find("stats")
    if stats_el is not None:
        for stat_el in stats_el.findall("stat"):
            stat_type = stat_el.attrib.get("type", "")
            stat_value = _to_float(stat_el.text)
            if stat_type in stats:
                stats[stat_type] = stat_value if stat_value is not None else 0.0
            else:
                _UNKNOWN_STATS.add(stat_type)

    # Skills granted by the item (passive / triggered item skills).
    skills = []
    skills_el = item_element.find("skills")
    if skills_el is not None:
        for skill_el in skills_el.findall("skill"):
            skills.append({
                "id": _to_int(skill_el.attrib.get("id")) or 0,
                "level": _to_int(skill_el.attrib.get("level")) or 1,
                "chance": _to_int(skill_el.attrib.get("type_chance")),
                "type": _clean(skill_el.attrib.get("type")),
            })

    # Conditions required to equip the item (race / class / castle / level ...).
    conditions_msg_id = None
    conditions_text = None
    conditions_el = item_element.find("conditions")
    if conditions_el is not None:
        conditions_msg_id = _to_int(conditions_el.attrib.get("msgId"))
        conditions_text = _clean("; ".join(_condition_to_text(child) for child in conditions_el))

    record["stats"] = stats
    record["skills"] = skills
    record["conditions_msg_id"] = conditions_msg_id
    record["conditions_text"] = conditions_text
    record["text_representation"] = _build_text_representation(record, stats, skills, conditions_text)

    return record


def _build_text_representation(
    record: dict,
    stats: dict,
    skills: list[dict],
    conditions_text: str | None,
) -> str:
    """Compose a natural-language summary used for RAG / vector search."""
    lines = [f"Item '{record['name']}' (ID: {record['id']}), type {record['type']}."]

    sub_type = record["weapon_type"] or record["armor_type"] or record["etcitem_type"]
    if sub_type:
        lines.append(f"Sub-type: {sub_type}.")
    if record["material"]:
        lines.append(f"Material: {record['material']}.")
    if record["bodypart"]:
        lines.append(f"Equip slot: {record['bodypart']}.")
    if record["crystal_type"]:
        lines.append(f"Crystal grade: {record['crystal_type']}.")

    trade = []
    if record["weight"] is not None:
        trade.append(f"Weight={record['weight']}")
    if record["price"] is not None:
        trade.append(f"Price={record['price']}")
    for key, label in (
        ("is_tradable", "Tradable"),
        ("is_dropable", "Dropable"),
        ("is_sellable", "Sellable"),
        ("is_stackable", "Stackable"),
        ("is_questitem", "QuestItem"),
    ):
        if record[key] is not None:
            trade.append(f"{label}={record[key]}")
    if trade:
        lines.append("Trade: " + ", ".join(trade) + ".")

    active_stats = [f"{key}={value}" for key, value in stats.items() if value]
    if active_stats:
        lines.append("Stats: " + ", ".join(active_stats) + ".")

    if skills:
        skill_str = ", ".join(f"SkillID:{skill['id']} (Lvl {skill['level']})" for skill in skills)
        lines.append(f"Skills: [{skill_str}].")

    if conditions_text:
        lines.append(f"Equip requirements: {conditions_text}.")

    return " ".join(lines)


def _build_schema() -> pa.Schema:
    """Builds the explicit Arrow schema used for the exported item table."""
    fields = [
        ("id", pa.int64()),
        ("name", pa.string()),
        ("type", pa.string()),
    ]
    fields += [(key, pa.string()) for key in SET_STRING_KEYS]
    fields += [(key, pa.int64()) for key in SET_INT_KEYS]
    fields += [(key, pa.bool_()) for key in SET_BOOL_KEYS]
    fields.append(("stats", pa.struct([(key, pa.float64()) for key in STAT_KEYS])))
    fields.append((
        "skills",
        pa.list_(pa.struct([
            ("id", pa.int64()),
            ("level", pa.int64()),
            ("chance", pa.int64()),
            ("type", pa.string()),
        ])),
    ))
    fields.append(("conditions_msg_id", pa.int64()))
    fields.append(("conditions_text", pa.string()))
    fields.append(("text_representation", pa.string()))
    return pa.schema(fields)


def process_directory(input_dir: str, output_parquet: str):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Target directory does not exist: {input_dir}")

    # Recursively find all .xml files in directory and subdirectories
    xml_files = sorted(input_path.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files in '{input_dir}'")

    all_items = []
    failed_files = 0

    for file_path in xml_files:
        try:
            root = ET.parse(file_path).getroot()

            # Handle both root <list><item>...</item></list> and standalone <item> root tags
            item_elements = root.findall(".//item") if root.tag != "item" else [root]

            for item_el in item_elements:
                all_items.append(parse_single_item(item_el))

        except Exception as e:
            print(f"[Parsing Error] File: {file_path.name} -> {e}")
            failed_files += 1

    print(f"Successfully parsed {len(all_items)} items (File errors: {failed_files}).")

    # The 'custom' subfolder may override item definitions found in the base files.
    duplicate_ids = sorted(
        item_id for item_id, count in Counter(item["id"] for item in all_items).items() if count > 1
    )
    if duplicate_ids:
        print(f"[Warning] {len(duplicate_ids)} duplicated item IDs (base vs 'custom' overrides), first 10: {duplicate_ids[:10]}")

    if _UNKNOWN_SETS:
        print(f"[Warning] Unmapped <set> names (dropped): {sorted(_UNKNOWN_SETS)}")
    if _UNKNOWN_STATS:
        print(f"[Warning] Unmapped <stat> types (dropped): {sorted(_UNKNOWN_STATS)}")

    if not all_items:
        print("No item records found to export.")
        return

    table = pa.Table.from_pylist(all_items, schema=_build_schema())
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius Item XML to Parquet Converter.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to directory with item XML files (e.g., L2J_Mobius/.../stats/items)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="l2jc6_items.parquet",
        help="Path/Name of output Parquet file (default: l2jc6_items.parquet)"
    )

    args = parser.parse_args()
    process_directory(args.input, args.output)
