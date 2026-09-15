import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_single_npc(npc_element: ET.Element) -> dict:
    """Parses a single <npc> tag and returns a dictionary with extracted attributes."""
    npc_id = int(npc_element.attrib.get("id"))
    name = npc_element.attrib.get("name", "")
    title = npc_element.attrib.get("title", None)
    npc_type = npc_element.attrib.get("type", "")
    level = int(npc_element.attrib.get("level", 0))

    race = npc_element.findtext("race", "")
    sex = npc_element.findtext("sex", "")

    # Acquire (EXP / SP)
    acq = npc_element.find("acquire")
    exp = int(acq.attrib.get("exp", 0)) if acq is not None else 0
    sp = int(acq.attrib.get("sp", 0)) if acq is not None else 0

    # Stats
    stats_el = npc_element.find("stats")
    stats = {
        "hp": 0.0,
        "mp": 0.0,
        "p_atack": 0.0,
        "m_atack": 0.0,
        "p_def": 0.0,
        "m_def": 0.0,
        "walk_speed": 0.0,
        "run_speed": 0.0,
    }

    if stats_el is not None:
        vitals = stats_el.find("vitals")
        if vitals is not None:
            stats["hp"] = float(vitals.attrib.get("hp", 0.0))
            stats["mp"] = float(vitals.attrib.get("mp", 0.0))

        attack = stats_el.find("attack")
        if attack is not None:
            stats["p_atack"] = float(attack.attrib.get("physical", 0.0))
            stats["m_atack"] = float(attack.attrib.get("magical", 0.0))

        defence = stats_el.find("defence")
        if defence is not None:
            stats["p_def"] = float(defence.attrib.get("physical", 0.0))
            stats["m_def"] = float(defence.attrib.get("magical", 0.0))

        speed = stats_el.find("speed")
        if speed is not None:
            walk_el = speed.find("walk")
            if walk_el is not None:
                stats["walk_speed"] = float(walk_el.attrib.get("ground", 0.0))
            
            run_el = speed.find("run")
            if run_el is not None:
                stats["run_speed"] = float(run_el.attrib.get("ground", 0.0))

    # Skills
    skills = []
    skill_list = npc_element.find("skillList")
    if skill_list is not None:
        for s in skill_list.findall("skill"):
            skills.append({
                "id": int(s.attrib["id"]),
                "level": int(s.attrib["level"])
            })

    # Drops & Spoil
    drop_items = []
    spoil_items = []
    drop_lists = npc_element.find("dropLists")

    if drop_lists is not None:
        # Drop Groups
        for drop_group in drop_lists.findall("drop/group"):
            group_chance = float(drop_group.attrib.get("chance", 100.0))
            for item in drop_group.findall("item"):
                item_chance = float(item.attrib.get("chance", 0.0))
                # Absolute drop chance calculation
                abs_chance = item_chance * (group_chance / 100.0)
                drop_items.append({
                    "item_id": int(item.attrib["id"]),
                    "min": int(item.attrib["min"]),
                    "max": int(item.attrib["max"]),
                    "chance": round(abs_chance, 6)
                })

        # Spoil
        for spoil_item in drop_lists.findall("spoil/item"):
            spoil_items.append({
                "item_id": int(spoil_item.attrib["id"]),
                "min": int(spoil_item.attrib["min"]),
                "max": int(spoil_item.attrib["max"]),
                "chance": float(spoil_item.attrib.get("chance", 0.0))
            })

    # AI & Aggro
    ai_el = npc_element.find("ai")
    is_aggressive = False
    aggro_range = 0
    if ai_el is not None:
        is_aggressive = ai_el.attrib.get("isAggressive", "false").lower() == "true"
        aggro_range = int(ai_el.attrib.get("aggroRange", 0))

    # Text Representation for RAG / Vector Search
    skills_str = ", ".join([f"SkillID:{s['id']} (Lvl {s['level']})" for s in skills])
    drop_str = ", ".join([f"Item:{d['item_id']} ({d['chance']}%)" for d in drop_items[:5]])

    text_rag = (
        f"NPC '{name}' (ID: {npc_id}), Level {level} {race} {npc_type}. "
        f"Title: '{title if title else 'None'}'. "
        f"Stats: HP={stats['hp']}, MP={stats['mp']}, P.Atk={stats['p_atack']}, P.Def={stats['p_def']}, "
        f"WalkSpeed={stats['walk_speed']}, RunSpeed={stats['run_speed']}. "
        f"Aggressive: {is_aggressive} (Aggro Range: {aggro_range}). "
        f"Exp Reward: {exp}, SP Reward: {sp}. "
        f"Key Skills: [{skills_str}]. "
        f"Main Drops: [{drop_str}]."
    )

    return {
        "id": npc_id,
        "name": name,
        "title": title,
        "type": npc_type,
        "level": level,
        "race": race,
        "sex": sex,
        "exp": exp,
        "sp": sp,
        "stats": stats,
        "is_aggressive": is_aggressive,
        "aggro_range": aggro_range,
        "skills": skills,
        "drop_list": drop_items,
        "spoil_list": spoil_items,
        "text_representation": text_rag
    }


def process_directory(input_dir: str, output_parquet: str):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Target directory does not exist: {input_dir}")

    # Recursively find all .xml files in directory and subdirectories
    xml_files = list(input_path.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files in '{input_dir}'")

    all_npcs = []
    failed_files = 0

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Handle both root <list><npc>...</npc></list> and standalone <npc> root tags
            npc_elements = root.findall(".//npc") if root.tag != "npc" else [root]

            for npc_el in npc_elements:
                npc_data = parse_single_npc(npc_el)
                all_npcs.append(npc_data)

        except Exception as e:
            print(f"[Parsing Error] File: {file_path.name} -> {e}")
            failed_files += 1

    print(f"Successfully parsed {len(all_npcs)} NPCs (File errors: {failed_files}).")

    if not all_npcs:
        print("No NPC records found to export.")
        return

    # Convert to pandas DataFrame
    df = pd.DataFrame(all_npcs)

    # Save to Parquet
    df.to_parquet(output_parquet, engine="pyarrow", index=False)
    print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius NPC XML to Parquet Converter.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to directory with XML files (e.g., L2J_Mobius/.../stats/npcs)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="l2j_npcs.parquet",
        help="Path/Name of output Parquet file (default: l2j_npcs.parquet)"
    )

    args = parser.parse_args()
    process_directory(args.input, args.output)
