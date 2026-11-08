"""
L2J Mobius player skill trees (data/stats/players/skillTrees) and trainers (data/SkillLearn.xml) to Parquet.

``l2jc6_skill_trees``: one row per skill level a class learns - class_id, skill_id, skill_level, name, get_level
(the character level it needs), sp (levelUpSp), auto_get (given without a trainer), learned_by_npc, tree (the
skillTree type, e.g. classSkillTree), items (what learning it consumes: list of {item_id, count}, e.g. a
spellbook), file. ``l2jc6_trainers``: npc_id and the class_ids it teaches.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def _bool(value) -> bool:
    return str(value).lower() == "true"


def skill_trees(root_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(root_dir.rglob("*.xml")):
        for tree in ET.parse(path).getroot().iter("skillTree"):
            class_id = tree.attrib.get("classId")
            for skill in tree.findall("skill"):
                rows.append({
                    "class_id": int(class_id) if class_id is not None else None,
                    "skill_id": int(skill.attrib["skillId"]),
                    "skill_level": int(skill.attrib["skillLevel"]),
                    "name": skill.attrib.get("skillName"),
                    "get_level": int(skill.attrib.get("getLevel", 0)),
                    "sp": int(skill.attrib.get("levelUpSp", 0)),
                    "auto_get": _bool(skill.attrib.get("autoGet")),
                    "learned_by_npc": _bool(skill.attrib.get("learnedByNpc")),
                    "tree": tree.attrib.get("type"),
                    "items": [{"item_id": int(i.attrib["id"]), "count": int(i.attrib.get("count", 1))}
                              for i in skill.findall("item")],
                    "file": str(path.relative_to(root_dir)),
                })
    return rows


def trainers(path: Path) -> list[dict]:
    return [{"npc_id": int(npc.attrib["id"]), "class_ids": [int(c.text) for c in npc.findall("classId")]}
            for npc in ET.parse(path).getroot().iter("npc")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius skill trees and trainers to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="The datapack's data folder (.../dist/game/data)")
    parser.add_argument("-o", "--output-dir", default=".", help="Folder of the two Parquet files")
    args = parser.parse_args()
    data, out = Path(args.input), Path(args.output_dir)
    trees = pd.DataFrame(skill_trees(data / "stats" / "players" / "skillTrees"))
    trees.to_parquet(out / "l2jc6_skill_trees.parquet", index=False)
    teach = pd.DataFrame(trainers(data / "SkillLearn.xml"))
    teach.to_parquet(out / "l2jc6_trainers.parquet", index=False)
    print(f"{len(trees)} skill levels, {len(teach)} trainers")
