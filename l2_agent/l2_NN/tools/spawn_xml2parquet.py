"""
L2J Mobius spawn XML (data/spawns) to Parquet: one row per <npc> of a <spawn> group.

A spawn is either a fixed point (x, y, z, heading) or ``count`` monsters spread over the group's territories
(polygons with minZ / maxZ); ``territories`` and ``banned_territories`` hold them as lists of {min_z, max_z, nodes}.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def _int(element: ET.Element, name: str, default=None):
    value = element.attrib.get(name)
    return int(value) if value not in (None, "") else default


def _territory(element: ET.Element) -> dict:
    return {
        "min_z": _int(element, "minZ"),
        "max_z": _int(element, "maxZ"),
        "nodes": [[int(n.attrib["x"]), int(n.attrib["y"])] for n in element.findall("node")],
    }


def parse_file(path: Path, root_dir: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    enabled = root.attrib.get("enabled", "true").lower() != "false"
    rows = []
    for spawn in root.findall("spawn"):
        territories = [_territory(t) for t in spawn.findall("territory")]
        banned = [_territory(t) for t in spawn.findall("banned_territory")]
        for npc in spawn.findall("npc"):
            rows.append({
                "npc_id": int(npc.attrib["id"]),
                "x": _int(npc, "x"),
                "y": _int(npc, "y"),
                "z": _int(npc, "z"),
                "heading": _int(npc, "heading"),
                "count": _int(npc, "count", 1),
                "respawn_delay": _int(npc, "respawnDelay"),
                "respawn_random": _int(npc, "respawnRandom"),
                "chase_range": _int(npc, "chaseRange"),
                "period_of_day": npc.attrib.get("periodOfDay"),
                "territories": territories,
                "banned_territories": banned,
                "spawn_name": spawn.attrib.get("name"),
                "zone": spawn.attrib.get("zone"),
                "enabled": enabled,
                "file": str(path.relative_to(root_dir)),
            })
    return rows


def process_directory(input_dir: str, output_parquet: str) -> None:
    root_dir = Path(input_dir)
    rows, failed = [], 0
    for path in sorted(root_dir.rglob("*.xml")):
        try:
            rows.extend(parse_file(path, root_dir))
        except Exception as e:
            print(f"[Parsing Error] File: {path} -> {e}")
            failed += 1
    print(f"Successfully parsed {len(rows)} spawns (File errors: {failed}).")
    if rows:
        pd.DataFrame(rows).to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius spawn XML to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="Directory with the spawn XML files (e.g. .../data/spawns)")
    parser.add_argument("-o", "--output", default="l2jc6_spawns.parquet", help="Output Parquet file")
    args = parser.parse_args()
    process_directory(args.input, args.output)
