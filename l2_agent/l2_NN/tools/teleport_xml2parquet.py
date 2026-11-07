"""
L2J Mobius teleporter XML (data/teleporters) to Parquet: one row per destination of a teleporter NPC.

``list_index`` is the destination's position in its list: the bypass ``teleport <type> <list name> <index>`` of the
NPC's dialog picks it. ``fee_item_id`` 57 (adena) when the XML does not name another item.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ADENA = 57


def parse_file(path: Path, root_dir: Path) -> list[dict]:
    rows = []
    for npc in ET.parse(path).getroot().findall("npc"):
        for teleport in npc.findall("teleport"):
            kind = teleport.attrib.get("type", "")
            for index, location in enumerate(teleport.findall("location")):
                fee_id, fee_count, castle = (location.attrib.get(k) for k in ("feeId", "feeCount", "castleId"))
                rows.append({
                    "npc_id": int(npc.attrib["id"]),
                    "type": kind,
                    "list_name": teleport.attrib.get("name", kind),
                    "list_index": index,
                    "name": location.attrib.get("name"),
                    "x": int(location.attrib["x"]),
                    "y": int(location.attrib["y"]),
                    "z": int(location.attrib["z"]),
                    "fee_item_id": int(fee_id) if fee_id else ADENA,
                    "fee_count": int(fee_count) if fee_count else 0,
                    "castle_ids": [int(c) for c in castle.split(";")] if castle else [],
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
    print(f"Successfully parsed {len(rows)} teleport destinations (File errors: {failed}).")
    if rows:
        pd.DataFrame(rows).to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius teleporter XML to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="Directory with the teleporter XML files (e.g. .../data/teleporters)")
    parser.add_argument("-o", "--output", default="l2jc6_teleports.parquet", help="Output Parquet file")
    args = parser.parse_args()
    process_directory(args.input, args.output)
