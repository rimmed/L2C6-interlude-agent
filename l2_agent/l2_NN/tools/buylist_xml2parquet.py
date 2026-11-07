"""
L2J Mobius buylist XML (data/buylists) to Parquet: one row per item of a merchant's list.

The list id is the file name (e.g. 3000100 for Lector's first list). ``price`` is empty when the list does not set
one: the server then sells at the item's own price (l2jc6_items) times the area's tax.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def parse_file(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    npc_ids = [int(n.text) for n in root.findall("npcs/npc") if n.text and n.text.strip()]
    rows = []
    for item in root.findall("item"):
        price, count, delay = item.attrib.get("price"), item.attrib.get("count"), item.attrib.get("restock_delay")
        rows.append({
            "list_id": int(path.stem),
            "npc_ids": npc_ids,
            "item_id": int(item.attrib["id"]),
            "price": int(price) if price else None,
            "count": int(count) if count else None,  # limited stock; empty = unlimited
            "restock_delay": int(delay) if delay else None,  # minutes
        })
    return rows


def process_directory(input_dir: str, output_parquet: str) -> None:
    rows, failed = [], 0
    for path in sorted(Path(input_dir).glob("*.xml")):
        try:
            rows.extend(parse_file(path))
        except Exception as e:
            print(f"[Parsing Error] File: {path.name} -> {e}")
            failed += 1
    print(f"Successfully parsed {len(rows)} buylist items (File errors: {failed}).")
    if rows:
        pd.DataFrame(rows).to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius buylist XML to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="Directory with the buylist XML files (e.g. .../data/buylists)")
    parser.add_argument("-o", "--output", default="l2jc6_buylists.parquet", help="Output Parquet file")
    args = parser.parse_args()
    process_directory(args.input, args.output)
