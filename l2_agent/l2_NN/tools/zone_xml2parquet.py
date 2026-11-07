"""
L2J Mobius zone XML (data/zones) to Parquet: one row per zone.

``shape`` is ``Cuboid`` (``nodes`` = two opposite corners), ``NPoly`` (``nodes`` = the polygon) or ``Cylinder``
(``nodes`` = the centre, ``radius``); the zone spans ``min_z`` .. ``max_z``. ``stats`` keeps the zone's
``<stat name val>`` settings ({name, value}). A water zone's ``max_z`` is the water surface.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def _int(value):
    return int(value) if value not in (None, "") else None


def parse_file(path: Path, root_dir: Path) -> list[dict]:
    rows = []
    for zone in ET.parse(path).getroot().iter("zone"):
        rows.append({
            "id": _int(zone.attrib.get("id")),
            "name": zone.attrib.get("name"),
            "type": zone.attrib.get("type"),
            "shape": zone.attrib.get("shape"),
            "min_z": _int(zone.attrib.get("minZ")),
            "max_z": _int(zone.attrib.get("maxZ")),
            "radius": _int(zone.attrib.get("rad")),
            "nodes": [[int(n.attrib["X"]), int(n.attrib["Y"])] for n in zone.findall("node")],
            "stats": [{"name": s.attrib.get("name"), "value": s.attrib.get("val")} for s in zone.findall("stat")],
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
    print(f"Successfully parsed {len(rows)} zones (File errors: {failed}).")
    if rows:
        pd.DataFrame(rows).to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius zone XML to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="Directory with the zone XML files (e.g. .../data/zones)")
    parser.add_argument("-o", "--output", default="l2jc6_zones.parquet", help="Output Parquet file")
    args = parser.parse_args()
    process_directory(args.input, args.output)
