"""
L2J geodata regions (``XX_YY.l2j``) to Parquet: one row per region with its decoded cell grid, for the agent's own
path finding.

A row holds ``layers`` (uint8, 2048 x 2048: layers per cell) and ``height`` (int16) / ``nswe`` (uint8) of shape
2048 x 2048 x ``max_layers`` (the cell's layers top-down, padded with 0), all as raw little-endian bytes in
C order; ``nswe`` bits: east 1, west 2, south 4, north 8. Several input folders are layered as on the server:
a region of a later folder replaces the one of an earlier folder. Each region is its own row group, so a reader
can load only the regions it needs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geo import l2j  # noqa: E402


def collect(inputs: list[str], regions: set[str] | None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for folder in inputs:
        for path in sorted(Path(folder).glob("*.l2j")):
            if regions is None or path.stem in regions:
                files[path.stem] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="L2J geodata (.l2j) to Parquet Converter.")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="Folders with .l2j regions, in the order the server overlays them")
    parser.add_argument("-r", "--regions", nargs="*", help="Only these regions (e.g. 17_25); default all")
    parser.add_argument("-o", "--output", default="l2jc6_geodata.parquet", help="Output Parquet file")
    parser.add_argument("--update", action="store_true",
                        help="Re-read only the regions whose file is newer than the output (or comes from another "
                             "folder now) and copy the others from it: after a geodata repair")
    args = parser.parse_args()

    files = collect(args.input, set(args.regions) if args.regions else None)
    schema = pa.schema([("region", pa.string()), ("rx", pa.int64()), ("ry", pa.int64()), ("max_layers", pa.int64()),
                        ("layers", pa.binary()), ("height", pa.binary()), ("nswe", pa.binary()), ("source", pa.string())])
    # one row group per region, written as it is read: a reader loads the regions it needs, and the world does not
    # have to fit in memory here
    old: dict[str, tuple[int, int, str]] = {}
    output = Path(args.output)
    if args.update and output.exists():
        previous = pq.ParquetFile(output)
        for g in range(previous.num_row_groups):
            for i, row in enumerate(previous.read_row_group(g, columns=["region", "source"]).to_pylist()):
                old[row["region"]] = (g, i, row["source"])
        built = output.stat().st_mtime
    tmp = output.with_suffix(".tmp")
    copied = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for name, path in sorted(files.items()):
            if name in old and old[name][2] == str(path) and path.stat().st_mtime <= built:
                group, row = old[name][0], old[name][1]
                writer.write_table(previous.read_row_group(group).slice(row, 1).cast(schema))
                copied += 1
                continue
            region = l2j.read(path)
            writer.write_table(pa.table({
                "region": [name], "rx": [region.rx], "ry": [region.ry], "max_layers": [int(region.height.shape[2])],
                "layers": [np.ascontiguousarray(region.layers, np.uint8).tobytes()],
                "height": [np.ascontiguousarray(region.height, "<i2").tobytes()],
                "nswe": [np.ascontiguousarray(region.nswe, np.uint8).tobytes()],
                "source": [str(path)],
            }, schema=schema))
            print(f"{name}: {region.height.shape[2]} layer(s) max, from {path}", flush=True)
    tmp.replace(output)
    print(f"Wrote {len(files)} region(s) to {args.output} ({copied} copied unchanged)")


if __name__ == "__main__":
    main()
