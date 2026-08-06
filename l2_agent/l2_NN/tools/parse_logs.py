import re
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

TIMESTAMP_CHAR_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+INFO\s+\d+\s+playeraction\s+\[(?P<character>[^\]]+)\]\s+(?P<action_info>.*?)\s*\[CharStatus:"
)
CHAR_STATUS_REGEX = re.compile(
    r"\[CharStatus:\s*HP:(?P<hp>[\d.]+)/(?P<max_hp>[\d.]+)\s+MP:(?P<mp>[\d.]+)/(?P<max_mp>[\d.]+)"
)
POS_REGEX = re.compile(
    r"\[Pos:(?P<pos_x>-?\d+),(?P<pos_y>-?\d+),(?P<pos_z>-?\d+)\s+Head:(?P<head>\d+)"
)
WEAPON_BLOCK_REGEX = re.compile(
    r"\[(?P<equip_str>Weapon:[^\]]+)\]"
)
ENV_BLOCK_REGEX = re.compile(
    r"\[Env:\s*(?P<env_str>.*?)\]\s*\[CD:"
)
MONSTER_REGEX = re.compile(
    r"Monster:\s*(?P<name>[^(]+)\s*\(NPC_ID:\s*(?P<npc_id>\d+),\s*Lvl:\s*(?P<lvl>\d+),\s*Distance:\s*(?P<distance>[\d.]+)\)"
)

SLOTS = ["Weapon", "Shield", "Head", "Chest", "Gloves", "Legs", "Feet", "Neck", "REar", "LEar", "RFinger", "LFinger"]

def parse_equipment(line: str) -> dict:
    match_block = WEAPON_BLOCK_REGEX.search(line)
    equipment = {slot: None for slot in SLOTS}

    if not match_block:
        return equipment

    equip_str = match_block.group("equip_str").strip()

    for slot in SLOTS:
        pattern = rf"{slot}:(.*?)(?=\s+(?:{'|'.join(SLOTS)}):|$)"
        match = re.search(pattern, equip_str)
        if match:
            val = match.group(1).strip()
            equipment[slot] = val if val != "" else None

    return equipment

def parse_env_objects(line: str) -> list[dict]:
    nearby_objects = []
    match_env = ENV_BLOCK_REGEX.search(line)
    if not match_env:
        return nearby_objects

    env_str = match_env.group("env_str")
    for match in MONSTER_REGEX.finditer(env_str):
        data = match.groupdict()
        nearby_objects.append({
            "type": "monster",
            "name": data["name"].strip(),
            "id": int(data["npc_id"]),
            "lvl": int(data["lvl"]),
            "distance": float(data["distance"])
        })
    return nearby_objects

def parse_log_line(line: str) -> dict | None:
    m_base = TIMESTAMP_CHAR_REGEX.search(line)
    m_status = CHAR_STATUS_REGEX.search(line)
    m_pos = POS_REGEX.search(line)

    if not (m_base and m_status and m_pos):
        return None

    return {
        "timestamp": m_base.group("timestamp"),
        "character": m_base.group("character"),
        "action_raw": m_base.group("action_info").strip(),
        "hp": float(m_status.group("hp")),
        "max_hp": float(m_status.group("max_hp")),
        "mp": float(m_status.group("mp")),
        "max_mp": float(m_status.group("max_mp")),
        "equipment": parse_equipment(line),
        "pos_x": int(m_pos.group("pos_x")),
        "pos_y": int(m_pos.group("pos_y")),
        "pos_z": int(m_pos.group("pos_z")),
        "head": int(m_pos.group("head")),
        "nearby_objects": parse_env_objects(line)
    }

def process_log_file_to_parquet(input_path: str, output_path: str):
    parsed_data = []

    print(f"Starting to read file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            parsed_row = parse_log_line(line)
            if parsed_row:
                parsed_data.append(parsed_row)

            if line_num % 100000 == 0:
                print(f"Processed {line_num} lines...")

    print(f"Successfully parsed {len(parsed_data)} action frames in total.")

    schema = pa.schema([
        ("timestamp", pa.string()),
        ("character", pa.string()),
        ("action_raw", pa.string()),
        ("hp", pa.float64()),
        ("max_hp", pa.float64()),
        ("mp", pa.float64()),
        ("max_mp", pa.float64()),
        ("equipment", pa.struct([
            ("Weapon", pa.string()),
            ("Shield", pa.string()),
            ("Head", pa.string()),
            ("Chest", pa.string()),
            ("Gloves", pa.string()),
            ("Legs", pa.string()),
            ("Feet", pa.string()),
            ("Neck", pa.string()),
            ("REar", pa.string()),
            ("LEar", pa.string()),
            ("RFinger", pa.string()),
            ("LFinger", pa.string()),
        ])),
        ("pos_x", pa.int64()),
        ("pos_y", pa.int64()),
        ("pos_z", pa.int64()),
        ("head", pa.int64()),
        ("nearby_objects", pa.list_(pa.struct([
            ("type", pa.string()),
            ("name", pa.string()),
            ("id", pa.int64()),
            ("lvl", pa.int64()),
            ("distance", pa.float64())
        ])))
    ])

    table = pa.Table.from_pylist(parsed_data, schema=schema)
    pq.write_table(table, output_path, compression="snappy")
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse player action log file and save to Parquet.")
    parser.add_argument("input_log", type=str, help="Path to the input log file.")
    parser.add_argument("output_parquet", type=str, help="Path to the output Parquet file.")
    args = parser.parse_args()

    process_log_file_to_parquet(args.input_log, args.output_parquet)