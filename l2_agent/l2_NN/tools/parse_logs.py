import re
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

# --- Regular Expressions for Blocks ---

TIMESTAMP_CHAR_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+INFO\s+\d+\s+playeraction\s+\[(?P<character>[^\]]+)\]\s+(?P<action_raw>.*?)\s*\[CharStatus:"
)

CHAR_STATUS_REGEX = re.compile(
    r"\[CharStatus:\s*HP:(?P<hp>[\d.]+)/(?P<max_hp>[\d.]+)\s+MP:(?P<mp>[\d.]+)/(?P<max_mp>[\d.]+)\s+CP:(?P<cp>[\d.]+)/(?P<max_cp>[\d.]+)\s+Exp:(?P<exp>\d+)\s+Sp:(?P<sp>\d+)\s+Lvl:(?P<lvl>\d+)\]"
)

CHAR_STATS_REGEX = re.compile(
    r"\[CharStats:\s*P\.Atk:(?P<p_atk>\d+)\s+M\.Atk:(?P<m_atk>\d+)\s+P\.Def:(?P<p_def>\d+)\s+M\.Def:(?P<m_def>\d+)\s+Accur:(?P<accur>\d+)\s+Crit\.rate:(?P<crit_rate>\d+)\s+Evasion:(?P<evasion>\d+)\s+Atk\.Speed:(?P<atk_speed>[\d.]+)\s+Cast\.Speed:(?P<cast_speed>\d+)\s+STR:(?P<str>\d+)\s+INT:(?P<int>\d+)\s+DEX:(?P<dex>\d+)\s+CON:(?P<con>\d+)\s+WIT:(?P<wit>\d+)\]"
)

WEAPON_BLOCK_REGEX = re.compile(
    r"\[(?P<equip_str>Weapon:[^\]]+)\]"
)

POS_REGEX = re.compile(
    r"\[Pos:(?P<pos_x>-?\d+),(?P<pos_y>-?\d+),(?P<pos_z>-?\d+)\s+Head:(?P<head>\d+)\s+Sit:(?P<sit>true|false)\s+Cast:(?P<cast>true|false)\s+Atk:(?P<atk>true|false)\s+Dead:(?P<dead>true|false)\s+Wt:(?P<wt>[\d.]+)%\]"
)

ENV_BLOCK_REGEX = re.compile(
    r"\[Env:\s*(?P<env_str>.*?)\]\s*\[CD:"
)

MONSTER_REGEX = re.compile(
    r"Monster:\s*(?P<name>[^(]+)\s*\(NPC_ID:\s*(?P<npc_id>\d+),\s*Lvl:\s*(?P<lvl>\d+),\s*Distance:\s*(?P<distance>[\d.]+)\)"
)

ENV_META_REGEX = re.compile(
    r"GroundItems:\s*(?P<ground_items>\d+),\s*InCombat:\s*(?P<in_combat>true|false)"
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


def parse_env(line: str) -> tuple[list[dict], int, bool]:
    match_env = ENV_BLOCK_REGEX.search(line)
    if not match_env:
        return [], 0, False

    env_str = match_env.group("env_str")

    monsters = []
    for match in MONSTER_REGEX.finditer(env_str):
        data = match.groupdict()
        monsters.append({
            "type": "monster",
            "name": data["name"].strip(),
            "id": int(data["npc_id"]),
            "lvl": int(data["lvl"]),
            "distance": float(data["distance"])
        })

    ground_items = 0
    in_combat = False
    match_meta = ENV_META_REGEX.search(env_str)
    if match_meta:
        ground_items = int(match_meta.group("ground_items"))
        in_combat = match_meta.group("in_combat").lower() == "true"

    return monsters, ground_items, in_combat


def parse_log_line(line: str) -> dict | None:
    m_base = TIMESTAMP_CHAR_REGEX.search(line)
    m_status = CHAR_STATUS_REGEX.search(line)
    m_stats = CHAR_STATS_REGEX.search(line)
    m_pos = POS_REGEX.search(line)

    if not (m_base and m_status and m_pos):
        return None

    pos_data = m_pos.groupdict()
    monsters, ground_items, in_combat = parse_env(line)
    stats_data = m_stats.groupdict() if m_stats else {}

    return {
        "timestamp": m_base.group("timestamp"),
        "character": m_base.group("character"),
        "action_raw": m_base.group("action_raw").strip(),
        # CharStatus
        "hp": float(m_status.group("hp")),
        "max_hp": float(m_status.group("max_hp")),
        "mp": float(m_status.group("mp")),
        "max_mp": float(m_status.group("max_mp")),
        "cp": float(m_status.group("cp")),
        "max_cp": float(m_status.group("max_cp")),
        "exp": int(m_status.group("exp")),
        "sp": int(m_status.group("sp")),
        "lvl": int(m_status.group("lvl")),
        # CharStats
        "char_stats": {
            "p_atk": int(stats_data.get("p_atk", 0)),
            "m_atk": int(stats_data.get("m_atk", 0)),
            "p_def": int(stats_data.get("p_def", 0)),
            "m_def": int(stats_data.get("m_def", 0)),
            "accur": int(stats_data.get("accur", 0)),
            "crit_rate": int(stats_data.get("crit_rate", 0)),
            "evasion": int(stats_data.get("evasion", 0)),
            "atk_speed": float(stats_data.get("atk_speed", 0.0)),
            "cast_speed": int(stats_data.get("cast_speed", 0)),
            "str": int(stats_data.get("str", 0)),
            "int": int(stats_data.get("int", 0)),
            "dex": int(stats_data.get("dex", 0)),
            "con": int(stats_data.get("con", 0)),
            "wit": int(stats_data.get("wit", 0)),
        },
        # Equipment
        "equipment": parse_equipment(line),
        # Pos
        "pos_x": int(pos_data["pos_x"]),
        "pos_y": int(pos_data["pos_y"]),
        "pos_z": int(pos_data["pos_z"]),
        "head": int(pos_data["head"]),
        "is_sitting": pos_data["sit"].lower() == "true",
        "is_casting": pos_data["cast"].lower() == "true",
        "is_attacking": pos_data["atk"].lower() == "true",
        "is_dead": pos_data["dead"].lower() == "true",
        "weight_pct": float(pos_data["wt"]),
        # Env
        "ground_items": ground_items,
        "in_combat": in_combat,
        "nearby_objects": monsters,
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
        # CharStatus
        ("hp", pa.float64()),
        ("max_hp", pa.float64()),
        ("mp", pa.float64()),
        ("max_mp", pa.float64()),
        ("cp", pa.float64()),
        ("max_cp", pa.float64()),
        ("exp", pa.int64()),
        ("sp", pa.int64()),
        ("lvl", pa.int64()),
        # CharStats
        ("char_stats", pa.struct([
            ("p_atk", pa.int64()),
            ("m_atk", pa.int64()),
            ("p_def", pa.int64()),
            ("m_def", pa.int64()),
            ("accur", pa.int64()),
            ("crit_rate", pa.int64()),
            ("evasion", pa.int64()),
            ("atk_speed", pa.float64()),
            ("cast_speed", pa.int64()),
            ("str", pa.int64()),
            ("int", pa.int64()),
            ("dex", pa.int64()),
            ("con", pa.int64()),
            ("wit", pa.int64()),
        ])),
        # Equipment
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
        # Pos
        ("pos_x", pa.int64()),
        ("pos_y", pa.int64()),
        ("pos_z", pa.int64()),
        ("head", pa.int64()),
        ("is_sitting", pa.bool_()),
        ("is_casting", pa.bool_()),
        ("is_attacking", pa.bool_()),
        ("is_dead", pa.bool_()),
        ("weight_pct", pa.float64()),
        # Env
        ("ground_items", pa.int64()),
        ("in_combat", pa.bool_()),
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
