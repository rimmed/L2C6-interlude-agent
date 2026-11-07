"""
L2J Mobius quest scripts (data/scripts/quests/QNNNNN_Name/*.java + *.htm) to Parquet: one row per quest.

The quests are Java code, so the facts are taken with patterns and are best effort: the NPCs, monsters and items
named by a constant, a literal or an int array are resolved; ``unresolved`` lists what was not (e.g. ids built in
code). ``source`` keeps the Java code and ``htmls`` the dialogs ({file, html}) for what the patterns miss.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

INT_CONST = re.compile(r"(?:private|protected|public)?\s*static\s+final\s+int\s+(\w+)\s*=\s*(-?\d+)\s*;")
INT_ARRAY = re.compile(r"static\s+final\s+int\[\]\s+(\w+)\s*=\s*(?:new\s+int\[\]\s*)?\{([^}]*)\}", re.S)
MAP_KEYS = re.compile(r"(\w+)\.put\(\s*(\w+)\s*,")
CALL = r"{name}\s*\(([^;]*?)\)\s*;"


def _calls(code: str, name: str) -> list[str]:
    return re.findall(CALL.format(name=name), code, re.S)


def _resolve(args: str, consts: dict, arrays: dict, maps: dict, unresolved: set) -> list[int]:
    ids = []
    for token in (t.strip() for t in re.split(r",", args)):
        token = re.sub(r"//.*", "", token).strip()
        if not token:
            continue
        if re.fullmatch(r"-?\d+", token):
            ids.append(int(token))
        elif token in consts:
            ids.append(consts[token])
        elif token in arrays:
            ids.extend(arrays[token])
        elif m := re.fullmatch(r"(\w+)\.keySet\(\)", token):
            ids.extend(maps.get(m[1], []))
            if m[1] not in maps:
                unresolved.add(token)
        else:
            unresolved.add(token)
    return ids


def _given(code: str, name: str, consts: dict, unresolved: set) -> list[list[int]]:
    """[item id, count] of giveItems / rewardItems(player, ITEM, COUNT); count -1 when it is an expression."""
    out = []
    for args in _calls(code, name):
        parts = [p.strip() for p in args.split(",")]
        if len(parts) < 3:
            continue
        item, count = parts[1], parts[2]
        item_id = int(item) if re.fullmatch(r"\d+", item) else consts.get(item)
        if item_id is None:
            unresolved.add(item)
            continue
        out.append([item_id, int(count) if re.fullmatch(r"\d+", count) else consts.get(count, -1)])
    return out


def parse_quest(folder: Path) -> dict | None:
    sources = sorted(folder.glob("*.java"))
    if not sources:
        return None
    code = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sources)
    header = re.search(r'super\(\s*(\d+)\s*,\s*"([^"]*)"', code)
    if not header:
        return None
    consts = {k: int(v) for k, v in INT_CONST.findall(code)}
    arrays = {k: [int(x) if re.fullmatch(r"-?\d+", x.strip()) else consts.get(x.strip(), 0)
                  for x in v.split(",") if x.strip()] for k, v in INT_ARRAY.findall(code)}
    maps: dict[str, list[int]] = {}
    for name, key in MAP_KEYS.findall(code):
        if re.fullmatch(r"\d+", key) or key in consts:
            maps.setdefault(name, []).append(int(key) if key.isdigit() else consts[key])
    unresolved: set[str] = set()

    def ids(call: str) -> list[int]:
        return sorted({i for args in _calls(code, call) for i in _resolve(args, consts, arrays, maps, unresolved)})

    levels_min = [int(x) for x in re.findall(r"getLevel\(\)\s*<\s*(\d+)", code)]
    levels_max = [int(x) for x in re.findall(r"getLevel\(\)\s*>\s*(\d+)", code)]
    if "MIN_LEVEL" in consts:
        levels_min.append(consts["MIN_LEVEL"])
    if "MAX_LEVEL" in consts:
        levels_max.append(consts["MAX_LEVEL"])
    htmls = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.glob("*.htm*"))}
    return {
        "id": int(header[1]),
        "name": header[2],
        "script": folder.name,
        "start_npcs": ids("addStartNpc"),
        "talk_npcs": ids("addTalkId"),
        "kill_npcs": ids("addKillId"),
        "attack_npcs": ids("addAttackId"),
        "quest_items": ids("registerQuestItems"),
        "min_level": min(levels_min) if levels_min else None,
        "max_level": max(levels_max) if levels_max else None,
        "races": sorted(set(re.findall(r"Race\.(\w+)", code))),
        "race_conditions": sorted(set(re.findall(r"getRace\(\)\s*[!=]=\s*Race\.\w+", code))),
        "mage_condition": "isMageClass()" in code,
        "class_conditions": sorted(set(re.findall(r"PlayerClass\.(\w+)", code))),
        "repeatable": bool(re.search(r"exitQuest\(\s*(true|QuestType\.REPEATABLE)", code)),
        "given_items": _given(code, "giveItems", consts, unresolved) + _given(code, "rewardItems", consts, unresolved),
        "adena": [int(x) for x in re.findall(r"giveAdena\(\s*\w+\s*,\s*(\d+)", code)],
        "exp_sp": [[int(a), int(b)] for a, b in re.findall(r"addExpAndSp\(\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)", code)],
        "constants": consts,
        "unresolved": sorted(unresolved),
        "htmls": htmls,
        "source": code,
    }


def process_directory(input_dir: str, output_parquet: str) -> None:
    rows, failed = [], 0
    for folder in sorted(p for p in Path(input_dir).iterdir() if p.is_dir()):
        try:
            row = parse_quest(folder)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"[Parsing Error] Quest: {folder.name} -> {e}")
            failed += 1
    print(f"Successfully parsed {len(rows)} quests (errors: {failed}).")
    if rows:
        df = pd.DataFrame(rows)
        df["constants"] = df["constants"].apply(lambda d: [{"name": k, "value": v} for k, v in d.items()])
        df["htmls"] = df["htmls"].apply(lambda d: [{"file": k, "html": v} for k, v in d.items()])
        df.to_parquet(output_parquet, engine="pyarrow", index=False)
        print(f"Data successfully saved to Parquet file: {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2J Mobius quest scripts to Parquet Converter.")
    parser.add_argument("-i", "--input", required=True, help="Directory with the quest folders (e.g. .../data/scripts/quests)")
    parser.add_argument("-o", "--output", default="l2jc6_quests.parquet", help="Output Parquet file")
    args = parser.parse_args()
    process_directory(args.input, args.output)
