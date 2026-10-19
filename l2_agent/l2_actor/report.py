"""
K/D report of a fleet run: kills and deaths per class and per level range.

Reads the fight summaries the actors write (``episodes_<account>.jsonl`` in the fleet's
``--output-dir``): one record per fight, ``reason`` ``kill`` or ``death`` plus the character's
``class_id`` and ``level`` at that moment (agents level up during a run, so a character can appear
in several level ranges). Fights cut off by stopping the actor are left out.

Usage (from ``l2_agent/``)::

    uv run python -m l2_actor.report checkpoints/rl_bc_v4/fleet_run2
    uv run python -m l2_actor.report checkpoints/rl_bc_v4/fleet_run1 checkpoints/rl_bc_v4/fleet_run2 --by class-level
    uv run python -m l2_actor.report checkpoints/rl_bc_v4/fleet_run2 --split 4 --csv kd.csv
    uv run python -m l2_actor.report checkpoints/rl_bc_v4/fleet_run2b --by skills --top 6

``--by actions`` / ``--by skills`` show what each class does in a fight (from runs whose actors
recorded their decisions): the share of every action type, and the skills it casts most.

``--split N`` cuts the run into N equal time windows (by the record time ``t``; runs without it are
cut by each actor's fight order) and adds the K/D of every window, which shows how it changed while
the policy was learning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CLASS_LIST = (Path(__file__).resolve().parents[2] / "L2J_Mobius" / "L2J_Mobius_CT_0_Interlude" / "dist" / "game"
              / "data" / "stats" / "players" / "classList.xml")


def class_names(path: Path = CLASS_LIST) -> dict[int, str]:
    """``class_id -> name`` from the datapack (``Gladiator``, ``Elemental Summoner`` ...)."""
    if not path.exists():
        return {}
    return {int(c.get("classId")): c.get("name") for c in ET.parse(path).getroot().iter("class")}


@dataclass
class Fight:
    actor: str
    character: str
    class_id: int
    level: int
    kill: bool
    reward: float
    #: position in time, 0..1 within the run (record time, else the actor's fight order)
    phase: float = 0.0
    #: decisions per action type and skill uses (``"<id>"``, ``"pet:<id>"``); ``None`` in older runs
    actions: dict[str, int] | None = None
    skills: dict[str, int] | None = None


def load(paths: list[Path]) -> list[Fight]:
    """Every finished fight of the given runs (directories of ``episodes_*.jsonl`` or the files)."""
    files: list[Path] = []
    for path in paths:
        files += sorted(path.glob("**/episodes_*.jsonl")) if path.is_dir() else [path]
    if not files:
        sys.exit(f"no episodes_*.jsonl under {', '.join(map(str, paths))}")

    fights: list[Fight] = []
    timed: list[tuple[Fight, float]] = []
    for file in files:
        records = []
        with open(file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("reason") in ("kill", "death") and not record.get("truncated"):
                    records.append(record)
        for i, record in enumerate(records):
            fight = Fight(actor=record.get("actor_id", file.stem), character=record.get("character", "?"),
                          class_id=int(record.get("class_id", -1)), level=int(record.get("level", 0)),
                          kill=record["reason"] == "kill", reward=float(record.get("return", 0.0)),
                          phase=i / max(len(records) - 1, 1), actions=record.get("actions"),
                          skills=record.get("skills"))
            fights.append(fight)
            if "t" in record:
                timed.append((fight, float(record["t"])))
    if timed and len(timed) == len(fights):
        start = min(t for _, t in timed)
        span = max(max(t for _, t in timed) - start, 1e-9)
        for fight, t in timed:
            fight.phase = (t - start) / span
    return fights


@dataclass
class Stats:
    kills: int = 0
    deaths: int = 0
    reward: float = 0.0
    characters: set[str] = field(default_factory=set)
    windows: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def add(self, fight: Fight, window: int) -> None:
        self.kills += fight.kill
        self.deaths += not fight.kill
        self.reward += fight.reward
        self.characters.add(fight.character)
        self.windows[window][0 if fight.kill else 1] += 1

    @property
    def fights(self) -> int:
        return self.kills + self.deaths


def kd(kills: int, deaths: int) -> float:
    return kills / deaths if deaths else math.inf if kills else math.nan


def level_range(level: int, step: int) -> str:
    low = (level // step) * step
    return f"{low:02d}-{low + step - 1:02d}"


def group_key(fight: Fight, by: str, names: dict[int, str], step: int) -> tuple:
    name = names.get(fight.class_id, f"class {fight.class_id}")
    if by == "class":
        return (name,)
    if by == "level":
        return (level_range(fight.level, step),)
    return (name, level_range(fight.level, step))


def build(fights: list[Fight], by: str, names: dict[int, str], step: int, split: int) -> dict[tuple, Stats]:
    table: dict[tuple, Stats] = defaultdict(Stats)
    for fight in fights:
        window = min(int(fight.phase * split), split - 1)
        table[group_key(fight, by, names, step)].add(fight, window)
        table[("TOTAL",) + ("",) * (by == "class-level")].add(fight, window)
    return table


def fmt_kd(value: float) -> str:
    return "inf" if math.isinf(value) else "-" if math.isnan(value) else f"{value:.2f}"


def rows(table: dict[tuple, Stats], by: str, split: int, min_fights: int, sort: str) -> tuple[list[str], list[list]]:
    head = ["class", "levels"] if by == "class-level" else ["class" if by == "class" else "levels"]
    head += ["chars", "fights", "kills", "deaths", "K/D", "reward/fight"]
    if split > 1:
        head += [f"K/D w{w + 1}" for w in range(split)]
    body = []
    for key, s in table.items():
        if key[0] == "TOTAL" or s.fights < min_fights:
            continue
        body.append((key, s))
    order = {
        "kd": lambda item: -kd(item[1].kills, item[1].deaths) if item[1].deaths else -1e9,
        "fights": lambda item: -item[1].fights,
        "name": lambda item: item[0],
    }[sort]
    body.sort(key=order)
    body.append(next((key, s) for key, s in table.items() if key[0] == "TOTAL"))
    out = []
    for key, s in body:
        row = list(key) + [len(s.characters), s.fights, s.kills, s.deaths, fmt_kd(kd(s.kills, s.deaths)),
                           f"{s.reward / max(s.fights, 1):+.3f}"]
        if split > 1:
            row += [fmt_kd(kd(*s.windows[w])) if sum(s.windows[w]) else "-" for w in range(split)]
        out.append(row)
    return head, out


#: Action types in the order of the action table, with their column titles.
ACTION_COLUMNS = (("selected_target", "target"), ("attacked", "attack"), ("used_skill", "skill"), ("moved", "move"),
                  ("picked_up", "pickup"), ("used_item", "potion"), ("pet_attack", "pet atk"), ("pet_skill", "pet skill"))
DATAPACK_SKILLS = CLASS_LIST.parents[1] / "skills"


def action_rows(fights: list[Fight], names: dict[int, str], min_fights: int) -> tuple[list[str], list[list]]:
    """Per class: decisions per fight and the share of each action type."""
    per_class: dict[str, list] = defaultdict(lambda: [0, defaultdict(int)])
    for fight in fights:
        if fight.actions is None:
            continue
        for key in (names.get(fight.class_id, f"class {fight.class_id}"), "TOTAL"):
            per_class[key][0] += 1
            for action, n in fight.actions.items():
                per_class[key][1][action] += n
    head = ["class", "fights", "decisions/fight"] + [f"{title} %" for _, title in ACTION_COLUMNS]
    body = []
    for name in sorted(k for k in per_class if k != "TOTAL") + ["TOTAL"]:
        count, actions = per_class[name]
        if count < min_fights and name != "TOTAL":
            continue
        total = sum(actions.values()) or 1
        body.append([name, count, f"{total / count:.1f}"] + [f"{100 * actions.get(a, 0) / total:.1f}" for a, _ in ACTION_COLUMNS])
    return head, body


def skill_rows(fights: list[Fight], names: dict[int, str], min_fights: int, top: int) -> tuple[list[str], list[list]]:
    """Per class: skill casts per fight and its most used skills (share of its casts)."""
    from l2_client.tools.data import skill_data  # the datapack catalog, loaded only for this table

    if DATAPACK_SKILLS.exists():
        skill_data.load_skill_data(str(DATAPACK_SKILLS))
    per_class: dict[str, list] = defaultdict(lambda: [0, defaultdict(int)])
    for fight in fights:
        if fight.skills is None:
            continue
        name = names.get(fight.class_id, f"class {fight.class_id}")
        per_class[name][0] += 1
        for skill, n in fight.skills.items():
            per_class[name][1][skill] += n

    def label(key: str) -> str:
        pet, _, sid = key.rpartition(":")
        title = skill_data.get_skill_name(int(sid)) or sid
        return f"{title} (pet)" if pet else title

    head = ["class", "fights", "casts/fight", f"top {top} skills (share of casts)"]
    body = []
    for name in sorted(per_class):
        count, skills = per_class[name]
        if count < min_fights:
            continue
        total = sum(skills.values())
        best = sorted(skills.items(), key=lambda item: -item[1])[:top]
        body.append([name, count, f"{total / count:.2f}",
                     ", ".join(f"{label(k)} {100 * n / total:.0f}%" for k, n in best) if total else "-"])
    return head, body


def markdown(head: list[str], body: list[list]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    return "\n".join([line(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in body])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="K/D of a fleet run per class and level.")
    p.add_argument("paths", nargs="+", type=Path, help="Fleet --output-dir(s) or episodes_*.jsonl files")
    p.add_argument("--by", choices=("class", "level", "class-level", "actions", "skills", "all"), default="all",
                   help="Table; 'all' prints the class, level, action and skill tables (default)")
    p.add_argument("--top", type=int, default=5, help="Skills listed per class in the skill table (default 5)")
    p.add_argument("--level-step", type=int, default=10, help="Width of a level range (default 10)")
    p.add_argument("--min-fights", type=int, default=20, help="Hide groups with fewer fights (default 20)")
    p.add_argument("--split", type=int, default=1, help="Also show the K/D of N equal time windows of the run")
    p.add_argument("--sort", choices=("kd", "fights", "name"), default="name")
    p.add_argument("--csv", type=Path, help="Write the (last) table to this CSV file as well")
    args = p.parse_args(argv)

    fights = load(args.paths)
    names = class_names()
    print(f"{len(fights)} fights of {len({f.character for f in fights})} characters from {', '.join(map(str, args.paths))}\n")
    has_actions = any(f.actions is not None for f in fights)
    tables = ("class", "level") + (("actions", "skills") if has_actions else ()) if args.by == "all" else (args.by,)
    for by in tables:
        if by in ("actions", "skills"):
            if not has_actions:
                print(f"No action records in these runs (actors before the action counters): no {by} table\n")
                continue
            head, body = (action_rows(fights, names, args.min_fights) if by == "actions"
                          else skill_rows(fights, names, args.min_fights, args.top))
            print(("Actions per class (share of the decisions)" if by == "actions" else "Skills per class") + "\n")
        else:
            head, body = rows(build(fights, by, names, args.level_step, max(args.split, 1)), by, max(args.split, 1),
                              args.min_fights, args.sort)
            print(f"K/D by {by.replace('-', ' and ')}\n")
        print(markdown(head, body) + "\n")
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([head] + body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
