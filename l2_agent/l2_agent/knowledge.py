"""
The agent's knowledge of the game world: the datapack converted to Parquet (``l2_NN/tools/*2parquet.py``).

Only Parquet is read here. A territory spawn has no fixed point: :meth:`Knowledge.spawn_points` gives the centre of
its polygon, which is where the agent goes to find those monsters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

#: The Parquet files lie in the repository root, next to the ``l2_agent`` project.
DATA_DIR = Path(__file__).resolve().parents[2]
ADENA = 57


@dataclass(frozen=True)
class Point:
    """A spot of the map."""

    x: int
    y: int
    z: int

    def distance(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass(frozen=True)
class Area:
    """A rectangle of the map (x / y) with a name, e.g. a hunting ground or a town."""

    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


#: Talking Island with its village; the test region of the agent.
TALKING_ISLAND = Area("Talking Island", -118000, -60000, 200000, 262000)


class Knowledge:
    """Lazy access to the Parquet tables; each table is read on first use."""

    def __init__(self, data_dir: Path | str = DATA_DIR, prefix: str = "l2jc6_"):
        self.data_dir = Path(data_dir)
        self.prefix = prefix

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.data_dir / f"{self.prefix}{name}.parquet")

    @cached_property
    def npcs(self) -> pd.DataFrame:
        return self._read("npcs").set_index("id", drop=False)

    @cached_property
    def items(self) -> pd.DataFrame:
        return self._read("items").set_index("id", drop=False)

    @cached_property
    def spawns(self) -> pd.DataFrame:
        spawns = self._read("spawns")
        spawns = spawns[spawns["enabled"]].copy()
        centres = spawns.apply(_spawn_centre, axis=1, result_type="expand")
        spawns[["cx", "cy", "cz"]] = centres
        return spawns

    @cached_property
    def buylists(self) -> pd.DataFrame:
        return self._read("buylists")

    @cached_property
    def teleports(self) -> pd.DataFrame:
        return self._read("teleports")

    @cached_property
    def skill_trees(self) -> pd.DataFrame:
        return self._read("skill_trees")

    @cached_property
    def trainers(self) -> pd.DataFrame:
        return self._read("trainers")

    @cached_property
    def zones(self) -> pd.DataFrame:
        return self._read("zones")

    @cached_property
    def quests(self) -> pd.DataFrame:
        return self._read("quests").set_index("id", drop=False)

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    def npc_name(self, npc_id: int) -> str:
        return str(self.npcs.at[npc_id, "name"]) if npc_id in self.npcs.index else f"npc {npc_id}"

    def item_name(self, item_id: int) -> str:
        return str(self.items.at[item_id, "name"]) if item_id in self.items.index else f"item {item_id}"

    def item_price(self, item_id: int) -> int:
        """The item's reference price: a merchant buys it for half of it."""
        if item_id not in self.items.index:
            return 0
        price = self.items.at[item_id, "price"]
        return 0 if pd.isna(price) else int(price)

    # ------------------------------------------------------------------
    # Places
    # ------------------------------------------------------------------

    def spawn_points(self, npc_id: int, area: Area | None = None) -> list[Point]:
        """Where ``npc_id`` spawns (the centre of a territory spawn), optionally inside ``area``."""
        rows = self.spawns[self.spawns["npc_id"] == npc_id]
        points = [Point(int(r.cx), int(r.cy), int(r.cz)) for r in rows.itertuples()]
        return [p for p in points if area is None or area.contains(p.x, p.y)]

    def nearest_spawn(self, npc_id: int, x: float, y: float, area: Area | None = None) -> Point | None:
        points = self.spawn_points(npc_id, area)
        return min(points, key=lambda p: p.distance(x, y)) if points else None

    def npcs_in(self, area: Area, types: tuple[str, ...] | None = None) -> pd.DataFrame:
        """The spawns of ``area`` joined with their NPC templates, optionally only some NPC ``types``."""
        spawns = self.spawns[(self.spawns.cx.between(area.x_min, area.x_max)) & (self.spawns.cy.between(area.y_min, area.y_max))]
        joined = spawns.merge(self.npcs[["name", "type", "level", "is_aggressive"]], left_on="npc_id",
                              right_index=True, how="left")
        return joined if types is None else joined[joined["type"].isin(types)]

    def hunting_grounds(self, area: Area, level: int, spread: int = 2) -> pd.DataFrame:
        """
        Monster spawns of ``area`` whose level is within ``spread`` of ``level`` (passive ones first), with how many
        monsters each holds: the places to hunt at that level.
        """
        mobs = self.npcs_in(area, ("Monster",))
        mobs = mobs[(mobs["level"] - level).abs() <= spread]
        return mobs.sort_values(["is_aggressive", "count"], ascending=[True, False])

    # ------------------------------------------------------------------
    # Trade and quests
    # ------------------------------------------------------------------

    def buylists_of(self, npc_id: int) -> pd.DataFrame:
        return self.buylists[self.buylists["npc_ids"].apply(lambda ids: npc_id in set(ids))]

    def sellers_of(self, item_id: int, area: Area | None = None) -> list[int]:
        """Merchants (NPC ids) whose lists hold ``item_id``, those with a spawn in ``area`` when given."""
        lists = self.buylists[self.buylists["item_id"] == item_id]
        npc_ids = sorted({int(n) for ids in lists["npc_ids"] for n in ids})
        return [n for n in npc_ids if area is None or self.spawn_points(n, area)]

    def quests_starting_in(self, area: Area, level: int) -> pd.DataFrame:
        """
        Quests whose start NPC stands in ``area`` and whose minimum level is reached. Race and class limits are not
        reliable in the extracted data (a race check may also pick a reward), so the start dialog decides them.
        """
        npc_ids = set(self.npcs_in(area)["npc_id"])
        quests = self.quests[self.quests["start_npcs"].apply(lambda ids: bool(npc_ids & set(ids)))]
        return quests[quests["min_level"].fillna(1) <= level]

    def quest_html(self, quest_id: int, file: str) -> str | None:
        for entry in self.quests.at[quest_id, "htmls"]:
            if entry["file"] == file:
                return entry["html"]
        return None


def _spawn_centre(row) -> tuple[float, float, float]:
    if not pd.isna(row["x"]):
        return float(row["x"]), float(row["y"]), float(row["z"])
    nodes = [n for t in row["territories"] for n in t["nodes"]]
    if not nodes:
        return math.nan, math.nan, math.nan
    zs = [t["min_z"] for t in row["territories"] if t["min_z"] is not None] + \
         [t["max_z"] for t in row["territories"] if t["max_z"] is not None]
    return (sum(n[0] for n in nodes) / len(nodes), sum(n[1] for n in nodes) / len(nodes),
            sum(zs) / len(zs) if zs else math.nan)

