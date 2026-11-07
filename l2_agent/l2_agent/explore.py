"""
Exploring an area: the agent walks to every NPC of the knowledge in the area (not the monsters), talks to it,
opens its shop lists, and writes what it found next to what the knowledge says: whether the NPC was there and
reachable, its first dialog and links, and the live prices of its lists against the buylist table.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

from .dialog import Talker, html_text
from .knowledge import Area
from .session import Session
from .trade import Trader

log = logging.getLogger(__name__)

#: NPC types worth a visit: the ones a player talks to
TALK_TYPES = ("Folk", "Merchant", "Teleporter", "Warehouse", "Trainer", "VillageMasterFighter", "VillageMasterPriest",
              "VillageMasterMystic", "Guard", "Adventurer", "Fisherman", "OlympiadManager")


def _save(visits: list[dict], shops: list[dict], out_dir: Path, area: Area) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"\W+", "_", area.name.lower())
    npc_path, shop_path = out_dir / f"explore_{tag}_npcs.parquet", out_dir / f"explore_{tag}_shops.parquet"
    pd.DataFrame(visits).to_parquet(npc_path, index=False)
    pd.DataFrame(shops).to_parquet(shop_path, index=False)
    return npc_path, shop_path


def explore(session: Session, talker: Talker, trader: Trader, area: Area, out_dir: Path) -> tuple[Path, Path]:
    """Visit the NPCs of ``area`` nearest first; write ``explore_npcs`` and ``explore_shops`` Parquet files."""
    k = session.knowledge
    npcs = k.npcs_in(area, TALK_TYPES).drop_duplicates("npc_id")
    todo = {int(r.npc_id): (r.cx, r.cy) for r in npcs.itertuples()}
    visits, shops = [], []
    while todo:
        x, y, _ = session.position
        npc_id = min(todo, key=lambda n: (todo[n][0] - x) ** 2 + (todo[n][1] - y) ** 2)
        todo.pop(npc_id)
        started = time.time()
        dialog = talker.talk(npc_id, timeout=300, attempts=2)
        live = session.find_npc(npc_id)
        visit = {
            "npc_id": npc_id, "name": k.npc_name(npc_id), "type": str(k.npcs.at[npc_id, "type"]),
            "seen": live is not None, "talked": dialog is not None,
            "x": live.x if live else None, "y": live.y if live else None, "z": live.z if live else None,
            "seconds": round(time.time() - started, 1),
            "text": html_text(dialog.html)[:2000] if dialog else None,
            "links": [{"kind": lk.kind, "command": re.sub(r"npc_\d+_", "npc_<id>_", lk.command), "text": lk.text}
                      for lk in dialog.links] if dialog else [],
        }
        visits.append(visit)
        _save(visits, shops, out_dir, area)  # after every NPC: an interrupted exploration keeps what it found
        log.info("Explored %s: seen %s, talked %s (%.0fs)", visit["name"], visit["seen"], visit["talked"],
                 visit["seconds"])
        if dialog is None:
            continue
        for lk in dialog.links:
            m = re.search(r"_Buy (\d+)$", lk.command)
            if not m:
                continue
            shop = trader.open_list(npc_id, int(m[1]))
            known = k.buylists[(k.buylists["list_id"] == int(m[1]))].set_index("item_id")
            for item in (shop or {}).get("items", []):
                listed = known["price"].get(item["item_id"]) if item["item_id"] in known.index else None
                shops.append({"npc_id": npc_id, "list_id": int(m[1]), "item_id": item["item_id"],
                              "name": k.item_name(item["item_id"]), "live_price": item["price"],
                              "base_price": listed if listed is not None and not pd.isna(listed)
                              else k.item_price(item["item_id"]),
                              "in_knowledge": item["item_id"] in known.index})
    npc_path, shop_path = _save(visits, shops, out_dir, area)
    stuck = talker.nav.stuck_spots
    log.info("Explored %d NPCs (%d talked to), %d shop items, %d stuck spot(s): %s", len(visits),
             sum(v["talked"] for v in visits), len(shops), len(stuck), stuck)
    return npc_path, shop_path
