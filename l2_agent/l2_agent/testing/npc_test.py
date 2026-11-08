"""
Can a character get the dialog of each NPC of a task list? For each task the test character is put at the task's
start point in the database (offline, General GM for ``//gmspeed 4``), logs in and either

* ``--mode click``: only clicks the NPC, so the server finds the whole way (a check of the server's geodata), or
* ``--mode talk``: walks its own geodata route next to the NPC and clicks there, as the agent does.

The task CSV comes from ``geo.reach`` (columns spawn_file, npc_id, x, y, z, start_x, start_y, start_z); each result
row is appended to ``--out``: spawn_file, npc_id, x, y, z, result (dialog / no_dialog / not_in_sight), seconds,
distance left, position. Several accounts can run at once on parts of a list.

Usage (from ``l2_agent/``)::

    python -m l2_agent.testing.npc_test --account aiagent02 --name GeoTestA --tasks tasks.csv --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time

from l2_client import packets

from ..dialog import Talker
from ..geomap import GeoMap
from ..navigation import Navigator
from ..session import Session, SessionConfig
from . import server

GM_ACCESS = 30  # General GM: //gmspeed, no aggro


def session(account: str, name: str, model: str) -> Session:
    create = dict(name=name, race=0, female=False, class_id=0, hair_style=1, hair_color=1, face=0)
    s = Session(SessionConfig(account=account, password=account + "pw", create=create, model=model))
    s.connect()
    return s


def click(s: Session, npc, timeout: float = 45.0, attempts: int = 2) -> bool:
    for _ in range(attempts):
        x, y, z = s.position
        clicked = time.time()
        s.send(packets.build_action(npc.object_id, x, y, z))
        if not s.wait_until(lambda: s.dialog is not None and s.dialog.received >= clicked, 0.8):
            s.send(packets.build_action(npc.object_id, x, y, z))
        if s.wait_until(lambda: s.dialog is not None and s.dialog.received >= clicked, timeout):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account", required=True, help="test account (password: <account>pw)")
    p.add_argument("--name", required=True, help="its character, created on the first login")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("click", "talk"), default="talk")
    p.add_argument("--speed", type=int, default=4, help="//gmspeed of the test character (0 = none)")
    p.add_argument("--model", default="checkpoints/agent/combat_policy.onnx")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format=f"%(asctime)s {args.account} %(message)s", datefmt="%H:%M:%S")
    geomap = GeoMap() if args.mode == "talk" else None
    session(args.account, args.name, args.model).close()  # makes sure the character exists
    time.sleep(2.0)
    with open(args.out, "a", newline="") as f:
        out = csv.writer(f)
        for t in csv.DictReader(open(args.tasks)):
            npc_id = int(t["npc_id"])
            start = int(float(t["start_x"])), int(float(t["start_y"])), int(float(t["start_z"]))
            if not server.set_position(args.name, *start, access_level=GM_ACCESS):
                time.sleep(3.0)  # the logout may not be saved yet
                if not server.set_position(args.name, *start, access_level=GM_ACCESS):
                    out.writerow([t["spawn_file"], npc_id, t["x"], t["y"], t["z"], "db_update_failed", "", "", ""])
                    continue
            s = session(args.account, args.name, args.model)
            try:
                if args.speed:
                    s.send(packets.build_bypass_build_cmd(f"gmspeed {args.speed}"))
                s.idle(2.0)
                if s.find_npc(npc_id) is None:
                    s.wait_until(lambda: s.find_npc(npc_id) is not None, 5.0)
                npc = s.find_npc(npc_id)
                if npc is None:
                    result, secs, dist = "not_in_sight", "", ""
                else:
                    asked = time.time()
                    if args.mode == "click":
                        ok = click(s, npc)
                    else:
                        ok = Talker(s, Navigator(s, geomap=geomap)).talk(npc_id, timeout=120, attempts=2) is not None
                    result, secs, dist = ("dialog" if ok else "no_dialog"), round(time.time() - asked, 1), \
                        round(s.distance(npc.x, npc.y))
                out.writerow([t["spawn_file"], npc_id, t["x"], t["y"], t["z"], result, secs, dist, s.position])
                f.flush()
                print(args.account, npc_id, t["spawn_file"], result, secs, dist, flush=True)
            finally:
                s.close()
                time.sleep(2.0)  # the server saves the position on logout; the next update comes after it
    server.set_access(args.name, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
