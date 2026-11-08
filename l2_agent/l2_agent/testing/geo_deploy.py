"""
Put the repaired geodata on the running server: the regions of ``geodata_work/fixed`` that differ from the server's
are backed up there (``~/geodata_backup/<time>/``), copied over and reloaded live with ``//geomap_reload`` by a test
character that is made master GM for the moment; then the agent's Parquet copy is brought up to date
(``geodata2parquet.py --update``), so the agent plans on what the server checks.

Usage (from ``l2_agent/``)::

    python -m l2_agent.testing.geo_deploy [--dry-run] [--account aiagent02 --name GeoTestA]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

from l2_client import packets
from l2_client.packets import GameServerPacket as G

from ..session import Session, SessionConfig
from . import server

ROOT = Path(__file__).resolve().parents[3]
FIXED = ROOT / "geodata_work" / "fixed"
STAZIS = ROOT / "geodata_work" / "stazis"
REMOTE = "server/game/data/geodata"
MASTER = 100


def changed_regions() -> list[str]:
    local = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(FIXED.glob("*.l2j"))}
    remote = {}
    for line in server.run(f"cd {REMOTE} && md5sum {' '.join(local)}").splitlines():
        digest, name = line.split()
        remote[name] = digest
    return [name for name, digest in local.items() if remote.get(name) != digest]


def reload_live(regions: list[str], account: str, name: str, model: str) -> list[str]:
    """Reload ``regions`` on the server through the test character; the server's answers."""
    server.set_access(name, MASTER)
    try:
        s = Session(SessionConfig(account=account, password=account + "pw", model=model))
        s.connect()
        said: list[str] = []
        s.listeners.append(lambda b, _now: said.append(str(packets.parse_creature_say(b)))
                           if b[0] == G.CREATURE_SAY else None)
        for region in regions:
            s.send(packets.build_bypass_build_cmd(f"geomap_reload {region}"))
            s.idle(3.0)
        s.close()
    finally:
        time.sleep(2.0)
        server.set_access(name, 0)
    return [m for m in said if "GeoMap" in m]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account", default="aiagent02")
    p.add_argument("--name", default="GeoTestA")
    p.add_argument("--model", default="checkpoints/agent/combat_policy.onnx")
    p.add_argument("--dry-run", action="store_true", help="only list the regions that differ")
    args = p.parse_args(argv)
    regions = changed_regions()
    print("regions to deploy:", " ".join(regions) or "none")
    if args.dry_run or not regions:
        return 0
    stamp = time.strftime("%Y%m%d%H%M")
    server.run(f"mkdir -p ~/geodata_backup/{stamp} && cd {REMOTE} && cp {' '.join(regions)} ~/geodata_backup/{stamp}/")
    for region in regions:
        server.copy_to(str(FIXED / region), f"{REMOTE}/{region}")
    left = changed_regions()
    if left:
        print("not copied:", left)
        return 1
    for answer in reload_live([r.removesuffix(".l2j") for r in regions], args.account, args.name, args.model):
        print(answer)
    subprocess.run([sys.executable, str(ROOT / "l2_agent" / "l2_NN" / "tools" / "geodata2parquet.py"), "-i", str(STAZIS),
                    str(FIXED), "-o", str(ROOT / "l2jc6_geodata.parquet"), "--update"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
