"""
A fleet of autonomous agents, each its own process (``python -m l2_agent``) with its own account and character:
half Human Fighters, half Human Mystics, started a few seconds apart and restarted when one crashes; one that
reached its goal level exits and stays out. ``--status`` prints the fleet's report from the agents' status files.

Usage (from ``l2_agent/``)::

    python -m l2_agent.fleet --count 16                 # accounts aifleet01..16, logs in checkpoints/agent/fleet/
    python -m l2_agent.fleet --status                   # level, step, adena, deaths, last update of each agent
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path("checkpoints/agent")
FLEET_DIR = STATE_DIR / "fleet"
CLASSES = {0: "Fighter", 10: "Mystic"}


def accounts(count: int, prefix: str) -> list[tuple[str, str, int]]:
    """(account, character name, class id) of the fleet: even indexes fighters, odd ones mystics."""
    out = []
    for i in range(1, count + 1):
        class_id = 0 if i % 2 else 10
        out.append((f"{prefix}{i:02d}", f"Ai{CLASSES[class_id]}{i:02d}", class_id))
    return out


def start(account: str, name: str, class_id: int, extra: list[str]) -> subprocess.Popen:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    log = open(FLEET_DIR / f"{account}.log", "a")
    cmd = [sys.executable, "-m", "l2_agent", "--account", account, "--password", account + "pw", "--name", name,
           "--class-id", str(class_id), *extra]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def run(args) -> int:
    fleet = accounts(args.count, args.prefix)
    procs: dict[str, subprocess.Popen] = {}
    restarts: dict[str, int] = {}
    for account, name, class_id in fleet:
        procs[account] = start(account, name, class_id, args.extra)
        print(f"started {account} {name} ({CLASSES[class_id]})", flush=True)
        time.sleep(args.stagger)
    try:
        while procs:
            time.sleep(10)
            for account, name, class_id in fleet:
                proc = procs.get(account)
                if proc is None or proc.poll() is None:
                    continue
                status = _status(account)
                if proc.returncode == 0 and status and status.get("level", 0) >= args.goal:
                    print(f"{account} reached level {status['level']}: done", flush=True)
                    del procs[account]
                    continue
                if proc.returncode == -signal.SIGTERM:  # stopped on purpose (to load new code): straight back
                    print(f"{account} stopped: restart", flush=True)
                    procs[account] = start(account, name, class_id, args.extra)
                    continue
                restarts[account] = restarts.get(account, 0) + 1
                if restarts[account] > args.max_restarts:
                    print(f"{account} exited {proc.returncode} too often: left out", flush=True)
                    del procs[account]
                    continue
                print(f"{account} exited {proc.returncode}: restart {restarts[account]}", flush=True)
                time.sleep(args.restart_delay)
                procs[account] = start(account, name, class_id, args.extra)
    except KeyboardInterrupt:
        for proc in procs.values():
            proc.terminate()
    return 0


def _status(account: str) -> dict | None:
    path = STATE_DIR / "status" / f"{account}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def report(prefix: str) -> int:
    now = time.time()
    rows = []
    for path in sorted((STATE_DIR / "status").glob(f"{prefix}*.json")):
        st = json.loads(path.read_text())
        hours = (now - st.get("started", now)) / 3600
        rows.append((st["account"], st.get("name", ""), CLASSES.get(st.get("class_id"), st.get("class_id")),
                     st.get("level"), st.get("area"), "/".join(str(x) for x in st.get("step", [])), st.get("result", ""),
                     st.get("adena"), st.get("deaths"), len(st.get("done_quests", [])), st.get("skills"),
                     f"{hours:.1f}h", f"{now - st.get('time', now):.0f}s"))
    head = ("account", "name", "class", "lvl", "area", "step", "result", "adena", "deaths", "quests", "skills",
            "playing", "updated")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    if rows:
        levels = [r[3] or 0 for r in rows]
        print(f"\n{len(rows)} agents, levels {min(levels)}-{max(levels)}, mean {sum(levels) / len(levels):.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--prefix", default="aifleet")
    p.add_argument("--goal", type=int, default=20)
    p.add_argument("--stagger", type=float, default=5.0, help="seconds between the starts")
    p.add_argument("--restart-delay", type=float, default=20.0)
    p.add_argument("--max-restarts", type=int, default=20)
    p.add_argument("--status", action="store_true", help="print the fleet's report and exit")
    p.add_argument("extra", nargs="*", help="more arguments for each agent (after --)")
    args = p.parse_args(argv)
    return report(args.prefix) if args.status else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
