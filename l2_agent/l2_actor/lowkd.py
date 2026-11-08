"""
Why the characters with the lowest K/D die: their deaths split into kinds, against the rest of the fleet.

Reads the fight summaries of a fleet run (``episodes_<account>.jsonl``) and, when given, the game server's log
(``java0.log``: which farm area RLFarmAreas assigned each character, when RLAgentManager moved it). A death fight is
sorted into one kind (the first that fits):

* ``weak start`` - the character took less than half of its HP in the fight that killed it: it went in hurt;
* ``mobbed``     - two or more monsters attacked it;
* ``fast``       - dead within 6 s (a monster far above it, or an aggressive pack);
* ``outfought``  - a long one-on-one fight it lost (dealt under 60 % of a monster's HP);
* ``even``       - the rest.

Usage (from ``l2_agent/``)::

    python -m l2_actor.lowkd checkpoints/rl_bc_v11n/fleet_run2 --log checkpoints/rl_bc_v11n/server_now.log
    python -m l2_actor.lowkd checkpoints/rl_bc_v11n/fleet_run2 --worst 20 --min-deaths 15 --since-minutes 60
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import re
import statistics
import time
from datetime import datetime, timezone

#: reward weights of the damage terms (l2_actor.reward): the terms divided by them are shares of HP
W_TAKEN, W_DEALT = -0.5, 0.3
KINDS = ("weak start", "mobbed", "fast", "outfought", "even")
_ASSIGN = re.compile(r"RLFarmAreas: (\S+) \((\w+), level (\d+)\) assigned to (\S+?)\.?$")
_MOVED = re.compile(r"RLAgentManager: (\S+) (?:\(.*?\) )?(died \d+ times without a kill|has kills/deaths \S+ in its farm "
                    r"area|gained no experience|made no progress)")


def death_kind(e: dict) -> str:
    taken = taken_share(e)
    dealt = dealt_share(e)
    if taken < 0.5:
        return "weak start"
    if e.get("attackers", 1) >= 2:
        return "mobbed"
    if e["duration_s"] < 6:
        return "fast"
    if dealt < 0.6:
        return "outfought"
    return "even"


def taken_share(e: dict) -> float:
    """Share of its own max HP the character lost in the fight."""
    return e["reward_terms"]["damage_taken"] / W_TAKEN


def dealt_share(e: dict) -> float:
    """Damage it dealt, in monsters' HP."""
    return e["reward_terms"]["damage_dealt"] / W_DEALT


def server_log(path: str | None):
    """Character -> [(time, area)] of its farm area assignments, and -> Counter of the moves RLAgentManager made."""
    assign: dict[str, list] = collections.defaultdict(list)
    moves: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    if not path:
        return assign, moves
    for line in open(path, errors="replace"):
        line = line.rstrip()
        m = _ASSIGN.search(line)
        if m:
            t = datetime.strptime(line[:19], "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            assign[m[1]].append((t, m[4]))
            continue
        m = _MOVED.search(line)
        if m:
            moves[m[1]][m[2].split(" ")[0] if not m[2].startswith("died") else "died w/o kill"] += 1
    return assign, moves


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="the fleet run's output directory")
    p.add_argument("--log", help="the game server's java0.log of the run")
    p.add_argument("--worst", type=int, default=15)
    p.add_argument("--min-deaths", type=int, default=10)
    p.add_argument("--since-minutes", type=float, default=0.0, help="only the fights of the last N minutes")
    args = p.parse_args(argv)

    assign, moves = server_log(args.log)
    times = {c: [t for t, _ in v] for c, v in assign.items()}
    since = time.time() - args.since_minutes * 60 if args.since_minutes else 0.0
    per: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    info: dict[str, tuple] = {}
    kinds: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    deaths: dict[str, list] = collections.defaultdict(list)
    areas: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for f in glob.glob(f"{args.run}/episodes_*.jsonl"):
        for line in open(f):
            e = json.loads(line)
            if e.get("truncated") or e.get("t", 0) < since:
                continue
            c, r = e["character"], e["reason"]
            per[c][r] += 1
            info[c] = (e["class_id"], e["level"], tuple(e["panel"].get("equipment") or ()))
            i = bisect.bisect_right(times.get(c, []), e["t"]) - 1
            area = assign[c][i][1] if i >= 0 else "?"
            areas[c][(area, r)] += 1
            if r == "death":
                kinds[c][death_kind(e)] += 1
                deaths[c].append(e)

    def kd(c: str) -> float:
        return per[c]["kill"] / max(per[c]["death"], 1)

    rated = [c for c in per if per[c]["death"] >= args.min_deaths]
    total_k = sum(per[c]["kill"] for c in per)
    total_d = sum(per[c]["death"] for c in per)
    print(f"{len(per)} characters, K/D {total_k}/{total_d} = {total_k / max(total_d, 1):.1f}; "
          f"{len(rated)} with at least {args.min_deaths} deaths")
    if not rated:
        return 0
    worst = sorted(rated, key=kd)[: args.worst]
    rest = [c for c in rated if c not in worst]

    def mix(chars) -> str:
        n = collections.Counter()
        for c in chars:
            n.update(kinds[c])
        s = sum(n.values()) or 1
        return "  ".join(f"{k} {100 * n[k] / s:.0f}%" for k in KINDS)

    print(f"\ndeath kinds, worst {len(worst)}: {mix(worst)}")
    print(f"death kinds, the rest:      {mix(rest)}")
    print("\nname          cls lvl   K/D  kills/deaths | death fights: s, taken, dealt, attackers, mp | kinds | areas (k/d) | moves")
    for c in worst:
        ds = deaths[c]
        med = lambda xs: statistics.median(xs) if xs else 0  # noqa: E731
        cls, lvl, eq = info[c]
        ar = collections.defaultdict(lambda: [0, 0])
        for (a, r), n in areas[c].items():
            ar[a][0 if r == "kill" else 1] += n
        top = sorted(ar.items(), key=lambda kv: -kv[1][1])[:3]
        print(f"{c:13s} {cls:3d} {lvl:3d} {kd(c):5.1f} {per[c]['kill']:5d}/{per[c]['death']:<4d}| "
              f"{med([d['duration_s'] for d in ds]):5.1f}s {med([taken_share(d) for d in ds]):4.2f} "
              f"{med([dealt_share(d) for d in ds]):4.2f} {med([d.get('attackers', 1) for d in ds]):3.0f} "
              f"{med([d.get('mp_end') or 0 for d in ds]):4.2f} | "
              + " ".join(f"{k.split()[0]}:{kinds[c][k]}" for k in KINDS if kinds[c][k])
              + " | " + " ".join(f"{a}({k}/{d})" for a, (k, d) in top)
              + " | " + " ".join(f"{k}:{n}" for k, n in moves[c].items()) + f" {eq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
