"""
Movement topology of a geodata area as the server walks it: from a layer, a step in a direction is allowed
when the layer's NSWE has that direction open, and lands on the neighbour's layer nearest in height (a step
up of more than 40 is refused). Finds one-way steps and pockets one can enter but not leave.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import l2j
from .repair import DIRECTIONS

STEP_UP_LIMIT = 40


def graph(region: l2j.Region, box: tuple[int, int, int, int], band: tuple[int, int]):
    """Directed moves between (cx, cy, layer) nodes of the box (region cells) whose layer is in the z band."""
    cx0, cy0, cx1, cy1 = box
    edges: dict[tuple, list[tuple]] = defaultdict(list)
    nodes = []
    for cx in range(cx0, cx1):
        for cy in range(cy0, cy1):
            k = int(region.layers[cx, cy])
            for li in range(k):
                h = int(region.height[cx, cy, li])
                if not band[0] <= h <= band[1]:
                    continue
                node = (cx, cy, li)
                nodes.append(node)
                ns = int(region.nswe[cx, cy, li])
                for bit, dx, dy, _opp in DIRECTIONS:
                    if not ns & bit:
                        continue
                    a, b = cx + dx, cy + dy
                    if not (cx0 <= a < cx1 and cy0 <= b < cy1):
                        continue
                    n = int(region.layers[a, b])
                    if not n:
                        continue
                    hs = region.height[a, b, :n]
                    m = int(np.argmin(np.abs(hs - h)))
                    if int(hs[m]) - h > STEP_UP_LIMIT or not band[0] <= int(hs[m]) <= band[1]:
                        continue
                    edges[node].append((a, b, m))
    return nodes, edges


def one_way(nodes, edges) -> list[tuple]:
    """Steps A -> B where B cannot step back to A."""
    return [(u, v) for u in nodes for v in edges.get(u, []) if u not in edges.get(v, [])]


def traps(nodes, edges, max_size: int = 400) -> list[set]:
    """
    Pockets: groups of nodes reachable from outside that cannot reach any node outside the group (strongly
    connected components without an exit), at most ``max_size`` nodes (a large one is the main area).
    """
    index, low, stack, on, comps = {}, {}, [], set(), []
    counter = [0]
    import sys
    sys.setrecursionlimit(1_000_000)

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in edges.get(v, []):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            comps.append(comp)

    for v in nodes:
        if v not in index:
            strong(v)
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    has_exit = [False] * len(comps)
    has_entry = [False] * len(comps)
    for u in nodes:
        for v in edges.get(u, []):
            if comp_of.get(v) is not None and comp_of[u] != comp_of[v]:
                has_exit[comp_of[u]] = True
                has_entry[comp_of[v]] = True
    return [c for i, c in enumerate(comps) if has_entry[i] and not has_exit[i] and len(c) <= max_size]


def close_pockets(region: l2j.Region, box: tuple[int, int, int, int], band: tuple[int, int], rounds: int = 20) -> dict[str, int]:
    """
    Close the steps into every pocket (clear the direction bit on the outside layer), repeating while
    closing them leaves new pockets: a character can then no longer walk into a place it cannot leave.
    """
    closed = pockets_total = 0
    for _ in range(rounds):
        nodes, edges = graph(region, box, band)
        pockets = traps(nodes, edges)
        if not pockets:
            break
        pockets_total += len(pockets)
        inside = {v for p in pockets for v in p}
        for u in nodes:
            if u in inside:
                continue
            for v in edges.get(u, []):
                if v in inside:
                    bit = next(b for b, dx, dy, _o in DIRECTIONS if (u[0] + dx, u[1] + dy) == (v[0], v[1]))
                    region.nswe[u[0], u[1], u[2]] &= ~bit
                    closed += 1
    return {"pockets": pockets_total, "steps_closed": closed}
