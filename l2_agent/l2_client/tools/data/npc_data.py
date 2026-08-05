"""
Lineage2 C6 Interlude -- NPC name lookup from server XML datapack.

The MIT License (MIT)

Copyright (c) 2026 Viacheslav Reutskyi <viacheslav.reutskyi@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re

# Path to the npc stats XML files relative to the project root
DEFAULT_NPC_XML_PATH = "../L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/npcs"

_npc_names: dict[int, str] = {}
_npc_levels: dict[int, int] = {}
_loaded = False


def load_npc_names(path: str = DEFAULT_NPC_XML_PATH) -> dict[int, str]:
    """
    Parse all NPC XML files and return a {npc_id: name} lookup table.

    Also populates the internal ``_npc_levels`` dict with the level
    attribute (e.g. ``<npc id="20120" level="4" …>``).

    :param path: directory containing ``20000-20099.xml`` etc.
    :returns: dict mapping npc template id -> name
    """
    global _npc_names, _npc_levels, _loaded
    if _loaded:
        return _npc_names

    if not os.path.isdir(path):
        print(f"[NPC_DATA]  !! Directory not found: {path}")
        _loaded = True
        return _npc_names

    pattern = re.compile(r'<npc\s+id="(\d+)"\s+level="(\d+)"[^>]*\sname="([^"]+)"')
    count = 0
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".xml"):
            continue
        fpath = os.path.join(path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception:
            continue
        for m in pattern.finditer(text):
            npc_id = int(m.group(1))
            level = int(m.group(2))
            name = m.group(3)
            _npc_names[npc_id] = name
            _npc_levels[npc_id] = level
            count += 1

    _loaded = True
    print(f"[NPC_DATA]  Loaded {count} NPC entries (name + level) from {path}")
    return _npc_names


def get_npc_name(template_id: int) -> str:
    """Return the human-readable name for an NPC template ID, or ``""``."""
    return _npc_names.get(template_id, "")


def get_npc_level(template_id: int) -> int:
    """Return the level for an NPC template ID, or 0 if unknown."""
    return _npc_levels.get(template_id, 0)
