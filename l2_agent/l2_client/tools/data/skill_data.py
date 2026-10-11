"""
Lineage2 C6 Interlude -- skill data lookup from server XML datapack.

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
import xml.etree.ElementTree as ET

# Path to skill XML files relative to the project root.
DEFAULT_SKILL_XML_PATH = "../L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/skills"

# Maps skill ID -> name.
_SKILL_NAME: dict[int, str] = {}

# Maps skill ID -> max level count.
_SKILL_LEVELS: dict[int, int] = {}

# Maps skill ID -> reuse delay in milliseconds (level 1 value for per-level tables).
_SKILL_REUSE_MS: dict[int, int] = {}

_loaded = False


def load_skill_data(path: str = DEFAULT_SKILL_XML_PATH) -> None:
    """
    Parse all skill XML files and populate name + level lookup tables.

    :param path: directory containing skill XML files (e.g. ``00100-00199.xml``).
    """
    global _loaded, _SKILL_NAME, _SKILL_LEVELS, _SKILL_REUSE_MS
    if _loaded:
        return

    if not os.path.isdir(path):
        print(f"[SKILL_DATA] !! Directory not found: {path}")
        _loaded = True
        return

    count = 0
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".xml"):
            continue
        fpath = os.path.join(path, fname)
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
        except Exception:
            continue

        for skill_elem in root.findall("skill"):
            try:
                skill_id = int(skill_elem.get("id", "0"))
                name = skill_elem.get("name", "")
                levels = int(skill_elem.get("levels", "1"))
            except (ValueError, TypeError):
                continue

            _SKILL_NAME[skill_id] = name
            _SKILL_LEVELS[skill_id] = levels
            reuse = _read_reuse_delay(skill_elem)
            if reuse is not None:
                _SKILL_REUSE_MS[skill_id] = reuse
            count += 1

    _loaded = True
    print(f"[SKILL_DATA] Loaded {count} skills from {path}")


def _read_reuse_delay(skill_elem: ET.Element) -> int | None:
    """
    Read ``<reuseDelay>`` of a skill element, in milliseconds.

    The value is either a literal (``13000``) or a reference to a per-level ``<table name="#x">``,
    in which case the level 1 entry is used.
    """
    text = (skill_elem.findtext("reuseDelay") or "").strip()
    if text.startswith("#"):
        table = next((t for t in skill_elem.findall("table") if t.get("name") == text), None)
        text = (table.text or "").split()[0] if table is not None and table.text else ""
    try:
        return int(float(text))
    except ValueError:
        return None


def get_skill_name(skill_id: int) -> str:
    """Return the human-readable name for a skill ID, or ``""`` if unknown."""
    return _SKILL_NAME.get(skill_id, "")


def get_skill_levels(skill_id: int) -> int:
    """Return the max level count for a skill ID, or ``1`` if unknown."""
    return _SKILL_LEVELS.get(skill_id, 1)


def get_skill_reuse_delay(skill_id: int) -> int:
    """Return the reuse delay of a skill in milliseconds, or ``0`` if unknown."""
    return _SKILL_REUSE_MS.get(skill_id, 0)
