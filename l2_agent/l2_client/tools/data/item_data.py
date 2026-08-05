"""
Lineage2 C6 Interlude -- item data lookup from server XML datapack.

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

# Path to item stats XML files relative to the project root
DEFAULT_ITEM_XML_PATH = "../L2J_Mobius/L2J_Mobius_CT_0_Interlude/dist/game/data/stats/items"

# Maps item (type, crystal) to weapon grade: 0=noGrade, 1=D, 2=C, 3=B, 4=A, 5=S
CRYSTAL_GRADE = {
    "none": 0, "d": 1, "c": 2, "b": 3, "a": 4, "s": 5,
}

# Item type -> (grade label when crystal ≠ none)
# For weapons/armor: "none"/"d"/"c"/"b"/"a"/"s"
# For soulshots: the crystal field IS the grade
_ITEM_TYPE_MAP: dict[str, str] = {}

# Item crystal field -> grade
_ITEM_GRADE: dict[int, int] = {}

# Item name
_ITEM_NAME: dict[int, str] = {}

_loaded = False


def load_item_data(path: str = DEFAULT_ITEM_XML_PATH):
    """
    Parse all item XML files and populate grade + name lookup tables.

    :param path: directory containing item XML files
    """
    global _loaded, _ITEM_TYPE_MAP, _ITEM_GRADE, _ITEM_NAME
    if _loaded:
        return

    if not os.path.isdir(path):
        print(f"[ITEM_DATA] !! Directory not found: {path}")
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

        for item_elem in root.findall("item"):
            try:
                item_id = int(item_elem.get("id", "0"))
                name = item_elem.get("name", "")
            except (ValueError, TypeError):
                continue

            _ITEM_NAME[item_id] = name

            # crystal_type is a <set name="crystal_type" val="D"/> child element
            crystal_el = item_elem.find("set[@name='crystal_type']")
            if crystal_el is not None:
                crystal = crystal_el.get("val", "").lower()
                grade = CRYSTAL_GRADE.get(crystal, -1)
                if grade >= 0:
                    _ITEM_GRADE[item_id] = grade
            count += 1

    _loaded = True
    print(f"[ITEM_DATA] Loaded {count} items (name + grade) from {path}")


def get_item_name(item_id: int) -> str:
    """Return the human-readable name for an item ID, or ``""``."""
    return _ITEM_NAME.get(item_id, "")


def get_item_grade(item_id: int) -> int:
    """Return the crystal grade for an item ID (0-5), or -1 if unknown."""
    return _ITEM_GRADE.get(item_id, -1)
