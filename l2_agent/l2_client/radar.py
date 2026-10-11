"""
Lineage2 C6 Interlude -- radar class for tracking nearby NPCs.

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

import math
import re
import time

from .tools.data import npc_data


class Radar:
    """
    Tracks nearby NPCs (monsters) within a configurable range.

    Each entry is keyed by NPC ``object_id`` and holds::

        {
            "object_id":  int,
            "npc_id":     int,   # template id (0 if unknown)
            "name":       str,
            "level":      int,
            "heading":    int,   # raw value from NpcInfo packet
            "cur_hp":     int,
            "max_hp":     int,
            "cur_mp":     int,
            "max_mp":     int,
            "x":          int,
            "y":          int,
            "z":          int,
            "dist":       float,  # distance from character
        }
    """

    def __init__(self, range_units: int = 5000):
        self._entries: dict[int, dict] = {}
        self._range = range_units
        self._last_print = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entries(self) -> dict[int, dict]:
        """
        Return a reference to the internal entries dict.

        Entries are keyed by NPC ``object_id``.  Each value is a dict
        with the shape described in the class docstring.

        :rtype: dict[int, dict]
        """
        return self._entries

    @property
    def range_units(self) -> int:
        """
        The detection radius in game units (default 5000).

        NPCs whose 2-D distance from the character exceeds this value
        are pruned via :meth:`prune`.

        :rtype: int
        """
        return self._range

    @property
    def count(self) -> int:
        """
        Return the number of NPCs currently tracked.

        :rtype: int
        """
        return len(self._entries)

    # ------------------------------------------------------------------
    # Data mutators
    # ------------------------------------------------------------------

    def add_or_update(self, npc: dict):
        """
        Add or update a radar entry from a parsed NpcInfo (0x16) dict.

        Skipped when the NPC is not attackable or is dead
        (``is_alike_dead``).  The human-readable *name* and *level* are
        resolved via :meth:`_resolve_name` and :meth:`_resolve_level`,
        which fall back to the XML datapack loaded by
        :mod:`npc_data`.

        :param npc: dict returned by :func:`packets.parse_npc_info`
        """
        if not npc.get("is_attackable") or npc.get("is_alike_dead"):
            return

        name = self._resolve_name(npc)
        level = self._resolve_level(npc)

        did = npc.get("display_id", 0)
        self._entries[npc["object_id"]] = {
            "object_id": npc["object_id"],
            "npc_id": did - 1000000 if did > 1000000 else did,
            "name": name,
            "level": level,
            "heading": npc.get("heading", 0),
            "cur_hp": 0,
            "max_hp": 0,
            "cur_mp": 0,
            "max_mp": 0,
            "x": npc["x"],
            "y": npc["y"],
            "z": npc["z"],
            "dist": 0,
        }

    def update_from_status(self, su: dict):
        """
        Apply HP / MP updates from a StatusUpdate (0x0E) packet.

        Only entries that already exist in the radar are updated.
        Unknown object IDs are silently ignored.

        :param su: dict returned by :func:`packets.parse_status_update`
        """
        oid = su.get("object_id")
        if not oid or oid not in self._entries:
            return
        attrs = su["attributes"]
        if 0x09 in attrs:
            self._entries[oid]["cur_hp"] = attrs[0x09]
        if 0x0A in attrs:
            self._entries[oid]["max_hp"] = attrs[0x0A]
        if 0x0B in attrs:
            self._entries[oid]["cur_mp"] = attrs[0x0B]
        if 0x0C in attrs:
            self._entries[oid]["max_mp"] = attrs[0x0C]

    def remove(self, object_id: int):
        """
        Remove an NPC from the radar by its runtime ``object_id``.

        Safe to call with an ID that is not currently tracked
        -- no error is raised.

        :param object_id: NPC object ID from Die / DeleteObject packets
        """
        self._entries.pop(object_id, None)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(self, char_x: int, char_y: int):
        """
        Recalculate distances and remove entries beyond :attr:`range_units`.

        Should be called once per game-loop tick (after position updates)
        to keep the radar bounded.

        :param char_x: character's current X coordinate
        :param char_y: character's current Y coordinate
        """
        for oid in list(self._entries.keys()):
            entry = self._entries[oid]
            dx = char_x - entry["x"]
            dy = char_y - entry["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self._range:
                del self._entries[oid]
            else:
                entry["dist"] = dist

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_if_due(self, char_x: int, char_y: int, char_z: int,
                     interval: float = 1.0) -> bool:
        """
        Print a formatted radar table once every *interval* seconds.

        The table includes::

            Name                Lv   Head         HP         MP   Dist objId

        HP/MP are shown as ``cur/max`` when *max* is known, otherwise
        ``?/?``.  Rows are sorted by distance (nearest first).

        :param char_x: character X
        :param char_y: character Y
        :param char_z: character Z (only used in the header line)
        :param interval: minimum seconds between prints (default 1.0)
        :returns: ``True`` if the table was printed, ``False`` otherwise
        :rtype: bool
        """
        now = time.time()
        if now - self._last_print < interval:
            return False
        self._last_print = now

        if not self._entries:
            print("[RADAR]  (empty)")
            return True

        print(f"\n[RADAR]  {len(self._entries)} NPCs within {self._range} units "
              f"(char pos: {char_x}, {char_y}, {char_z})")
        print(f"  {'Name':<18} {'Lv':>3} {'Head':>6} {'HP':>10} {'MP':>10} {'Dist':>6} {'objId'}")
        print(f"  {'-'*18} {'-'*3} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*10}")

        for e in sorted(self._entries.values(), key=lambda e: e["dist"]):
            hp = f"{e['cur_hp']}/{e['max_hp']}" if e['max_hp'] > 0 else "?/?"
            mp = f"{e['cur_mp']}/{e['max_mp']}" if e['max_mp'] > 0 else "?/?"
            print(f"  {e['name']:<18} {e['level']:>3} {e['heading']:>6}  "
                  f"{hp:>10} {mp:>10} {e['dist']:>5.0f}  {e['object_id']}")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_name(npc: dict) -> str:
        """
        Determine a human-readable name for an NPC.

        Resolution order:

        1. ``npc["name"]`` from the packet (non-empty for named NPCs).
        2. ``npc["title"]`` after stripping ``Lv N``, ``[A]``, ``[G]``
           prefix markers.
        3. XML datapack lookup via :func:`npc_data.get_npc_name`.
        4. Template ID as ``"tpl#NNNN"``.
        5. ``"objId=…"`` as an absolute fallback.

        :param npc: dict returned by :func:`packets.parse_npc_info`
        :returns: display label for the radar table
        :rtype: str
        """
        name = npc["name"].strip() if npc.get("name") else ""
        title = npc.get("title", "").strip()

        if name:
            return name
        if title:
            clean = re.sub(r'^Lv\s+\d+\s*', '', title)
            clean = clean.replace('[A]', '').replace('[G]', '').strip()
            return clean if clean else title

        did = npc.get("display_id", 0)
        real_id = did - 1000000 if did > 1000000 else did
        npc_name = npc_data.get_npc_name(real_id) if real_id else ""
        if npc_name:
            return npc_name
        if real_id > 0:
            return f"tpl#{real_id}"
        return f"objId={npc['object_id']}"

    @staticmethod
    def _resolve_level(npc: dict) -> int:
        """
        Determine an NPC's level, falling back to the XML datapack.

        1. Use ``npc["level"]`` if > 0 (parsed from the title field).
        2. Fall back to :func:`npc_data.get_npc_level`.
        3. Return ``1`` if both are unknown.

        :param npc: dict returned by :func:`packets.parse_npc_info`
        :returns: NPC level (≥ 1)
        :rtype: int
        """
        lvl = npc.get("level", 0)
        if lvl > 0:
            return lvl
        did = npc.get("display_id", 0)
        real_id = did - 1000000 if did > 1000000 else did
        lvl = npc_data.get_npc_level(real_id)
        return lvl if lvl > 0 else 1
