"""
Talking to NPCs: walk up to one (its spawn from the knowledge, then the NPC itself once it is in sight), open its
dialog and follow links. A dialog is the server's HTML; its links are bypasses or links to other HTML files.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Callable

from l2_client import packets

from .knowledge import Area, Point
from .navigation import Navigator
from .session import Dialog, Link, Npc, Session

log = logging.getLogger(__name__)

#: the server answers an Action on an NPC farther than this with a walk to it first
TALK_DISTANCE = 120
#: the server drops a bypass that comes within FloodProtectorServerBypassInterval (300 ms) of the last one
BYPASS_INTERVAL_S = 0.5
#: the agent clicks an NPC from at most this far: the server's Npc.INTERACTION_DISTANCE (250), a little less
CLICK_DISTANCE = 200


def html_text(html: str) -> str:
    """The text of a dialog without tags, for logs and matching."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class Talker:
    """Walks to NPCs, opens their dialogs and follows links."""

    def __init__(self, session: Session, navigator: Navigator, area: Area | None = None):
        self.session = session
        self.nav = navigator
        self.area = area
        #: every dialog seen: (npc id, html) - the exploration report reads it
        self.seen: list[tuple[int, str]] = []
        self._bypass_at = 0.0

    # ------------------------------------------------------------------
    # Reaching an NPC
    # ------------------------------------------------------------------

    def approach(self, npc_id: int, timeout: float = 600.0) -> Npc | None:
        """Walk to the nearest spawn of ``npc_id`` and return the NPC once it is in reach."""
        s = self.session
        npc = s.find_npc(npc_id)
        if npc is None:
            spawn = s.knowledge.nearest_spawn(npc_id, *s.position[:2], area=self.area)
            if spawn is None:
                log.warning("No spawn of %s known", s.knowledge.npc_name(npc_id))
                return None
            log.info("Going to %s at %s", s.knowledge.npc_name(npc_id), spawn)
            self.nav.move_to(spawn, arrive=400, timeout=timeout, interrupt=lambda: s.find_npc(npc_id) is not None
                             and s.distance(s.find_npc(npc_id).x, s.find_npc(npc_id).y) < 900)
            npc = s.find_npc(npc_id)
            if npc is None and s.wait_until(lambda: s.find_npc(npc_id) is not None, 3.0):
                npc = s.find_npc(npc_id)
        if npc is None:
            log.warning("%s is not where the knowledge puts it", s.knowledge.npc_name(npc_id))
        return npc

    def _walk_around(self, npc: Npc) -> bool:
        """
        Walk next to ``npc`` through points around it: the server finds no path to a spot its geodata gets wrong
        (e.g. only the roof of the building the NPC stands in), but to the floor beside it.
        """
        s = self.session
        x, y, _ = s.position
        start = math.atan2(y - npc.y, x - npc.x)  # the side facing the agent first
        for radius in (80, 160):
            for k in range(8):
                angle = start + (k + 1) // 2 * (1 if k % 2 else -1) * math.pi / 4
                spot = Point(int(npc.x + radius * math.cos(angle)), int(npc.y + radius * math.sin(angle)), npc.z)
                self.nav.move_to(spot, arrive=40, timeout=25)
                if s.distance(npc.x, npc.y) <= TALK_DISTANCE:
                    return True
        log.warning("Found no way to %s at %s", s.knowledge.npc_name(npc.npc_id), (npc.x, npc.y, npc.z))
        return False

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def talk(self, npc_id: int, timeout: float = 600.0, attempts: int = 3) -> Dialog | None:
        """Open the first dialog of ``npc_id`` (``attempts`` clicks, each with a walk around it when out of reach)."""
        s = self.session
        npc = self.approach(npc_id, timeout)
        if npc is None:
            return None
        for _ in range(attempts):
            # Within INTERACTION_DISTANCE the server opens the dialog at once; from farther it walks the character to
            # the NPC on its own path, and finds none to a merchant shut in behind his counter. So the agent walks
            # its own route next to the NPC first and only then clicks.
            if s.distance(npc.x, npc.y) > CLICK_DISTANCE:
                self.nav.move_to(Point(npc.x, npc.y, npc.z), arrive=CLICK_DISTANCE - 50, timeout=120)
            asked = time.time()
            x, y, z = s.position
            s.send(packets.build_action(npc.object_id, x, y, z))  # selects the NPC (talks to it when it already is)
            opened = lambda: s.dialog is not None and s.dialog.received >= asked  # noqa: E731
            if not s.wait_until(opened, 0.8):
                s.send(packets.build_action(npc.object_id, x, y, z))  # ... and talks to it
            # A second click on an open dialog would send the page again, and it would pass for the next answer.
            walk = s.distance(npc.x, npc.y) / 80.0
            if s.wait_until(lambda: s.dialog is not None and s.dialog.received >= asked, 4.0 + walk):
                self._note(npc.npc_id)
                return s.dialog
            if s.distance(npc.x, npc.y) > TALK_DISTANCE:
                self._walk_around(npc)
        log.warning("%s does not answer", s.knowledge.npc_name(npc_id))
        return None

    def follow(self, link: Link, timeout: float = 4.0, expect: Callable[[], bool] | None = None) -> Dialog | None:
        """Click ``link``; return the next dialog (``expect`` instead decides what answer to wait for)."""
        s = self.session
        wait = self._bypass_at + BYPASS_INTERVAL_S - time.time()
        if wait > 0:
            s.idle(wait)
        asked = time.time()
        self._bypass_at = asked
        npc_id = self._dialog_npc_id()
        log.info("Link: %s (%s %s)", link.text, link.kind, link.command)
        s.send(packets.build_bypass_to_server(link.command) if link.kind == "bypass" else packets.build_link_html(link.command))
        done = expect or (lambda: s.dialog is not None and s.dialog.received >= asked)
        if not s.wait_until(done, timeout):
            return None
        if s.dialog is not None and s.dialog.received >= asked:
            self._note(npc_id)
        return s.dialog

    def choose(self, dialog: Dialog | None, *patterns: str) -> Link | None:
        """The first link whose text or command matches one of ``patterns`` (regular expressions, case ignored)."""
        if dialog is None:
            return None
        for pattern in patterns:
            for link in dialog.links:
                if re.search(pattern, link.text, re.I) or re.search(pattern, link.command, re.I):
                    return link
        return None

    def _dialog_npc_id(self) -> int:
        s = self.session
        if s.dialog is None:
            return 0
        npc = s.npcs.get(s.dialog.npc_object_id)
        return npc.npc_id if npc else 0

    def _note(self, npc_id: int) -> None:
        dialog = self.session.dialog
        self.seen.append((npc_id, dialog.html))
        log.info("%s: %s", self.session.knowledge.npc_name(npc_id), html_text(dialog.html)[:160].replace("\n", " "))
