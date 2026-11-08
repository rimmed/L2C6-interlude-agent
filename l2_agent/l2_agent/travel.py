"""
Travel by a gatekeeper: its Teleport list is a dialog page whose links name the destinations of the teleport table
(``l2jc6_teleports``); the fee comes from there too. Arriving is a TeleportToLocation of the character.
"""

from __future__ import annotations

import logging

from .dialog import Talker
from .session import Session

log = logging.getLogger(__name__)


class Traveller:
    """Teleports through gatekeepers."""

    def __init__(self, session: Session, talker: Talker):
        self.session = session
        self.talker = talker

    def fee(self, npc_id: int, destination: str) -> int | None:
        t = self.session.knowledge.teleports
        rows = t[(t["npc_id"] == npc_id) & (t["type"] == "NORMAL") & (t["name"] == destination)]
        return int(rows.iloc[0]["fee_count"]) if len(rows) else None

    def teleport(self, npc_id: int, destination: str) -> bool:
        """Ask gatekeeper ``npc_id`` for ``destination`` (a name of its list); ``True`` once the character arrived."""
        s, t = self.session, self.talker
        fee = self.fee(npc_id, destination)
        if fee is None:
            log.warning("%s does not go to %s", s.knowledge.npc_name(npc_id), destination)
            return False
        if s.adena < fee:
            log.info("Teleport to %s costs %d adena, the character has %d", destination, fee, s.adena)
            return False
        dialog = t.talk(npc_id)
        show = t.choose(dialog, r"_showTeleports$")
        listing = t.follow(show) if show is not None else dialog
        link = t.choose(listing, rf"^{destination}\b", destination.split(",")[0])
        if link is None:
            log.warning("%s's list has no %s: %s", s.knowledge.npc_name(npc_id), destination,
                        [lk.text for lk in (listing.links if listing else [])])
            return False
        teleports = s.world.teleports
        t.follow(link, expect=lambda: s.world.teleports > teleports, timeout=15.0)
        if s.world.teleports > teleports:
            s.idle(2.0)  # the new surroundings arrive
            log.info("Teleported to %s for %d adena", destination, fee)
            return True
        log.warning("The teleport to %s did not happen", destination)
        return False
