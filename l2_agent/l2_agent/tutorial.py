"""
The newbie tutorial (quest 255) talks to the client with its own packets. The agent goes through it as briefly as
a player who knows the game: it leaves each window by its "Exit" link, clicks the question marks it is shown and
reports the client events it is asked for at once (the tutorial waits for them, e.g. "the player moved").
"""

from __future__ import annotations

import logging
import time

from l2_client import packets
from l2_client.packets import GameServerPacket as G

from .session import Session, parse_links

log = logging.getLogger(__name__)


class Tutorial:
    """Answers the tutorial windows, question marks and client events."""

    #: a window is answered at most this often (a window that comes back at once would loop)
    ANSWER_INTERVAL_S = 1.0

    def __init__(self, session: Session):
        self.session = session
        self._answered_at = 0.0
        self.windows = 0
        session.listeners.append(self.on_packet)

    def on_packet(self, body: bytes, now: float) -> None:
        pid = body[0]
        if pid == G.TUTORIAL_SHOW_HTML:
            self.windows += 1
            if now - self._answered_at < self.ANSWER_INTERVAL_S:
                return
            self._answered_at = now
            links = parse_links(packets.parse_tutorial_show_html(body)["html"])
            if not links:
                self.session.send(packets.build_tutorial_pass_cmd("close"))
                return
            link = next((lk for lk in links if "exit" in lk.text.lower()), links[0])
            log.info("Tutorial window: %s (%s)", link.text, link.command)
            build = packets.build_tutorial_link_html if link.kind == "link" else packets.build_tutorial_pass_cmd
            self.session.send(build(link.command))
        elif pid == G.TUTORIAL_SHOW_QUESTION_MARK:
            mark = packets.parse_tutorial_int(body).get("id", 0)
            log.info("Tutorial question mark %d", mark)
            self.session.send(packets.build_tutorial_question_mark(mark))
        elif pid == G.TUTORIAL_ENABLE_CLIENT_EVENT:
            event = packets.parse_tutorial_int(body).get("id", 0)
            if event:
                self.session.send(packets.build_tutorial_client_event(event))

    def settle(self, seconds: float = 3.0) -> None:
        """Let the first windows of a fresh character come and go."""
        end = time.time() + seconds
        while time.time() < end:
            self.session.pump(0.2)
