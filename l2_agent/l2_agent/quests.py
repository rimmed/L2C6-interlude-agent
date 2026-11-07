"""
Quests through the NPC dialogs. The facts come from the quest table (``l2jc6_quests``): the start and talk NPCs,
the monsters, the quest items and, from the script, which dialog event starts the quest (the event after which the
script calls ``startQuest()``). What to do in which order is scripted per quest in :data:`PLANS`.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import cached_property

from l2_client import packets

from .combat import Fighter
from .dialog import Talker, html_text
from .knowledge import Area, Point
from .session import Dialog, Link, Session

log = logging.getLogger(__name__)

#: links that only close a dialog or go back
_DEAD_ENDS = re.compile(r"\b(close|back|return|quit|cancel|no\b|later)", re.I)


@dataclass
class QuestFacts:
    """The facts of one quest taken from the quest table."""

    id: int
    name: str
    script: str
    start_npcs: list[int]
    talk_npcs: list[int]
    kill_npcs: list[int]
    quest_items: list[int]
    min_level: int
    repeatable: bool
    start_events: list[str]


class QuestBook:
    """The quest table, read through the session's knowledge."""

    def __init__(self, session: Session):
        self.session = session

    def facts(self, quest_id: int) -> QuestFacts:
        q = self.session.knowledge.quests.loc[quest_id]
        source = q["source"]
        events = re.findall(r'(?:equals(?:IgnoreCase)?\("([^"]+)"\)|case\s+"([^"]+)"\s*:)[^{}]*?\{?[^{}]*?startQuest',
                            source, re.S)
        return QuestFacts(int(q["id"]), q["name"], q["script"], [int(x) for x in q["start_npcs"]],
                          [int(x) for x in q["talk_npcs"]], [int(x) for x in q["kill_npcs"]],
                          [int(x) for x in q["quest_items"]], int(q["min_level"]) if q["min_level"] == q["min_level"] else 1,
                          bool(q["repeatable"]), [a or b for a, b in events])


class QuestRunner:
    """Starts and advances quests through dialogs and hunts their items."""

    def __init__(self, session: Session, talker: Talker, fighter: Fighter, area: Area | None = None):
        self.session = session
        self.talker = talker
        self.fighter = fighter
        self.area = area
        self.book = QuestBook(session)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def refresh(self, timeout: float = 3.0) -> dict[int, int]:
        """Ask the server for the quest list and return {quest id: cond} of the started quests."""
        s = self.session
        asked = time.time()
        s.send(packets.build_request_quest_list())
        s.wait_until(lambda: s.quests_at >= asked, timeout)
        return s.quests

    def started(self, quest_id: int) -> bool:
        return quest_id in self.session.quests

    # ------------------------------------------------------------------
    # Dialog steps
    # ------------------------------------------------------------------

    def open(self, npc_id: int, facts: QuestFacts | None = None) -> Dialog | None:
        """Talk to ``npc_id`` and open its quest dialog (of ``facts`` when it offers several)."""
        t = self.talker
        dialog = t.talk(npc_id)
        for _ in range(3):
            if dialog is None:
                return None
            if facts is not None and self._quest_link(dialog, facts) is None and not self._quest_menu(dialog):
                return dialog  # already the quest's own page
            link = (self._quest_link(dialog, facts) if facts else None) or self._quest_menu(dialog)
            if link is None:
                return dialog
            dialog = t.follow(link)
        return dialog

    def advance(self, npc_id: int, facts: QuestFacts, max_links: int = 6) -> Dialog | None:
        """
        Talk to ``npc_id`` about the quest and follow its links forward (the start event first, never a dead end)
        until the dialog has no more; return the last dialog.
        """
        dialog = self.open(npc_id, facts)
        for _ in range(max_links):
            link = self._forward_link(dialog, facts)
            if link is None:
                break
            dialog = self.talker.follow(link) or dialog
        if dialog is not None:
            log.info("[%s] %s: %s", facts.name, self.session.knowledge.npc_name(npc_id),
                     html_text(dialog.html)[:200].replace("\n", " "))
        self.refresh()
        return dialog

    def _quest_menu(self, dialog: Dialog) -> Link | None:
        return next((lk for lk in dialog.links if re.search(r"_Quest$|^Quest$|^Script$", lk.command)
                     or lk.text.strip().lower() == "quest"), None)

    def _quest_link(self, dialog: Dialog, facts: QuestFacts | None) -> Link | None:
        if facts is None:
            return None
        return next((lk for lk in dialog.links if facts.script in lk.command and not re.search(r"\.htm", lk.command)
                     or lk.text.strip().lower().startswith(facts.name.lower())), None)

    def _forward_link(self, dialog: Dialog | None, facts: QuestFacts) -> Link | None:
        if dialog is None or not dialog.links:
            return None
        links = [lk for lk in dialog.links if not _DEAD_ENDS.search(lk.text)]
        for event in facts.start_events:
            for lk in links:
                if event in lk.command:
                    return lk
        own = [lk for lk in links if facts.script in lk.command or re.search(r"\d{5}-\d+\w*\.htm", lk.command)]
        return own[0] if own else None

    # ------------------------------------------------------------------
    # Hunting for quest items
    # ------------------------------------------------------------------

    def hunt_items(self, facts: QuestFacts, item_id: int, count: int, timeout: float = 3600.0) -> bool:
        """Kill the quest's monsters near their nearest spawn until ``count`` of ``item_id`` are in the bag."""
        s = self.session
        spawns = [p for m in facts.kill_npcs for p in s.knowledge.spawn_points(m, self.area)]
        if not spawns:
            log.warning("[%s] no spawn of its monsters %s known", facts.name, facts.kill_npcs)
            return False
        x, y, _ = s.position
        home = min(spawns, key=lambda p: p.distance(x, y))
        log.info("[%s] hunting %s for %d x %s at %s", facts.name, [s.knowledge.npc_name(m) for m in facts.kill_npcs],
                 count, s.knowledge.item_name(item_id), home)
        if not self.talker.nav.move_to(home, arrive=600, timeout=900):
            log.warning("[%s] could not reach the hunting ground", facts.name)
        result = self.fighter.hunt(home, until=lambda: s.item_count(item_id) >= count, timeout=timeout,
                                   monsters=set(facts.kill_npcs), max_deaths=5)
        log.info("[%s] hunt: %s, %d/%d", facts.name, result, s.item_count(item_id), count)
        return s.item_count(item_id) >= count

    @cached_property
    def plans(self) -> dict:
        return PLANS


# ----------------------------------------------------------------------
# Plans: what to do for each quest, in order. A step is ("talk", npc id) or ("hunt", item id, count).
# ----------------------------------------------------------------------

PLANS: dict[int, list[tuple]] = {
    # Letters of Love: Darin's letter to Roxxy, her kerchief back, the receipt to Baulro, his potion back.
    1: [("talk", 30048), ("talk", 30006), ("talk", 30048), ("talk", 30033), ("talk", 30048)],
    # Bring Wolf Pelts: 40 pelts from wolves to Lector.
    258: [("talk", 30001), ("hunt", 702, 40), ("talk", 30001)],
    # The Guard is Busy: amulets, necklaces and fangs from orcs and werewolves to Gilbert (10+ for the bonus).
    257: [("talk", 30039), ("hunt_any", (752, 1085, 1086), 10), ("talk", 30039)],
}


def run_plan(runner: QuestRunner, quest_id: int) -> bool:
    """Do the plan of ``quest_id``; ``True`` when its last step was reached."""
    facts = runner.book.facts(quest_id)
    s = runner.session
    if s.char.level < facts.min_level:
        log.info("[%s] needs level %d", facts.name, facts.min_level)
        return False
    for step in PLANS[quest_id]:
        if step[0] == "talk":
            if runner.advance(step[1], facts) is None:
                return False
        elif step[0] == "hunt":
            if not runner.hunt_items(facts, step[1], step[2]):
                return False
        elif step[0] == "hunt_any":
            items, count = step[1], step[2]
            ok = runner.fighter.hunt(_home(runner, facts), until=lambda: sum(s.item_count(i) for i in items) >= count,
                                     monsters=set(facts.kill_npcs), timeout=3600, max_deaths=5)
            if ok.reason != "done":
                return False
    return True


def _home(runner: QuestRunner, facts: QuestFacts) -> Point:
    s = runner.session
    spawns = [p for m in facts.kill_npcs for p in s.knowledge.spawn_points(m, runner.area)]
    x, y, _ = s.position
    home = min(spawns, key=lambda p: p.distance(x, y))
    runner.talker.nav.move_to(home, arrive=600, timeout=900)
    return home
