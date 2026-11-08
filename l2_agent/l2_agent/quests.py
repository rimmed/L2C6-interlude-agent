"""
Quests through the NPC dialogs. The facts come from the quest table (``l2jc6_quests``): the start and talk NPCs,
the monsters, the quest items and, from the script, which dialog event starts the quest (the event after which the
script calls ``startQuest()``). What to do in which order is scripted per quest in :data:`PLANS`.
"""

from __future__ import annotations

import logging
import random
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
        home = pick_spawn(spawns, *s.position[:2])
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
    active = quest_id in runner.refresh()
    for step in PLANS[quest_id]:
        if step[0] == "talk":
            if runner.advance(step[1], facts) is None:
                return False
            quests = runner.refresh()
            if active and quest_id not in quests:
                return True  # finished at this talk (the quest was further along than the plan's start)
            active = quest_id in quests
        elif step[0] == "hunt":
            if not runner.hunt_items(facts, step[1], step[2]):
                return False
        elif step[0] == "hunt_any":
            items, count = step[1], step[2]
            home = _home(runner, facts)
            if home is None:
                return False
            ok = runner.fighter.hunt(home, until=lambda: sum(s.item_count(i) for i in items) >= count,
                                     monsters=set(facts.kill_npcs), timeout=3600, max_deaths=5)
            if ok.reason != "done":
                return False
    return True


def _home(runner: QuestRunner, facts: QuestFacts) -> Point | None:
    s = runner.session
    spawns = [p for m in facts.kill_npcs for p in s.knowledge.spawn_points(m, runner.area)]
    if not spawns:
        return None
    home = pick_spawn(spawns, *s.position[:2])
    runner.talker.nav.move_to(home, arrive=600, timeout=900)
    return home


def pick_spawn(spawns: list[Point], x: float, y: float) -> Point:
    """
    A random one of the (up to 4) spawns not much farther than the nearest: players spread over the hunting places
    instead of crowding the same monsters (whose drops go to whoever hit them most).
    """
    spawns = sorted(spawns, key=lambda p: p.distance(x, y))
    nearest = spawns[0].distance(x, y)
    return random.choice([p for p in spawns if p.distance(x, y) <= nearest + 2500][:4])


# ----------------------------------------------------------------------
# Quests without a plan: from the quest table alone
# ----------------------------------------------------------------------

_ITEM_TARGET = re.compile(r"getQuestItemsCount\(\s*\w+\s*,\s*(\w+)\s*\)\s*(>=|<|==|>)\s*(\d+)")


def _named_first(npc_ids: list[int], text: str, knowledge, speaker: int | None = None) -> list[int]:
    """
    ``npc_ids`` with those whose name ``text`` (said by ``speaker``) mentions first, the last mentioned first ("I see
    you talked with Sir Collin Windawood! You should now speak back to Gatekeeper Roxxy").
    """
    where = {}
    for n in npc_ids:
        if text and n != speaker:
            hits = [m.start() for m in re.finditer(rf"\b{re.escape(knowledge.npc_name(n))}\b", text, re.I)]
            if hits:
                where[n] = hits[-1]
    named = sorted(where, key=lambda n: -where[n])
    return named + [n for n in npc_ids if n not in where]


def item_targets(facts: QuestFacts, source: str, constants: dict[str, int]) -> dict[int, int]:
    """How many of each quest item the script checks for (``getQuestItemsCount(player, X) >= N``)."""
    targets: dict[int, int] = {}
    for name, op, number in _ITEM_TARGET.findall(source):
        item = int(name) if name.isdigit() else constants.get(name)
        if item is None or item not in facts.quest_items:
            continue
        need = int(number) + (1 if op == ">" else 0)
        if need > 0:
            targets[item] = max(targets.get(item, 0), need)
    return targets


def run_generic(runner: QuestRunner, quest_id: int, max_rounds: int = 6) -> str:
    """
    A quest without a plan: start it at its start NPC, then by turns talk to its NPCs (the start NPC last) and,
    when no talk moves it on (no new cond, no new item), hunt its monsters for the quest items its script counts.
    ``done`` when it left the quest list, ``refused`` when it did not start (level, race, class), ``stuck`` else.
    """
    s = runner.session
    facts = runner.book.facts(quest_id)
    row = s.knowledge.quests.loc[quest_id]
    constants = {c["name"]: int(c["value"]) for c in row["constants"]}
    targets = item_targets(facts, row["source"], constants)
    resumed = quest_id in runner.refresh()  # taken before (e.g. before a restart)
    if not resumed:
        x, y, _ = s.position
        starts = [n for n in facts.start_npcs if s.knowledge.nearest_spawn(n, x, y, runner.area)]
        if not starts:
            return "refused"
        runner.advance(min(starts, key=lambda n: s.knowledge.nearest_spawn(n, x, y, runner.area).distance(x, y)), facts)
        if quest_id not in runner.refresh():
            log.info("[%s] did not start", facts.name)
            return "refused"
    talkers = [n for n in facts.talk_npcs if n not in facts.start_npcs] + [n for n in facts.talk_npcs if n in facts.start_npcs]
    said, speaker = "", None  # the last reply: an NPC it names ("go to Gatekeeper Roxxy") is the next to visit
    stale = 0
    while stale < max_rounds:
        quests = runner.refresh()
        if quest_id not in quests:
            return "done"
        cond = quests[quest_id]
        bag = {i: s.item_count(i) for i in facts.quest_items}
        moved = False
        # A quest taken before whose counted items are still short: nobody has anything new to say, hunt first.
        remaining = [] if resumed and targets and any(s.item_count(i) < n for i, n in targets.items()) else list(talkers)
        resumed = False
        while remaining:  # after each talk the NPC its reply names comes next
            npc_id = _named_first(remaining, said, s.knowledge, speaker)[0]
            remaining.remove(npc_id)
            dialog = runner.advance(npc_id, facts)
            said, speaker = (html_text(dialog.html) if dialog is not None else ""), npc_id
            quests = runner.refresh()
            if quest_id not in quests:
                return "done"
            if quests[quest_id] != cond or {i: s.item_count(i) for i in facts.quest_items} != bag:
                moved = True
                break
        if moved:
            stale = 0
            continue
        stale += 1
        if facts.kill_npcs:
            need = {i: n for i, n in targets.items() if s.item_count(i) < n}
            home = _home(runner, facts)
            if home is None:
                log.info("[%s] its monsters do not live in the area", facts.name)
                return "stuck"
            # the items the script counts, or the cond moving on (kills it counts in its state; the server telling
            # that enough was collected)
            def until(need=need, cond=cond) -> bool:
                return s.quests.get(quest_id, -1) != cond or bool(need) and all(
                    s.item_count(i) >= n for i, n in need.items())
            result = runner.fighter.hunt(home, until=until, monsters=set(facts.kill_npcs), timeout=1200, max_deaths=3)
            log.info("[%s] hunt for %s: %s", facts.name, {s.knowledge.item_name(i): n for i, n in need.items()} or "the cond",
                     result)
    log.info("[%s] no progress after %d rounds", facts.name, max_rounds)
    return "stuck"
