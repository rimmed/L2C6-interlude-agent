"""
Learning skills at a trainer, as a player does at a new level: the class's skill tree (``l2jc6_skill_trees``) tells
which skills it may learn now and for how much SP, the trainer table (``l2jc6_trainers``) which NPCs teach the class;
the trainer's AcquireSkillList is the live list, and each skill is asked for (RequestAcquireSkillInfo) and learned
(RequestAcquireSkill) while the SP last.
"""

from __future__ import annotations

import logging
import time

from l2_client import packets

from .dialog import Talker
from .knowledge import Area
from .session import Session

log = logging.getLogger(__name__)


class SkillTrainer:
    """Finds the class's trainer and learns the skills that are due."""

    def __init__(self, session: Session, talker: Talker, area: Area | None = None):
        self.session = session
        self.talker = talker
        self.area = area

    def known(self) -> dict[int, int]:
        """Skill id -> level the character has."""
        return {int(s["id"]): int(s["level"]) for s in self.session.char.skills}

    def due(self) -> list[tuple[int, int, int]]:
        """(skill id, level, sp) the character can learn from a trainer now, by the skill tree and its SP."""
        s, k = self.session, self.session.knowledge
        tree = k.skill_trees
        tree = tree[(tree["class_id"] == s.char.class_id) & tree["learned_by_npc"] & (tree["get_level"] <= s.char.level)]
        known, sp = self.known(), int(getattr(s.char, "sp", 0) or 0)
        out = []
        for row in tree.sort_values(["skill_id", "skill_level"]).itertuples():
            if known.get(int(row.skill_id), 0) + 1 == int(row.skill_level) and row.sp <= sp \
                    and self._has_items(row.items):
                out.append((int(row.skill_id), int(row.skill_level), int(row.sp)))
                known[int(row.skill_id)] = int(row.skill_level)  # the next level may follow in the same visit
        return out

    def _has_items(self, items) -> bool:
        """The character carries what learning the skill consumes (e.g. its spellbook)."""
        return all(self.session.item_count(int(i["item_id"])) >= int(i["count"]) for i in (items if items is not None else []))

    def books_wanted(self) -> list[int]:
        """
        Items (spellbooks) the skills of the character's level need and it does not carry: a player buys them
        before visiting the trainer.
        """
        s, k = self.session, self.session.knowledge
        tree = k.skill_trees
        tree = tree[(tree["class_id"] == s.char.class_id) & tree["learned_by_npc"] & (tree["get_level"] <= s.char.level)]
        known = self.known()
        out = []
        for row in tree.itertuples():
            if known.get(int(row.skill_id), 0) < int(row.skill_level):
                out += [int(i["item_id"]) for i in (row.items if row.items is not None else [])
                        if s.item_count(int(i["item_id"])) < int(i["count"])]
        return sorted(set(out))

    def _needs(self, skill_id: int, level: int):
        tree = self.session.knowledge.skill_trees
        rows = tree[(tree["class_id"] == self.session.char.class_id) & (tree["skill_id"] == skill_id)
                    & (tree["skill_level"] == level)]
        return rows.iloc[0]["items"] if len(rows) else []

    def trainer(self) -> int | None:
        """The nearest NPC in the area that teaches the character's class."""
        s, k = self.session, self.session.knowledge
        ids = [int(r.npc_id) for r in k.trainers.itertuples() if s.char.class_id in set(r.class_ids)]
        x, y, _ = s.position
        best = None
        for npc_id in ids:
            spawn = k.nearest_spawn(npc_id, x, y, self.area)
            if spawn is not None and (best is None or spawn.distance(x, y) < best[1]):
                best = (npc_id, spawn.distance(x, y))
        return best[0] if best else None

    def learn(self) -> list[tuple[int, int]]:
        """Visit the trainer and learn every skill its list offers and the SP pay for; the (id, level) learned."""
        s = self.session
        npc_id = self.trainer()
        if npc_id is None:
            log.warning("No trainer of class %d in the area", s.char.class_id)
            return []
        dialog = self.talker.talk(npc_id)
        link = self.talker.choose(dialog, r"_SkillList$")
        if link is None:
            log.warning("%s offers no skill list", s.knowledge.npc_name(npc_id))
            return []
        learned: list[tuple[int, int]] = []
        for _ in range(20):
            asked = time.time()
            self.talker.follow(link, expect=lambda: (s.skill_list or {}).get("received", 0) >= asked
                               or (s.dialog is not None and s.dialog.received >= asked))
            offer = [sk for sk in (s.skill_list or {}).get("skills", []) if (s.skill_list or {}).get("received", 0) >= asked]
            sp = int(getattr(s.char, "sp", 0) or 0)
            affordable = [sk for sk in offer if sk["sp"] <= sp and self._has_items(self._needs(sk["id"], sk["level"]))]
            if not affordable:
                break
            skill = affordable[0]
            before = self.known().get(skill["id"], 0)
            s.send(packets.build_request_acquire_skill_info(skill["id"], skill["level"]))
            s.idle(0.3)
            s.send(packets.build_request_acquire_skill(skill["id"], skill["level"]))
            if not s.wait_until(lambda: self.known().get(skill["id"], 0) > before, 4.0):
                log.warning("Learning %s %d failed", skill["id"], skill["level"])
                break
            learned.append((skill["id"], skill["level"]))
            log.info("Learned skill %d level %d for %d SP", skill["id"], skill["level"], skill["sp"])
        if learned:
            s.encoder.bind_character(s.char)  # the combat model's skill panel
        return learned
