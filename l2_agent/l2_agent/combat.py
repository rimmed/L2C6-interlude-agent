"""
Fighting with the trained combat model: the same decision loop as the RL actor (``l2_actor.actor``), without the
training records. The model picks the target among the monsters in sight, the skill or the attack, loot and
potions; the executor sits down to regenerate between fights. On death the agent asks for the town respawn.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from l2_actor.executor import find_healing_potion
from l2_client import packets

from .knowledge import Point
from .session import Session

log = logging.getLogger(__name__)


@dataclass
class CombatConfig:
    """How the combat loop runs the model."""

    decision_interval: float = 0.5
    #: argmax of the policy; sampling plays more varied but worse
    greedy: bool = True
    #: seconds dead before the agent asks for the town respawn
    respawn_after: float = 3.0
    buff_potions: bool = True
    #: items on the ground within this distance are picked up between fights (the model loots only now and then)
    pickup_range: float = 900.0


#: farther items the agent walks to before it asks the server to pick them up
PICKUP_WALK = 150.0


@dataclass
class HuntResult:
    """Kills and deaths of a hunt and why it ended."""

    kills: int = 0
    deaths: int = 0
    reason: str = ""
    #: items picked up, items on the ground left (someone else's or out of reach)
    picked: int = 0
    left: int = 0
    #: drops of its kills the server kept for the player who did more damage
    refused: int = 0


class Fighter:
    """The combat model's decision loop around a hunting spot."""

    def __init__(self, session: Session, cfg: CombatConfig | None = None, seed: int | None = None):
        self.session = session
        self.cfg = cfg or CombatConfig()
        self.rng = np.random.default_rng(seed)
        self._last_decision = 0.0
        self._restart_sent = 0.0
        self._death_at = 0.0
        self._pickup: tuple[int, float] = (0, 0.0)
        self._pickup_distance = 0.0
        self._pickup_tries = 0
        self.loot_left = 0
        self.loot_refused = 0
        self.kills = 0
        self.deaths = 0
        #: walks back to the hunting ground after a respawn (set by the agent; ``None``: stays where it is)
        self.nav = None

    def hunt(self, home: Point, until: Callable[[], bool], timeout: float = 1800.0, max_deaths: int = 3,
             monsters: set[int] | None = None) -> HuntResult:
        """
        Fight around ``home`` until ``until()`` holds. ``monsters``: template ids the agent may fight; others in
        sight are ignored for a while (the model then picks among the allowed ones).
        """
        s = self.session
        world = s.world
        world.home = (home.x, home.y, home.z)
        result = HuntResult()
        kills0, deaths0, picked0, left0 = world.events.kills, self.deaths, world.events.items_picked, self.loot_left
        refused0 = self.loot_refused
        deadline = time.time() + timeout
        was_dead = False
        while True:
            if was_dead and not world.dead and self.nav is not None:
                # respawned in town: back to the hunting ground (the walk fights back on the way)
                s.idle(2.0)
                log.info("Back to the hunting ground at %s", home)
                self.nav.move_to(home, arrive=400, timeout=min(600.0, max(deadline - time.time(), 1.0)))
                world.home = (home.x, home.y, home.z)
            was_dead = world.dead
            if until():
                result.reason = "done"
                break
            if time.time() > deadline:
                result.reason = "timeout"
                break
            if self.deaths - deaths0 >= max_deaths:
                result.reason = "deaths"
                break
            s.pump(0.05)
            self.tick(time.time(), monsters)
        result.kills = world.events.kills - kills0
        result.picked, result.left = world.events.items_picked - picked0, self.loot_left - left0
        result.refused = self.loot_refused - refused0
        result.deaths = self.deaths - deaths0
        return result

    def tick(self, now: float, monsters: set[int] | None = None) -> None:
        s, world, executor = self.session, self.session.world, self.session.executor
        world.char.radar.prune(*world.xy)
        if world.dead:
            self._on_dead(now)
            return
        self._restart_sent = 0.0
        if monsters is not None:
            for entry in list(world.char.radar.entries.values()):
                # a monster that attacks the agent is fought back whatever it is
                if entry.get("npc_id") not in monsters and entry["object_id"] != world.target_id \
                        and entry.get("attacking_me_until", 0.0) <= now:
                    world.ignore_target(entry["object_id"], 30.0, now)
        executor.sync_cooldowns(now)
        if self.cfg.buff_potions:
            executor.drink_buff_potions(now)
        executor.drop_stuck_target(now)
        if self._pick_up(now):
            return
        if executor.rest(now):
            return
        if now - self._last_decision < self.cfg.decision_interval or executor.busy(now):
            return
        obs = s.encoder.observe(world.char, world.target_id, world.xy, has_loot=any(i.get("mine") for i in world.pending_loot),
                                has_potion=find_healing_potion(world.char) is not None, now=now)
        decision = s.encoder.act(obs, sample=not self.cfg.greedy, rng=self.rng)
        executor.execute(decision, now)
        self._last_decision = now

    def _pick_up(self, now: float) -> bool:
        """
        Pick up the drops of the agent's own kills when no monster is fighting it; ``True`` while it does. One item
        at a time until it is taken, refused (another player's) or failed twice; the agent walks to far ones itself.
        """
        s, world = self.session, self.session.world
        object_id, asked = self._pickup
        current = next((i for i in world.pending_loot if i["object_id"] == object_id), None) if object_id else None
        if object_id and current is None:
            log.debug("Item %d gone after %.1fs (picked %d)", object_id, now - asked, world.events.items_picked)
            self._pickup, self._pickup_tries = (0, 0.0), 0
        if current is not None and world.pickup_failed_at >= asked:
            # the server refused it: the main damage dealer of that monster was someone else
            self.loot_refused += 1
            log.debug("Pick up of %d refused", object_id)
            # the rest of that monster's drop is the other player's too
            dropper = current.get("dropper_id")
            others = [i["object_id"] for i in world.pending_loot
                      if dropper and i.get("dropper_id") == dropper and i["object_id"] != object_id]
            self.loot_refused += len(others)
            world.pending_loot = [i for i in world.pending_loot if i["object_id"] not in others]
            self._drop_loot(object_id)
            return False
        if not world.pending_loot or s.executor.busy(now):
            return False
        if any(m.get("attacking_me_until", 0.0) > now for m in world.char.radar.entries.values()):
            return False
        if world.target_alive():
            # The model takes the next monster right after a kill: the drop comes first while that one is untouched
            # (no hit on it since it became the target, it does not attack); the fight follows.
            ex = s.executor
            if getattr(ex, "_engaged", 0) != world.target_id or world.last_hit_at >= getattr(ex, "_engaged_at", now):
                return False
        speed = max(float(getattr(world.char, "run_spd", 0) or 0), 80.0)
        if current is not None:
            if now - asked < 2.0 + self._pickup_distance / speed:
                return True  # the server is at it
            if self._pickup_tries >= 2:
                self.loot_left += 1
                log.info("Left %s x%s at %.0f (target %s, me %s, it %s)", s.knowledge.item_name(current.get("item_id", 0)),
                         current.get("count"), world.distance_to(current["x"], current["y"]), world.target_id or "-",
                         world.pos, (current["x"], current["y"], current["z"]))
                self._drop_loot(object_id)
                return False
            item = current
        else:
            # only the drops of the agent's own kills: the others belong to whoever killed the monster
            near = [i for i in world.pending_loot
                    if i.get("mine") and world.distance_to(i["x"], i["y"]) <= self.cfg.pickup_range]
            if not near:
                return False
            item = min(near, key=lambda i: world.distance_to(i["x"], i["y"]))
            self._pickup_tries = 0
        far = world.distance_to(item["x"], item["y"])
        if far > PICKUP_WALK and self.nav is not None:
            # the server does not walk the character to a far item: the agent walks there itself
            self.nav.move_to(Point(item["x"], item["y"], item["z"]), arrive=40, timeout=4.0 + far / speed)
            if not any(i["object_id"] == item["object_id"] for i in world.pending_loot):
                return True  # someone took it meanwhile
            now = time.time()
        log.debug("Pick up %s %d at %.0f (me %s, try %d)", s.knowledge.item_name(item.get("item_id", 0)),
                  item["object_id"], world.distance_to(item["x"], item["y"]), world.pos, self._pickup_tries + 1)
        s.send(packets.build_action(item["object_id"], item["x"], item["y"], item["z"]))
        self._pickup = (item["object_id"], now)
        self._pickup_tries += 1
        self._pickup_distance = world.distance_to(item["x"], item["y"])
        return True

    def _drop_loot(self, object_id: int) -> None:
        world = self.session.world
        world.pending_loot = [i for i in world.pending_loot if i["object_id"] != object_id]
        self._pickup, self._pickup_tries = (0, 0.0), 0

    def attackers(self, now: float | None = None) -> list[dict]:
        """The monsters in sight that are attacking the character."""
        now = time.time() if now is None else now
        return [m for m in self.session.world.char.radar.entries.values() if m.get("attacking_me_until", 0.0) > now]

    def revive(self, timeout: float = 60.0) -> bool:
        """Dead: ask for the town respawn and wait until the character stands there again."""
        s = self.session
        deadline = time.time() + timeout
        while s.world.dead and time.time() < deadline:
            s.pump(0.1)
            self._on_dead(time.time())
        if not s.world.dead:
            s.idle(2.0)  # the new surroundings arrive
        return not s.world.dead

    def defend(self, timeout: float = 180.0) -> HuntResult:
        """Fight back the monsters attacking the character (as a player does on the way), then go on."""
        s = self.session
        here = Point(*s.position)
        log.info("Attacked on the way by %s", [m.get("name") for m in self.attackers()])
        result = self.hunt(here, until=lambda: not self.attackers() and not s.world.target_alive(),
                           timeout=timeout, max_deaths=1)
        return result

    def _on_dead(self, now: float) -> None:
        world = self.session.world
        if world.dead_since != self._death_at:
            self._death_at = world.dead_since
            self.deaths += 1
            log.warning("Died at %s", world.pos)
        if now - world.dead_since >= self.cfg.respawn_after and now - self._restart_sent > 10.0:
            self.session.send(packets.build_request_restart_point(0))  # to the nearest town
            self._restart_sent = now
