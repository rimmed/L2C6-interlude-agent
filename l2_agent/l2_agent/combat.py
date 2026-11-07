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


@dataclass
class HuntResult:
    """Kills and deaths of a hunt and why it ended."""

    kills: int = 0
    deaths: int = 0
    reason: str = ""


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
        self.kills = 0
        self.deaths = 0

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
        kills0, deaths0 = world.events.kills, self.deaths
        deadline = time.time() + timeout
        while True:
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
                if entry.get("npc_id") not in monsters and entry["object_id"] != world.target_id:
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
        obs = s.encoder.observe(world.char, world.target_id, world.xy, has_loot=bool(world.pending_loot),
                                has_potion=find_healing_potion(world.char) is not None, now=now)
        decision = s.encoder.act(obs, sample=not self.cfg.greedy, rng=self.rng)
        executor.execute(decision, now)
        self._last_decision = now

    def _pick_up(self, now: float) -> bool:
        """Walk to the nearest item on the ground when no monster is fighting the agent; ``True`` while it does."""
        s, world = self.session, self.session.world
        if not world.pending_loot or world.target_alive() or s.executor.busy(now):
            return False
        if any(m.get("attacking_me_until", 0.0) > now for m in world.char.radar.entries.values()):
            return False
        near = [i for i in world.pending_loot if world.distance_to(i["x"], i["y"]) <= self.cfg.pickup_range]
        if not near:
            return False
        item = min(near, key=lambda i: world.distance_to(i["x"], i["y"]))
        object_id, asked = self._pickup
        if object_id == item["object_id"]:
            if now - asked < 3.0:
                return True
            # Asked twice already: someone else's drop or out of reach, leave it.
            world.pending_loot = [i for i in world.pending_loot if i["object_id"] != object_id]
            return False
        self.session.send(packets.build_action(item["object_id"], item["x"], item["y"], item["z"]))
        self._pickup = (item["object_id"], now)
        return True

    def _on_dead(self, now: float) -> None:
        world = self.session.world
        if world.dead_since != self._death_at:
            self._death_at = world.dead_since
            self.deaths += 1
            log.warning("Died at %s", world.pos)
        if now - world.dead_since >= self.cfg.respawn_after and now - self._restart_sent > 10.0:
            self.session.send(packets.build_request_restart_point(0))  # to the nearest town
            self._restart_sent = now
