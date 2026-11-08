"""
Turns the model's decisions into game packets.

The executor carries the lessons of the live tests with ``client_demo.py``:

* a click on the already selected monster is a weapon hit in L2 - casters (mystic classes) keep
  their target instead, the training data shows them casting, not meleeing;
* ``AttackRequest`` is not re-sent on every step for the same target (auto-attack keeps going);
* while our own cast runs (``MagicSkillUse``) and while the character walks to a picked item, no new
  decision is taken - a new command would cancel the cast / the walk;
* a target that took no hit for a while is unreachable (terrain, no pathfinding) and gets ignored;
* low on MP or HP and out of combat, the character sits down until it is nearly full (the model has
  no rest action; a mystic without MP only meleed with its staff and died).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from l2_client import packets
from l2_client.policy import CombatPolicy, Decision
from l2_client.tools.data import item_data, skill_data, summon_data

from .world import WorldState

log = logging.getLogger(__name__)

#: Mystic classes (``PlayerClass.isMage()`` on the server).
MAGE_CLASS_IDS = frozenset((10, 11, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42, 43,
                            49, 50, 51, 52, 94, 95, 96, 97, 98, 103, 104, 105, 110, 111, 112, 115, 116))
#: Greater, normal and Lesser Healing Potion, best first.
HEALING_POTION_IDS = (1539, 1061, 1060)


@dataclass
class ExecutorConfig:
    """Timings and distances of the executor's safeguards."""

    #: do not re-send AttackRequest on the same target more often than this (seconds)
    attack_repeat_s: float = 1.5
    #: after a pickup command, hold new decisions this long while the item is still on the ground
    pickup_commit_s: float = 3.0
    #: give up a target that took no hit for this long ...
    stuck_target_s: float = 15.0
    #: ... and keep it off the radar for this long
    ignore_target_s: float = 60.0
    #: "moved" walks to the target only when farther than this
    approach_range: int = 100
    #: "moved" with nothing to walk to: random step of up to this many units ...
    roam_step: int = 400
    #: ... staying within this distance of the farm-zone point (walks back when farther)
    roam_radius: int = 1200
    #: RequestSkillCoolTime at most this often while a skill cools down
    cooldown_sync_s: float = 1.0
    #: sit down out of combat below this share of MP ...
    rest_mp: float = 0.3
    #: ... of MP for these classes: they fight with nukes only, and a fight started low on MP is lost far more often
    rest_mp_caster: float = 0.5
    caster_classes: frozenset[int] = frozenset()
    #: ... or of HP ...
    rest_hp: float = 0.4
    #: ... ask for a full restore (.heal) once seated, again after this time when it did not come ...
    rest_heal: bool = True
    rest_heal_retry_s: float = 2.0
    #: ... and stand up when both are back at this share (or when attacked)
    rest_until: float = 0.95
    #: re-send the sit / stand toggle when ChangeWaitType did not come in this time
    rest_retry_s: float = 2.0
    #: below this share of HP the character casts its own heal (``SELF_HEAL_SKILLS``) when it has one ready:
    #: scripted like the buff potions, a hint for the model, which hardly ever picks it (the Elven Scout cast
    #: Elemental Heal once in 60 000 fights of rl_bc_v11q); 0 = off
    self_heal_hp: float = 0.4


#: Shop buff potions (RLBuffPotions on the server) -> the skill of their effect; Greater ones first.
BUFF_POTIONS: dict[int, int] = {1374: 2034, 734: 2011, 1375: 2035, 735: 2012, 6036: 2169, 6035: 2169}
#: A drunk potion's effect may take a moment to show up in AbnormalStatusUpdate: do not drink again meanwhile.
POTION_RETRY_S = 8.0
#: Heals a fighter casts on itself (Elemental Heal of the elven fighters), best first.
SELF_HEAL_SKILLS: tuple[int, ...] = (58,)


def find_healing_potion(char) -> dict | None:
    """Best healing potion in the inventory, or ``None``."""
    for potion_id in HEALING_POTION_IDS:
        potion = next((i for i in char.items if i["item_id"] == potion_id and i["count"] > 0), None)
        if potion:
            return potion
    return None


class ActionExecutor:
    """Sends the packets for :class:`~l2_client.policy.Decision` objects."""

    def __init__(self, send: Callable[[bytes], None], world: WorldState, encoder: CombatPolicy,
                 config: ExecutorConfig | None = None, rng: random.Random | None = None):
        self._send = send
        self.world = world
        self.encoder = encoder
        self.cfg = config or ExecutorConfig()
        self.rng = rng or random.Random()
        self._attack_target = 0
        self._attack_at = 0.0
        self._loot_id = 0
        self._loot_at = 0.0
        self._pickup_until = 0.0
        self._engaged = 0
        self._engaged_at = 0.0
        self._last_cooldown_sync = 0.0
        self._potion_tried: dict[str, float] = {}
        self._toggle_at = 0.0
        #: real time spent resting in the current fight (episode statistics)
        self.rest_s = 0.0
        #: times the character asked for .heal in the current fight (episode statistics)
        self.heals = 0
        #: own heals the executor cast in the current fight (``self_heal``)
        self.self_heals = 0
        self._self_heal_at = 0.0
        self._rest_tick = 0.0
        self._heal_at = 0.0

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def busy(self, now: float) -> bool:
        """``True`` while a cast or a walk to loot is running; a decision now would cancel it."""
        if now < self.world.cast_until:
            return True
        if now < self._pickup_until and any(item["object_id"] == self._loot_id for item in self.world.pending_loot):
            return True
        return False

    def rest(self, now: float) -> bool:
        """
        Sit down when MP or HP is low out of combat, restore both with .heal and stand up; stand up as well when
        a monster attacks. Scripted like the buff potions. ``True`` while the character rests: no decision then.
        """
        world, char, cfg = self.world, self.world.char, self.cfg
        attacked = any(mob.get("attacking_me_until", 0.0) > now for mob in char.radar.entries.values())
        mp = char.cur_mp / char.max_mp if char.max_mp else 1.0
        hp = char.cur_hp / char.max_hp if char.max_hp else 1.0
        rest_mp = cfg.rest_mp_caster if char.class_id in cfg.caster_classes else cfg.rest_mp
        if world.sitting:
            self.rest_s += now - self._rest_tick if self._rest_tick else 0.0
            self._rest_tick = now
            if not attacked and (mp < cfg.rest_until or hp < cfg.rest_until):
                if cfg.rest_heal and now - self._heal_at >= cfg.rest_heal_retry_s:
                    self._send(packets.build_say2(".heal"))
                    self._heal_at = now
                    self.heals += 1
                return True
            if now - self._toggle_at >= cfg.rest_retry_s:
                self._send(packets.build_request_action_use(0))  # 0 = sit / stand
                self._toggle_at = now
            return True  # until ChangeWaitType says it stands
        self._rest_tick = 0.0
        if now - self._toggle_at < cfg.rest_retry_s:
            return now - self._toggle_at < 1.0 and (mp < rest_mp or hp < cfg.rest_hp)  # a sit on its way
        if attacked or world.target_alive() or self.busy(now) or (mp >= rest_mp and hp >= cfg.rest_hp):
            return False
        self._send(packets.build_request_action_use(0))
        self._toggle_at = now
        return True

    def drop_stuck_target(self, now: float) -> bool:
        """Ignore the current target when it took no hit for ``stuck_target_s`` (it is unreachable)."""
        world = self.world
        if world.target_id != self._engaged:
            self._engaged, self._engaged_at = world.target_id, now
        if not world.target_id:
            return False
        if now - max(self._engaged_at, world.last_hit_at) <= self.cfg.stuck_target_s:
            return False
        mob = world.char.radar.entries.get(world.target_id, {})
        log.info("No hit on %d for %.0fs -- ignoring it for %.0fs (me at %s, it at %s)", world.target_id,
                 self.cfg.stuck_target_s, self.cfg.ignore_target_s, world.pos,
                 (mob.get("x"), mob.get("y"), mob.get("z")))
        world.ignore_target(world.target_id, self.cfg.ignore_target_s, now)
        self.encoder.reset_session()
        return True

    def drink_buff_potions(self, now: float) -> str | None:
        """
        Drink a buff potion whose kind of effect (run / attack / casting speed) is not on the character.

        Scripted like automatic soulshots, not a model decision: the model has no buff features yet.
        At most one potion per call; returns its name or ``None``.
        """
        char = self.world.char
        active = {skill_data.get_abnormal_type(b["skill_id"]) for b in char.buffs}
        for item_id, effect_skill in BUFF_POTIONS.items():
            kind = skill_data.get_abnormal_type(effect_skill)
            if not kind or kind in active or now - self._potion_tried.get(kind, 0.0) < POTION_RETRY_S:
                continue
            potion = next((i for i in char.items if i["item_id"] == item_id and i["count"] > 0), None)
            if potion is None:
                continue
            self._send(packets.build_use_item(potion["object_id"]))
            self._potion_tried[kind] = now
            return item_data.get_item_name(item_id) or str(item_id)
        return None

    def self_heal(self, now: float) -> str | None:
        """
        Cast the character's own heal (``SELF_HEAL_SKILLS``) when its HP is below ``self_heal_hp`` and the heal is
        off cooldown. Scripted like the buff potions; returns the skill's name or ``None``.
        """
        world, char, cfg = self.world, self.world.char, self.cfg
        if cfg.self_heal_hp <= 0 or world.dead or world.sitting or now < world.cast_until or char.max_hp <= 0:
            return None
        if char.cur_hp / char.max_hp >= cfg.self_heal_hp:
            return None
        known = {s["id"] for s in char.skills if not s.get("passive")}
        skill_id = next((s for s in SELF_HEAL_SKILLS if s in known), None)
        if skill_id is None:
            return None
        # its reuse (game time) at the server's speed, and a margin for the cast itself
        reuse = (skill_data.get_skill_reuse_delay(skill_id) or 10000) / 1000.0 / max(world.server_speed, 1e-6) + 0.5
        if now - self._self_heal_at < reuse:
            return None
        self._send(packets.build_use_skill(skill_id))
        self._self_heal_at = now
        self.self_heals += 1
        return skill_data.get_skill_name(skill_id) or str(skill_id)

    def sync_cooldowns(self, now: float) -> None:
        """Ask the server for the real cooldowns while a skill is cooling down."""
        if self.encoder.wants_cooldown_sync(now) and now - self._last_cooldown_sync >= self.cfg.cooldown_sync_s:
            self._send(packets.build_request_skill_cool_time())
            self._last_cooldown_sync = now

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, decision: Decision, now: float | None = None) -> str:
        """Send the packets of ``decision``; return a short description of what was sent."""
        now = time.time() if now is None else now
        world, char = self.world, self.world.char
        radar = char.radar.entries
        x, y, z = world.pos
        mob_id = decision.mob_object_id
        action = decision.action

        if action == "selected_target" and mob_id is not None:
            if mob_id != world.target_id:
                return self._select(mob_id)
            if char.class_id in MAGE_CLASS_IDS:
                return "keep target"  # a caster waits for the next cast instead of meleeing
            if self._attack_target != mob_id or now - self._attack_at >= self.cfg.attack_repeat_s:
                # A click on the already selected monster is an attack in L2.
                self._send(packets.build_action(mob_id, x, y, z, shift_click=False))
                self._attack_target, self._attack_at = mob_id, now
                self.encoder.note_combat(now)
                return f"attack click on {mob_id}"
            return "keep attacking"

        if action == "attacked":
            target = world.target_id if world.target_alive() else mob_id
            if target is None:
                return "no target"
            world.target_id = target
            self.encoder.note_combat(now)
            if self._attack_target == target and now - self._attack_at < self.cfg.attack_repeat_s:
                return "keep attacking"
            self._send(packets.build_attack_request(target, x, y, z))
            self._attack_target, self._attack_at = target, now
            return f"AttackRequest on {target}"

        if action == "used_skill" and decision.skill_id is not None:
            selected = ""
            if not world.target_alive() and mob_id is not None:
                selected = self._select(mob_id) + ", "  # a skill needs a selected target
            self._send(packets.build_use_skill(decision.skill_id, ctrl_pressed=True))
            self._send(packets.build_request_skill_cool_time())
            self._last_cooldown_sync = now
            self.encoder.note_skill_used(decision.skill_id, now)
            self.encoder.note_combat(now)
            return f"{selected}UseSkill {skill_data.get_skill_name(decision.skill_id) or decision.skill_id}"

        if action == "picked_up" and world.pending_loot:
            nearest = min(world.pending_loot, key=lambda item: world.distance_to(item["x"], item["y"]))
            if self._loot_id == nearest["object_id"] and now - self._loot_at <= 2.0:
                return "walking to loot"
            # Action on an item makes the server walk there and pick it up.
            self._send(packets.build_action(nearest["object_id"], nearest["x"], nearest["y"], nearest["z"]))
            self._loot_id, self._loot_at = nearest["object_id"], now
            self._pickup_until = now + self.cfg.pickup_commit_s
            return f"pickup {item_data.get_item_name(nearest['item_id']) or nearest['item_id']}"

        if action == "used_item":
            potion = find_healing_potion(char)
            if potion:
                self._send(packets.build_use_item(potion["object_id"]))
                return f"UseItem {item_data.get_item_name(potion['item_id'])}"
            return "no potion"

        if action == "pet_attack":
            pet = char.pet
            target = mob_id if mob_id is not None else (world.target_id if world.target_alive() else None)
            if pet is None or target is None:
                return "no summon / target"
            selected = self._select(target) + ", " if target != world.target_id else ""
            # The summon attacks the owner's target: the selection above reaches the server first.
            servitor, pet_type = summon_data.ATTACK_ACTIONS
            self._send(packets.build_request_action_use(servitor if pet.get("summon_type") == 1 else pet_type))
            self.encoder.note_combat(now)
            return f"{selected}summon attacks {target}"

        if action == "pet_skill" and decision.pet_action_id is not None:
            selected = ""
            if not decision.pet_on_owner and not world.target_alive() and mob_id is not None:
                selected = self._select(mob_id) + ", "  # a summon skill hits the owner's target
            self._send(packets.build_request_action_use(decision.pet_action_id, ctrl_pressed=True))
            self.encoder.note_combat(now)
            return f"{selected}summon skill {skill_data.get_skill_name(decision.skill_id) or decision.skill_id}"

        if action == "moved":
            goal_id = world.target_id if world.target_alive() else mob_id
            goal = radar.get(goal_id) if goal_id is not None else None
            if goal and world.distance_to(goal["x"], goal["y"]) <= self.cfg.approach_range:
                # Radar positions are spawn points, so "already close" can repeat forever while the
                # monster is elsewhere: engage it, or - if it is already the target - step away.
                if goal_id != world.target_id:
                    return self._select(goal_id) + " (already close)"
                tx, ty, tz = self._roam_target()
            elif goal:
                tx, ty, tz = goal["x"], goal["y"], goal["z"]
            else:
                tx, ty, tz = self._roam_target()
            self._send(packets.build_move_to_location(tx, ty, tz, x, y, z, 1))
            world.pos = (tx, ty, z)  # optimistic: the next UserInfo corrects it
            return f"move to ({tx},{ty})"

        return "nothing to do"

    def _roam_target(self) -> tuple[int, int, int]:
        """A random step that keeps the character around its farm zone (no pathfinding: stay where mobs are)."""
        x, y, z = self.world.pos
        home = self.world.home
        if home is not None and self.world.distance_to(home[0], home[1]) > self.cfg.roam_radius:
            return home  # wandered off: walk back to the zone
        step = self.cfg.roam_step
        return x + self.rng.randint(-step, step), y + self.rng.randint(-step, step), z

    def _select(self, object_id: int) -> str:
        x, y, z = self.world.pos
        self._send(packets.build_action(object_id, x, y, z, shift_click=False))
        self.world.target_id = object_id
        return f"select {object_id}"
