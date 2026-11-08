"""
The actor's model of the game world, kept up to date from server packets.

:class:`WorldState` owns the character (:class:`~l2_client.character.L2Character` with its radar),
the loot on the ground, the current target and the death / cast / teleport flags. It also
accumulates :class:`StepEvents` - damage dealt and taken, kills, deaths, items picked up - which
the actor turns into rewards, and forwards the environment events to the observation encoder so the
model sees the same event rows as in training.

:meth:`WorldState.handle` returns *signals* for the actor: ``"kill"`` and ``"death"`` end a combat
episode, ``"revive"`` / ``"teleport"`` follow the server-side respawn, ``"skills_changed"`` asks
for the skill panel to be re-bound.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, fields

from l2_client import packets
from l2_client.character import L2Character
from l2_client.packets import GameServerPacket, SystemMessageId
from l2_client.policy import CombatPolicy
from l2_client.tools.data import npc_data

log = logging.getLogger(__name__)

#: Extra wait after the server-announced cast time before the next decision.
CAST_MARGIN_S = 0.15
#: A monster another player hit counts as "attacked by another player" this long (game seconds).
OTHER_ATTACKER_MEMORY_S = 10.0
#: A monster that hit the character counts as attacking it this long (game seconds): the ``attacking_me`` feature.
ATTACKER_MEMORY_S = 5.0
#: system messages "You have failed to pick up $s1 adena / $s1 / $s2 $s1(s)"
PICKUP_FAILED = (55, 56, 57)
#: An attack of the summon counts as "attacking" this long (game seconds): the logs' ``atk`` flag.
PET_ATTACK_MEMORY_S = 2.0


@dataclass
class StepEvents:
    """What happened in the game since the last decision step (the raw material of the reward)."""

    damage_dealt: int = 0
    #: damage dealt as a share of the target's maximum HP (1.0 = a whole monster), the reward's damage term
    damage_dealt_share: float = 0.0
    damage_taken: int = 0
    kills: int = 0
    deaths: int = 0
    items_picked: int = 0
    #: the character's summon died
    pet_deaths: int = 0

    def take(self) -> StepEvents:
        """Return a copy of the counters and reset them."""
        snapshot = StepEvents(**{f.name: getattr(self, f.name) for f in fields(self)})
        for f in fields(self):
            setattr(self, f.name, 0)
        return snapshot


class WorldState:
    """Game state of one character, fed packet by packet."""

    def __init__(self, encoder: CombatPolicy, send: Callable[[bytes], None], server_speed: float = 1.0):
        """
        :param encoder: observation encoder; receives the environment events (damage, kills) and the
            server's skill cooldowns
        :param send: sends a raw packet to the game server (used to finish teleports with Appearing)
        :param server_speed: server ``TimeScale``; converts game-time cast durations to real time
        """
        self.encoder = encoder
        self._send = send
        self.server_speed = max(server_speed, 1e-6)

        self.char = L2Character()
        #: the character's summon (``None`` without one): PetInfo / PetStatusUpdate state plus its
        #: ``target_id`` and ``attacking_until`` / ``casting_until`` (real time), see :meth:`_on_pet_info`
        self.char.pet = None
        self.pos: tuple[int, int, int] = (0, 0, 0)
        self.target_id: int = 0
        self.pending_loot: list[dict] = []
        self.events = StepEvents()

        self.dead = False
        self.dead_since = 0.0
        #: the character sits (ChangeWaitType); it regenerates faster and cannot act until it stands up
        self.sitting = False
        #: real time until which our own cast is still running
        self.cast_until = 0.0
        #: real time of the last damage we dealt (stuck-target detection)
        self.last_hit_at = 0.0
        #: object id -> real time until which the monster is ignored (unreachable)
        self.ignored: dict[int, float] = {}
        self._ignored_entries: dict[int, dict] = {}
        self.teleports = 0
        #: real time of the last "You have failed to pick up ..." (the drop belongs to another player)
        self.pickup_failed_at = 0.0
        #: where the server put the character (its farm zone); roaming stays around it
        self.home: tuple[int, int, int] | None = None

        self.has_user_info = False
        #: opcodes of packets that failed to parse (logged once each)
        self._bad_packets: set[int] = set()
        self.has_skills = False
        self.has_items = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """The character data needed for observations has arrived."""
        return self.has_user_info and self.has_skills

    @property
    def xy(self) -> tuple[int, int]:
        return self.pos[0], self.pos[1]

    def target_alive(self) -> bool:
        return bool(self.target_id) and self.target_id in self.char.radar.entries

    def distance_to(self, x: int, y: int) -> float:
        return math.hypot(self.pos[0] - x, self.pos[1] - y)

    def ignore_target(self, object_id: int, seconds: float, now: float | None = None) -> None:
        """Drop a monster from the radar and keep it off for ``seconds``."""
        now = time.time() if now is None else now
        self.ignored[object_id] = now + seconds
        entry = self.char.radar.entries.get(object_id)
        if entry is not None:
            self._ignored_entries[object_id] = entry  # back on the radar if it attacks the character
        self.char.radar.remove(object_id)
        if self.target_id == object_id:
            self.target_id = 0

    # ------------------------------------------------------------------
    # Packet handling
    # ------------------------------------------------------------------

    def handle(self, body: bytes, now: float | None = None) -> list[str]:
        """Apply one server packet and return the signals it produced (a packet that does not parse is skipped)."""
        try:
            return self._handle(body, time.time() if now is None else now)
        except struct.error as exc:
            # A packet layout the parser does not expect must not end the actor: log it once per opcode.
            pid = body[0] if body else -1
            if pid not in self._bad_packets:
                self._bad_packets.add(pid)
                log.warning("Skipped packet 0x%02X (%d bytes) that does not parse: %s; head %s", pid, len(body), exc,
                            body[:48].hex())
            return []

    def _handle(self, body: bytes, now: float) -> list[str]:
        pid = body[0]
        char = self.char

        if pid == GameServerPacket.USER_INFO:
            char.apply_user_info(packets.parse_user_info(body))
            self.pos = (char.x, char.y, char.z)
            self.has_user_info = True

        elif pid == GameServerPacket.STATUS_UPDATE:
            su = packets.parse_status_update(body)
            char.radar.update_from_status(su)
            CombatPolicy.update_self_status(char, su)

        elif pid == GameServerPacket.NPC_INFO:
            npc = packets.parse_npc_info(body)
            if npc and self.ignored.get(npc["object_id"], 0.0) <= now:
                char.radar.add_or_update(npc)

        elif pid == GameServerPacket.DELETE_OBJECT:
            obj = packets.parse_delete_object(body)
            if obj:
                oid = obj["object_id"]
                char.radar.remove(oid)
                self._ignored_entries.pop(oid, None)
                self.pending_loot = [item for item in self.pending_loot if item["object_id"] != oid]
                if oid == self.target_id:
                    self.target_id = 0

        elif pid == GameServerPacket.DIE:
            return self._on_die(packets.parse_die(body), now)

        elif pid == GameServerPacket.REVIVE:
            revived = packets.parse_revive(body)
            if revived and revived["object_id"] == char.object_id and self.dead:
                self.dead = False
                return ["revive"]

        elif pid == GameServerPacket.TELEPORT_TO_LOCATION:
            tp = packets.parse_teleport_to_location(body)
            if tp and tp["object_id"] == char.object_id:
                self._send(packets.build_appearing())  # finishes the teleport on the server
                self.pos = (tp["x"], tp["y"], tp["z"])
                self.home = self.pos
                char.x, char.y, char.z = self.pos
                char.radar.entries.clear()
                self.pending_loot.clear()
                self.target_id = 0
                self.sitting = False
                self.teleports += 1
                return ["teleport"]

        elif pid in (GameServerPacket.SPAWN_ITEM, GameServerPacket.DROP_ITEM):
            item = (packets.parse_spawn_item if pid == GameServerPacket.SPAWN_ITEM else packets.parse_drop_item)(body)
            if item:
                # the server sends the drop before the Die of the monster: it is ours while that is our target
                pet_target = char.pet.get("target_id") if char.pet else None
                item["mine"] = bool(item.get("dropper_id")) and item["dropper_id"] in (self.target_id, pet_target)
                self.pending_loot.append(item)

        elif pid == GameServerPacket.GET_ITEM:
            got = packets.parse_get_item(body)
            if got:
                self.pending_loot = [item for item in self.pending_loot if item["object_id"] != got["object_id"]]
                if got["picker_id"] == char.object_id:
                    self.events.items_picked += 1

        elif pid == GameServerPacket.ITEM_LIST:
            char.apply_item_list(packets.parse_item_list(body))
            self.has_items = True

        elif pid == GameServerPacket.CHANGE_WAIT_TYPE:
            wait = packets.parse_change_wait_type(body)
            if wait and wait["object_id"] == char.object_id and wait["wait_type"] in (0, 1):
                self.sitting = wait["wait_type"] == 0

        elif pid == GameServerPacket.INVENTORY_UPDATE:
            char.apply_inventory_update(packets.parse_inventory_update(body))

        elif pid == GameServerPacket.SKILL_LIST:
            char.apply_skill_list(packets.parse_skill_list(body))
            self.has_skills = True
            return ["skills_changed"]

        elif pid == GameServerPacket.SKILL_COOL_TIME:
            self.encoder.apply_skill_cool_time(packets.parse_skill_cool_time(body), now)

        elif pid == GameServerPacket.ATTACK:
            attack = packets.parse_attack(body)
            if attack:
                pet = char.pet
                self._note_attack_on_me(attack["attacker_id"], [hit[0] for hit in attack["hits"]], now)
                if pet and attack["attacker_id"] == pet["object_id"]:
                    pet["target_id"] = attack["hits"][0][0]
                    pet["attacking_until"] = now + PET_ATTACK_MEMORY_S / self.server_speed
                else:
                    self._note_other_attack(attack["attacker_id"], [hit[0] for hit in attack["hits"]], now)

        elif pid == GameServerPacket.MAGIC_SKILL_USE:
            cast = packets.parse_magic_skill_use(body)
            pet = char.pet
            if cast:
                self._note_attack_on_me(cast["caster_id"], [cast["target_id"]], now)
            if cast and pet and cast["caster_id"] == pet["object_id"]:
                pet["casting_until"] = now + cast["hit_time"] / 1000.0 / self.server_speed + CAST_MARGIN_S
                if cast["target_id"] != char.object_id and cast["target_id"] != pet["object_id"]:
                    pet["target_id"] = cast["target_id"]
                self.encoder.note_pet_skill_used(cast["skill_id"], now)
            elif cast and cast["caster_id"] != char.object_id:
                self._note_other_attack(cast["caster_id"], [cast["target_id"]], now)
            if cast and cast["caster_id"] == char.object_id:
                self.cast_until = now + cast["hit_time"] / 1000.0 / self.server_speed + CAST_MARGIN_S

        elif pid == GameServerPacket.SYSTEM_MESSAGE:
            self._on_system_message(packets.parse_system_message(body), now)

        elif pid == GameServerPacket.ABNORMAL_STATUS_UPDATE:
            char.apply_abnormal_status_update(packets.parse_abnormal_status_update(body))

        elif pid == GameServerPacket.PET_INFO:
            return self._on_pet_info(packets.parse_pet_info(body))

        elif pid == GameServerPacket.PET_STATUS_UPDATE:
            status = packets.parse_pet_status_update(body)
            if status and char.pet and status["object_id"] == char.pet["object_id"]:
                char.pet.update({key: status[key] for key in ("x", "y", "z", "cur_hp", "max_hp", "cur_mp", "max_mp")})

        elif pid == GameServerPacket.PET_DELETE:
            gone = packets.parse_pet_delete(body)
            if gone and char.pet and gone["object_id"] == char.pet["object_id"]:
                char.pet = None
                return ["unsummoned"]

        return []

    def _on_pet_info(self, info: dict) -> list[str]:
        """A summon appeared (or its state changed): keep what the packets do not repeat."""
        if not info:
            return []
        pet = self.char.pet
        if pet and pet["object_id"] == info["object_id"]:
            pet.update(info)
            return []
        self.char.pet = {**info, "target_id": 0, "attacking_until": 0.0, "casting_until": 0.0}
        return ["summoned"]

    def _on_die(self, die: dict, now: float) -> list[str]:
        if not die:
            return []
        oid, char = die["object_id"], self.char
        pet = char.pet
        if pet and oid == pet["object_id"]:
            pet["dead"] = True
            self.events.pet_deaths += 1
            return ["pet_died"]
        if pet and oid == pet.get("target_id") and oid != self.target_id:
            # The summon finished a monster it was ordered at (not the character's target): a kill too.
            self.encoder.record_event(char, self.target_id, self.xy, now)
            self.events.kills += 1
            char.radar.remove(oid)
            pet["target_id"] = 0
            return ["kill"]
        if oid == char.object_id:
            self.dead = True
            self.sitting = False
            self.dead_since = now
            self.target_id = 0
            self.pending_loot.clear()
            self.events.deaths += 1
            return ["death"]
        if oid == self.target_id:
            # The training data logs the kill with the dying target still on the radar.
            self.encoder.record_event(char, self.target_id, self.xy, now)
            self.events.kills += 1
            char.radar.remove(oid)
            self.target_id = 0
            return ["kill"]
        char.radar.remove(oid)
        return []

    def _note_attack_on_me(self, attacker_id: int, target_ids: list[int], now: float) -> None:
        """A monster on the radar attacked the character: it counts as attacking it for a while."""
        mob = self.char.radar.entries.get(attacker_id)
        if mob is None and self.char.object_id in target_ids and attacker_id in self._ignored_entries:
            # an ignored monster attacks: it is a fight again, whatever made it ignored
            self.ignored.pop(attacker_id, None)
            mob = self._ignored_entries.pop(attacker_id)
            self.char.radar.entries[attacker_id] = mob
        if mob is not None and self.char.object_id in target_ids:
            mob["attacking_me_until"] = now + ATTACKER_MEMORY_S / self.server_speed

    def _note_other_attack(self, attacker_id: int, target_ids: list[int], now: float) -> None:
        """
        Remember that another player attacks these monsters (the ``by_other`` mob feature): the
        attacker is not this character and not a monster on the radar (a monster hitting a player).
        """
        if attacker_id == self.char.object_id or attacker_id in self.char.radar.entries:
            return
        until = now + OTHER_ATTACKER_MEMORY_S / self.server_speed
        for target_id in target_ids:
            mob = self.char.radar.entries.get(target_id)
            if mob is not None:
                mob["by_other_until"] = until

    def _target_max_hp(self) -> float:
        """Maximum HP of the current target (standard server): the StatusUpdate, else the datapack's base HP."""
        return self._mob_max_hp(self.target_id)

    def _mob_max_hp(self, object_id: int | None) -> float:
        mob = self.char.radar.entries.get(object_id) if object_id else None
        if mob is None:
            return 0.0
        return float(mob.get("max_hp") or npc_data.get_npc_max_hp(mob.get("npc_id", 0)))

    def _on_system_message(self, msg: dict, now: float) -> None:
        if not msg:
            return
        params = msg["params"]
        if msg["msg_id"] in (SystemMessageId.YOU_DID_DAMAGE, SystemMessageId.PET_DID_DAMAGE) and params \
                and isinstance(params[0], int):
            damage = params[0]
            self.events.damage_dealt += damage
            if len(params) >= 4 and all(isinstance(p, int) for p in params[1:4]):
                # RL-environment server: the message also names the monster hit and its HP (a polearm or an
                # area skill hits several monsters, one message each).
                victim, cur_hp, max_hp = params[1:4]
                mob = self.char.radar.entries.get(victim)
                if mob is not None:
                    mob["cur_hp"], mob["max_hp"] = cur_hp, max_hp
            else:  # standard server: assume the current target (the summon's, for its damage)
                pet = self.char.pet
                max_hp = (self._mob_max_hp(pet.get("target_id")) if msg["msg_id"] == SystemMessageId.PET_DID_DAMAGE and pet
                          else self._target_max_hp())
            if max_hp > 0:
                self.events.damage_dealt_share += min(1.0, damage / max_hp)
            self.last_hit_at = now
        elif msg["msg_id"] in PICKUP_FAILED:
            self.pickup_failed_at = now  # the item is someone else's (drop protection)
            return
        elif msg["msg_id"] in (SystemMessageId.DAMAGE_TAKEN, SystemMessageId.DAMAGE_TAKEN_2) and len(params) > 1 \
                and isinstance(params[1], int):
            self.events.damage_taken += params[1]
        else:
            return
        self.encoder.note_combat(now)
        self.encoder.record_event(self.char, self.target_id, self.xy, now)
