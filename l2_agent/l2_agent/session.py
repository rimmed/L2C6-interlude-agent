"""
One logged-in character: the connection, the combat world state (``l2_actor.world.WorldState``) and what the
non-combat parts of the agent need on top of it: every NPC in sight (the combat radar keeps monsters only), the
character's own movement, the last dialog, trade lists and the quest list.

Everything runs in one thread: :meth:`Session.pump` reads the packets, the agent's skills call it while they wait.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from l2_actor.executor import ActionExecutor, ExecutorConfig
from l2_actor.world import WorldState
from l2_client import packets
from l2_client.connection import GameConnection, login
from l2_client.packets import GameServerPacket as G
from l2_client.policy import CombatPolicy
from l2_client.tools.data import item_data, npc_data, skill_data

from .knowledge import Knowledge, Point

log = logging.getLogger(__name__)

#: The combat model's encoder still reads its features from the datapack XML (the same code as in training);
#: the agent's own knowledge comes from Parquet.
DATAPACK = Path(__file__).resolve().parents[2] / "L2J_Mobius" / "L2J_Mobius_CT_0_Interlude" / "dist" / "game" / "data" / "stats"
STOP_MOVE = 0x47
MOVE_TO_PAWN = 0x60
VALIDATE_LOCATION = 0x61


@dataclass
class Npc:
    """An NPC or monster in sight (NpcInfo)."""

    object_id: int
    npc_id: int
    x: int
    y: int
    z: int
    attackable: bool
    dead: bool
    seen_at: float


@dataclass
class Motion:
    """The character's last move: from ``origin`` to ``destination``, started at ``started`` (real time)."""

    origin: tuple[int, int, int]
    destination: tuple[int, int, int]
    started: float
    speed: float  # units per real second

    def position(self, now: float) -> tuple[int, int, int]:
        ox, oy, oz = self.origin
        dx, dy, dz = (d - o for d, o in zip(self.destination, self.origin))
        length = math.hypot(dx, dy)
        if length < 1 or self.speed <= 0:
            return self.destination
        share = min(1.0, (now - self.started) * self.speed / length)
        return int(ox + dx * share), int(oy + dy * share), int(oz + dz * share)

    def arrived(self, now: float) -> bool:
        return self.position(now) == self.destination


@dataclass
class Dialog:
    """The last NpcHtmlMessage."""

    npc_object_id: int
    html: str
    received: float
    links: list["Link"] = field(default_factory=list)


@dataclass(frozen=True)
class Link:
    """A link of a dialog: ``kind`` is ``bypass`` (RequestBypassToServer) or ``link`` (RequestLinkHtml)."""

    kind: str
    command: str
    text: str


@dataclass
class SessionConfig:
    """Server, account and model of a session."""

    server_ip: str = "192.168.1.200"
    login_port: int = 2106
    account: str = "aiagent01"
    password: str = "aiagent01pw"
    char_slot: int = 0
    #: the character created when the account has none (``packets.build_character_create`` arguments)
    create: dict | None = None
    model: str = ""
    server_speed: float = 1.0
    device: str = "cpu"
    threads: int = 1


class Session:
    """A character in the world."""

    def __init__(self, cfg: SessionConfig, knowledge: Knowledge | None = None):
        self.cfg = cfg
        self.knowledge = knowledge or Knowledge()
        self.encoder: CombatPolicy | None = None
        self.conn: GameConnection | None = None
        self.world: WorldState | None = None
        self.executor: ActionExecutor | None = None
        self.npcs: dict[int, Npc] = {}
        self.motion: Motion | None = None
        self.dialog: Dialog | None = None
        self.buy_list: dict | None = None
        self.sell_list: dict | None = None
        self.quests: dict[int, int] = {}
        self.quests_at = 0.0
        self.system_messages: list[tuple[float, int]] = []
        #: callbacks (packet body, now) run for every packet, e.g. the tutorial handler
        self.listeners: list[Callable[[bytes, float], None]] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, ready_timeout: float = 30.0) -> None:
        cfg = self.cfg
        npc_data.load_npc_names(str(DATAPACK / "npcs"))
        item_data.load_item_data(str(DATAPACK / "items"))
        skill_data.load_skill_data(str(DATAPACK / "skills"))
        self.encoder = CombatPolicy(cfg.model, server_speed=cfg.server_speed, device=cfg.device, threads=cfg.threads)
        ticket = login(cfg.server_ip, cfg.account, cfg.password, port=cfg.login_port)
        self.conn = GameConnection(ticket, char_slot=cfg.char_slot, create=cfg.create)
        self.conn.enter_world()
        self.world = WorldState(self.encoder, self.conn.send, cfg.server_speed)
        # No .heal: the agent plays by the rules and regenerates sitting.
        self.executor = ActionExecutor(self.conn.send, self.world, self.encoder, ExecutorConfig(rest_heal=False))
        if not self.wait_until(lambda: self.world.ready, ready_timeout):
            raise TimeoutError("no UserInfo / SkillList from the server")
        self.encoder.bind_character(self.world.char)
        self.world.home = self.world.pos
        self.send(packets.build_request_quest_list())
        char = self.world.char
        log.info("In the world as '%s': level %d, class %d, at %s", self.conn.character_name, char.level,
                 char.class_id, self.world.pos)

    def send(self, packet: bytes) -> None:
        self.conn.send(packet)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

    # ------------------------------------------------------------------
    # Packets
    # ------------------------------------------------------------------

    def pump(self, timeout: float = 0.05) -> None:
        for body in self.conn.poll(timeout):
            now = time.time()  # after the wait: an answer must not look older than the request it answers
            self.world.handle(body, now)
            self._handle(body, now)
            for listener in self.listeners:
                listener(body, now)

    def wait_until(self, predicate: Callable[[], bool], timeout: float, step: float = 0.1) -> bool:
        deadline = time.time() + timeout
        while not predicate():
            if time.time() >= deadline:
                return False
            self.pump(step)
        return True

    def idle(self, seconds: float) -> None:
        self.wait_until(lambda: False, seconds)

    def _handle(self, body: bytes, now: float) -> None:
        pid = body[0]
        char = self.world.char
        if pid == G.NPC_INFO:
            npc = packets.parse_npc_info(body)
            if npc:
                did = npc["display_id"]
                self.npcs[npc["object_id"]] = Npc(npc["object_id"], did - 1000000 if did > 1000000 else did, npc["x"],
                                                  npc["y"], npc["z"], npc["is_attackable"], bool(npc.get("is_alike_dead")), now)
        elif pid == G.DELETE_OBJECT:
            gone = packets.parse_delete_object(body)
            if gone:
                self.npcs.pop(gone["object_id"], None)
        elif pid == G.DIE:
            die = packets.parse_die(body)
            if die and die["object_id"] in self.npcs:
                self.npcs[die["object_id"]].dead = True
        elif pid == G.CHAR_MOVE_TO_LOCATION and len(body) >= 29:
            oid, tx, ty, tz, ox, oy, oz = struct.unpack_from("<iiiiiii", body, 1)
            if oid == char.object_id:
                speed = float(getattr(char, "run_spd", 0) or 120) * float(getattr(char, "move_multiplier", 1.0) or 1.0)
                self.motion = Motion((ox, oy, oz), (tx, ty, tz), now, speed * self.cfg.server_speed)
            elif oid in self.npcs:
                npc = self.npcs[oid]
                npc.x, npc.y, npc.z = tx, ty, tz
        elif pid == MOVE_TO_PAWN and len(body) >= 37:
            # The server is to walk the character to an NPC or monster it clicked. Where it really goes comes in the
            # CharMoveToLocation legs that follow (a blocked way leaves it short of the target); this packet only
            # tells where the server has the character now.
            oid, _target, _reach, x, y, z = struct.unpack_from("<iiiiii", body, 1)
            if oid == char.object_id:
                self.motion = None
                self.world.pos = (x, y, z)
                char.x, char.y, char.z = x, y, z
        elif pid in (STOP_MOVE, VALIDATE_LOCATION) and len(body) >= 17:
            oid, x, y, z = struct.unpack_from("<iiii", body, 1)
            if oid == char.object_id:
                self.motion = None
                self.world.pos = (x, y, z)
                char.x, char.y, char.z = x, y, z
        elif pid == G.TELEPORT_TO_LOCATION:
            self.motion = None
            self.npcs.clear()
        elif pid == G.NPC_HTML_MESSAGE:
            msg = packets.parse_npc_html_message(body)
            self.dialog = Dialog(msg["npc_object_id"], msg["html"], now, parse_links(msg["html"]))
        elif pid == G.BUY_LIST:
            self.buy_list = {**packets.parse_buy_list(body), "received": now}
        elif pid == G.SELL_LIST:
            self.sell_list = {**packets.parse_sell_list(body), "received": now}
        elif pid == G.QUEST_LIST:
            self.quests = packets.parse_quest_list(body)["quests"]
            self.quests_at = now
        elif pid == G.SYSTEM_MESSAGE:
            msg = packets.parse_system_message(body)
            if msg:
                self.system_messages.append((now, msg.get("msg_id", 0)))
                del self.system_messages[:-50]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def char(self):
        return self.world.char

    @property
    def position(self) -> tuple[int, int, int]:
        """Where the character is now: along its last move, else the last known spot."""
        if self.motion is not None:
            pos = self.motion.position(time.time())
            self.world.pos = pos
            return pos
        return self.world.pos

    def distance(self, x: float, y: float) -> float:
        px, py, _ = self.position
        return math.hypot(px - x, py - y)

    def find_npc(self, npc_id: int, near: Point | None = None) -> Npc | None:
        """The nearest living NPC of template ``npc_id`` in sight."""
        found = [n for n in self.npcs.values() if n.npc_id == npc_id and not n.dead]
        if not found:
            return None
        x, y = (near.x, near.y) if near else self.position[:2]
        return min(found, key=lambda n: math.hypot(n.x - x, n.y - y))

    def item_count(self, item_id: int) -> int:
        return sum(int(i.get("count", 0)) for i in self.char.items if i.get("item_id") == item_id)

    def items_of(self, item_id: int) -> list[dict]:
        return [i for i in self.char.items if i.get("item_id") == item_id]

    @property
    def adena(self) -> int:
        return self.item_count(57)


def parse_links(html: str) -> list[Link]:
    """The links of a dialog in their order: ``<a action="bypass -h X">text</a>`` and ``<a action="link X">``."""
    import re

    links = []
    for action, text in re.findall(r'<a\s+[^>]*action="([^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
        text = re.sub(r"<[^>]+>", "", text).strip()
        if action.startswith("bypass"):
            command = re.sub(r"^bypass\s+(-h\s+)?", "", action)
            links.append(Link("bypass", command, text))
        elif action.startswith("link"):
            links.append(Link("link", action[4:].strip(), text))
    return links
