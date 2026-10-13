"""
Lineage2 C6 Interlude -- login + game-server connection for headless bots.

:func:`login` runs the login-server handshake and returns the ``PlayOk`` keys,
:class:`GameConnection` enters the world with one character slot and then exposes a non-blocking
packet pump (:meth:`GameConnection.poll`) that also answers the protocol housekeeping on its own:
GameGuard queries and the client keep-alive ping.

The handshake is the same as in ``client_demo.py``; this module is the reusable version for the RL
actors (``l2_actor``), which must not block on a quiet socket between two decisions.
"""

from __future__ import annotations

import logging
import select
import socket
import struct
import time
from dataclasses import dataclass

from . import crypto, packets
from .packets import GameServerPacket, LoginServerPacket

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 746
#: The client pings the server this often (the server never pings first).
PING_INTERVAL_S = 5.0


class ConnectionFailed(RuntimeError):
    """The server refused or broke the handshake."""


@dataclass(frozen=True)
class LoginTicket:
    """What the login server hands out for entering a game server."""

    account: str
    login_ok1: int
    login_ok2: int
    play_ok1: int
    play_ok2: int
    game_ip: str
    game_port: int


def login(server_ip: str, account: str, password: str, *, port: int = 2106, server_index: int = 0,
          timeout: float = 10.0) -> LoginTicket:
    """
    Log in and pick a game server.

    With ``AutoCreateAccounts = True`` on the login server an unknown account is created on the fly.
    """
    sock = socket.create_connection((server_ip, port), timeout=timeout)
    try:
        init = crypto.decrypt_init_packet(crypto.recv_packet(sock), crypto.L2MOBIUS_STATIC_KEY)
        if init[0] != LoginServerPacket.INIT:
            raise ConnectionFailed(f"unexpected login packet 0x{init[0]:02X} instead of Init")
        session_id = struct.unpack_from("<I", init, 1)[0]
        rsa_modulus = bytearray(init[9:9 + 128])
        crypto.unscramble_modulus(rsa_modulus)
        key = init[9 + 128 + 16:9 + 128 + 16 + 16]

        crypto.send_encrypted(sock, packets.build_auth_game_guard(session_id), key)
        crypto.decrypt_l2_packet(crypto.recv_packet(sock), key)

        crypto.send_encrypted(sock, packets.build_request_auth_login(account, password, session_id, rsa_modulus), key)
        reply = crypto.decrypt_l2_packet(crypto.recv_packet(sock), key)
        if reply[0] == LoginServerPacket.LOGIN_FAIL:
            raise ConnectionFailed(f"login failed for '{account}' (reason {reply[1]})")
        if reply[0] != LoginServerPacket.LOGIN_OK:
            raise ConnectionFailed(f"unexpected login packet 0x{reply[0]:02X} instead of LoginOk")
        login_ok1, login_ok2 = struct.unpack_from("<II", reply, 1)

        crypto.send_encrypted(sock, packets.build_request_server_list(login_ok1, login_ok2), key)
        servers = _parse_server_list(crypto.decrypt_l2_packet(crypto.recv_packet(sock), key))
        if server_index >= len(servers):
            raise ConnectionFailed(f"server index {server_index} not in the list ({len(servers)} server(s))")
        server = servers[server_index]

        crypto.send_encrypted(sock, packets.build_request_server_login(login_ok1, login_ok2, server["id"]), key)
        play = crypto.decrypt_l2_packet(crypto.recv_packet(sock), key)
        if play[0] != LoginServerPacket.PLAY_OK:
            raise ConnectionFailed(f"game server login refused (packet 0x{play[0]:02X})")
        play_ok1, play_ok2 = struct.unpack_from("<II", play, 1)
        log.info("Logged in as '%s', game server %s:%d", account, server["ip"], server["port"])
        return LoginTicket(account, login_ok1, login_ok2, play_ok1, play_ok2, server["ip"], server["port"])
    finally:
        sock.close()


def _parse_server_list(body: bytes) -> list[dict]:
    if body[0] != LoginServerPacket.SERVER_LIST:
        raise ConnectionFailed(f"unexpected login packet 0x{body[0]:02X} instead of ServerList")
    servers, pos = [], 3
    for _ in range(body[1]):
        server_id = body[pos]; pos += 1
        ip = body[pos:pos + 4]; pos += 4
        port = struct.unpack_from("<I", body, pos)[0]; pos += 4
        pos += 1 + 1 + 2 + 2 + 1 + 4 + 1  # age limit, pvp, online, max online, status, type, brackets
        servers.append({"id": server_id, "ip": ".".join(str(b) for b in ip), "port": port})
    return servers


class GameConnection:
    """A game-server session of one character."""

    def __init__(self, ticket: LoginTicket, *, char_slot: int = 0, timeout: float = 30.0):
        self.ticket = ticket
        self.char_slot = char_slot
        self.character_name: str | None = None
        self._sock = socket.create_connection((ticket.game_ip, ticket.game_port), timeout=timeout)
        self._sock.settimeout(timeout)
        self._ping_counter = 0
        self._last_ping = 0.0

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def enter_world(self) -> None:
        """Authenticate, select ``char_slot`` and send EnterWorld."""
        sock, t = self._sock, self.ticket
        crypto.send_raw(sock, packets.build_proto_version(PROTOCOL_VERSION))
        if crypto.recv_packet(sock)[0] != GameServerPacket.KEY_PACKET:
            raise ConnectionFailed("no KeyPacket from the game server")

        crypto.send_raw(sock, packets.build_auth_login_gs(t.account, t.play_ok2, t.play_ok1, t.login_ok1, t.login_ok2))
        info = crypto.recv_packet(sock)
        if info[0] == GameServerPacket.LOGIN_FAIL:
            raise ConnectionFailed(f"game server authentication failed (reason {info[1]})")
        if info[0] != GameServerPacket.CHAR_SELECT_INFO:
            raise ConnectionFailed(f"unexpected packet 0x{info[0]:02X} instead of CharSelectInfo")
        count = struct.unpack_from("<I", info, 1)[0]
        if count <= self.char_slot:
            raise ConnectionFailed(f"account '{t.account}' has {count} character(s), slot {self.char_slot} is empty "
                                   "(an RL-environment server creates one for agent accounts)")
        self.character_name = self._slot_name(info, self.char_slot)

        crypto.send_raw(sock, packets.build_character_select(self.char_slot))
        for _ in range(50):  # the server may send e.g. a StatusUpdate before CharSelected
            pid = crypto.recv_packet(sock)[0]
            if pid == GameServerPacket.CHAR_SELECTED:
                break
            if pid in (GameServerPacket.SERVER_CLOSE, GameServerPacket.LOGIN_FAIL):
                raise ConnectionFailed(f"character selection refused ({packets.PACKET_NAMES.get(pid)}) -- "
                                       "is the character still logged in?")
        else:
            raise ConnectionFailed("CharSelected not received")

        crypto.send_raw(sock, packets.build_enter_world())
        self._last_ping = time.time()
        log.info("Entered the world as '%s' (slot %d)", self.character_name, self.char_slot)

    @staticmethod
    def _slot_name(info: bytes, slot: int) -> str:
        """Name of a CharSelectInfo slot (each entry: name, object id, then fixed-size data)."""
        pos = 5
        for index in range(slot + 1):
            name, pos = packets._read_utf16le_string(info, pos)
            if index == slot:
                return name
            pos += 4  # object id
            _, pos = packets._read_utf16le_string(info, pos)  # account name
            pos += 200
        return ""

    # ------------------------------------------------------------------
    # Packet pump
    # ------------------------------------------------------------------

    def poll(self, timeout: float) -> list[bytes]:
        """
        Return every packet that arrives within ``timeout`` seconds (may be empty).

        Waits with ``select`` and only then reads, so a quiet socket never blocks and a packet is
        never cut in half. GameGuard queries are answered here and the keep-alive ping is sent when due.
        """
        received: list[bytes] = []
        wait = timeout
        while select.select([self._sock], [], [], wait)[0]:
            body = crypto.recv_packet(self._sock)
            wait = 0.0  # drain what is already buffered, then return
            if not body:
                continue
            if body[0] == GameServerPacket.GAME_GUARD_QUERY:
                self.send(packets.build_game_guard_reply())
                continue
            if body[0] == GameServerPacket.SERVER_CLOSE:
                raise ConnectionError("server closed the connection")
            received.append(body)

        if time.time() - self._last_ping >= PING_INTERVAL_S:
            self.send(packets.build_net_ping(self._ping_counter))
            self._ping_counter += 1
            self._last_ping = time.time()
        return received

    @property
    def sock(self) -> socket.socket:
        """The game-server socket, for callers that run their own receive loop (``client_demo.py``)."""
        return self._sock

    def send(self, data: bytes) -> None:
        crypto.send_raw(self._sock, data)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
