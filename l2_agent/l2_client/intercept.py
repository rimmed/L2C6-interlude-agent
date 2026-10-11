#!/usr/bin/env python3
"""
Lineage2 C6 Interlude -- TCP proxy relay with packet dumping and command injection.

The MIT License (MIT)

Copyright (c) 2026 Viacheslav Reutskyi <viacheslav.reutskyi@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

USAGE

Start the proxy **before** launching the game client (from ``l2_agent/``)::

    uv run l2_client/intercept.py

The proxy listens on two ports:

* ``:2106`` -- login server relay -> ``127.0.0.1:2106``
* ``:7776`` -- game server relay -> ``127.0.0.1:7777``

Configure your game client (or ``l2.ini`` / hosts file) to connect to
``127.0.0.1`` instead of the real server.  Every packet passing through is
printed with its direction (C->S or S->C) and, for the game server, its name
from the opcode enums in :mod:`l2_client.packets`.  Login server traffic is
Blowfish-encrypted, so its packets are only printed with their size.

COMMANDS
--------

The REPL prompt ``[CMD] >>`` accepts the following commands:

    say <message>
        Broadcast *message* in public chat.

    move <x> <y> <z>
        Move the character to the given coordinates.

    attack <targetObjId> [x y z] [--shift]
        Attack *targetObjId*.  If coordinates are omitted the last known
        position is used.  ``--shift`` sends a shift-click attack.

    pickup <objectId> [x y z]
        Pick up the ground item identified by *objectId*.

    help
        Show the command reference.

    exit
        Shut down the proxy.

CONFIGURATION
-------------

Edit the constants at the top of this file:

* ``LOGIN_LOCAL_PORT`` / ``LOGIN_REMOTE_HOST`` / ``LOGIN_REMOTE_PORT``
* ``GAME_LOCAL_PORT`` / ``GAME_REMOTE_HOST`` / ``GAME_REMOTE_PORT``
"""

import select
import socket
import struct
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path

if __package__:
    from . import packets
else:
    # Run as a script (``uv run l2_client/intercept.py``): make ``l2_client`` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from l2_client import packets

# --- CONFIGURATION -------------------------------------------------------

LOGIN_LOCAL_PORT  = 2106
LOGIN_REMOTE_HOST = "127.0.0.1"
LOGIN_REMOTE_PORT = 2106

GAME_LOCAL_PORT   = 7776
GAME_REMOTE_HOST  = "127.0.0.1"
GAME_REMOTE_PORT  = 7777

BUFFER_SIZE = 65536

# --- Packet dump -------------------------------------------------------

def _packet_name(opcodes: type[IntEnum], pid: int) -> str:
    """Name of opcode *pid* in *opcodes*, or its hex value when the enum does not know it."""
    try:
        return opcodes(pid).name
    except ValueError:
        return f"0x{pid:02X}"


def _dump_packets(data: bytes, label: str, direction: str, opcodes: type[IntEnum] | None):
    """
    Print every complete packet in *data*.

    :param opcodes: opcode enum of this direction, or ``None`` for encrypted traffic, whose
        first body byte is ciphertext rather than an opcode
    """
    while len(data) >= 2:
        sz = struct.unpack_from("<H", data)[0]
        if sz < 2 or len(data) < sz:
            break
        body = data[2:sz]
        if opcodes is None:
            name = "(encrypted)"
        else:
            name = _packet_name(opcodes, body[0]) if body else "(empty)"
        print(f"[RELAY {label}] {direction} {name}  {sz}B")
        data = data[sz:]


# --- RELAY ---------------------------------------------------------------

def relay(src: socket.socket, dst: socket.socket, label: str,
          c2s_opcodes: type[IntEnum] | None, s2c_opcodes: type[IntEnum] | None):
    """
    Bidirectional copier between client *src* and server *dst* that dumps every packet.

    :param c2s_opcodes: opcode enum for client -> server packets (``None`` = encrypted)
    :param s2c_opcodes: opcode enum for server -> client packets (``None`` = encrypted)
    """

    def copy_one_way(reader, writer, direction, opcodes):
        try:
            data = reader.recv(BUFFER_SIZE)
        except Exception:
            return False
        if not data:
            return False
        _dump_packets(data, label, direction, opcodes)
        try:
            writer.sendall(data)
        except Exception:
            return False
        return True

    while True:
        r, _, _ = select.select([src, dst], [], [], 1.0)
        if src in r:
            if not copy_one_way(src, dst, "C->S", c2s_opcodes):
                break
        if dst in r:
            if not copy_one_way(dst, src, "S->C", s2c_opcodes):
                break

    print(f"[RELAY {label}]  Connection closed")
    src.close()
    dst.close()


# --- Login listener ------------------------------------------------------

def login_listener():
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", LOGIN_LOCAL_PORT))
    ls.listen(1)
    print(f"[RELAY LOGIN]  Listening on :{LOGIN_LOCAL_PORT} -> {LOGIN_REMOTE_HOST}:{LOGIN_REMOTE_PORT}")
    while True:
        client, addr = ls.accept()
        print(f"[RELAY LOGIN]  Client connected from {addr}")
        remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote.connect((LOGIN_REMOTE_HOST, LOGIN_REMOTE_PORT))
        print("[RELAY LOGIN]  Connected to remote")
        # Login traffic is Blowfish-encrypted end to end, so no opcode can be read from it here.
        threading.Thread(target=relay, args=(client, remote, "LOGIN", None, None), daemon=True).start()


# --- Game listener (with injection support) -----------------------------

_game_upstream: socket.socket | None = None   # client -> server socket
_game_inject_lock = threading.Lock()


def game_listener():
    global _game_upstream
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", GAME_LOCAL_PORT))
    ls.listen(1)
    print(f"[RELAY GAME]   Listening on :{GAME_LOCAL_PORT} -> {GAME_REMOTE_HOST}:{GAME_REMOTE_PORT}")
    while True:
        client, addr = ls.accept()
        print(f"[RELAY GAME]   Client connected from {addr}")
        remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote.connect((GAME_REMOTE_HOST, GAME_REMOTE_PORT))
        print("[RELAY GAME]   Connected to remote")

        with _game_inject_lock:
            _game_upstream = remote   # injection target: remote = client->server

        threading.Thread(target=_relay_with_inject, args=(client, remote), daemon=True).start()


def _relay_with_inject(client: socket.socket, server_side: socket.socket):
    """Same as ``relay`` but checks ``_game_upstream`` for injected bytes."""
    while True:
        r, _, _ = select.select([client, server_side], [], [], 1.0)
        if client in r:
            try:
                data = client.recv(BUFFER_SIZE)
            except Exception:
                break
            if not data:
                break
            _dump_packets(data, "GAME", "C->S", packets.GameClientPacket)
            try:
                server_side.sendall(data)
            except Exception:
                break
        if server_side in r:
            try:
                data = server_side.recv(BUFFER_SIZE)
            except Exception:
                break
            if not data:
                break
            _dump_packets(data, "GAME", "S->C", packets.GameServerPacket)
            try:
                client.sendall(data)
            except Exception:
                break

    print("[RELAY GAME]   Connection closed")
    with _game_inject_lock:
        _game_upstream = None
    client.close()
    server_side.close()


# --- Packet injection ----------------------------------------------------

def inject_raw_data(data: bytes):
    """Queue *data* onto the game-server upstream connection."""
    with _game_inject_lock:
        if _game_upstream is None:
            print("[INJECT]  !! No game-server connection")
            return False
        try:
            _game_upstream.sendall(data)
            print(f"[INJECT]  -> {len(data)} bytes injected")
            return True
        except Exception as e:
            print(f"[INJECT]  !! Send error: {e}")
            return False


def parse_coords(arg: str) -> tuple[int, int, int]:
    parts = arg.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("need 3 integers: x y z")
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_int(arg: str) -> int:
    return int(arg)


CMDS_HELP = """
COMMANDS -------------------------------------------------------------------
  say <message>              Broadcast a message in public chat
  move <x> <y> <z>           Move to coordinates
  attack <targetObjId> [x y z]  Attack a target (uses current position if
                                 no coords given; shift-click with --shift)
  pickup <objectId> [x y z]  Pick up an item
  help                       Show this message
  exit                       Quit the proxy
---------------------------------------------------------------------------
"""


def command_loop():
    print(CMDS_HELP)
    cur_x, cur_y, cur_z = 0, 0, 0
    while True:
        try:
            line = input("[CMD] >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[CMD]  Shutting down")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "help":
                print(CMDS_HELP)

            elif cmd == "exit":
                print("[CMD]  Shutting down")
                break

            elif cmd == "say":
                msg = " ".join(parts[1:]) if len(parts) > 1 else ""
                if not msg:
                    print("[CMD]  Usage: say <message>")
                    continue
                pkt = packets.build_say2(msg)
                inject_raw_data(pkt)

            elif cmd == "move":
                if len(parts) < 4:
                    print("[CMD]  Usage: move <x> <y> <z>")
                    continue
                tx, ty, tz = parse_coords(" ".join(parts[1:4]))
                pkt = packets.build_move_to_location(tx, ty, tz, cur_x, cur_y, cur_z, mode=1)
                inject_raw_data(pkt)
                cur_x, cur_y, cur_z = tx, ty, tz

            elif cmd == "attack" or cmd == "a":
                if len(parts) < 2:
                    print("[CMD]  Usage: attack <targetObjId> [x y z] [--shift]")
                    continue
                target_id = parse_int(parts[1])
                shift = "--shift" in parts
                if len(parts) >= 4:
                    # coords given
                    remaining = [p for p in parts[2:] if not p.startswith("--")]
                    if len(remaining) >= 3:
                        x, y, z = parse_coords(" ".join(remaining[:3]))
                    else:
                        x, y, z = cur_x, cur_y, cur_z
                else:
                    x, y, z = cur_x, cur_y, cur_z
                pkt = packets.build_attack_request(target_id, x, y, z, shift_click=shift)
                inject_raw_data(pkt)

            elif cmd == "pickup" or cmd == "p":
                if len(parts) < 2:
                    print("[CMD]  Usage: pickup <objectId> [x y z]")
                    continue
                target_id = parse_int(parts[1])
                if len(parts) >= 5:
                    x, y, z = parse_coords(" ".join(parts[2:5]))
                else:
                    x, y, z = cur_x, cur_y, cur_z
                pkt = packets.build_action(target_id, x, y, z, shift_click=False)
                inject_raw_data(pkt)

            else:
                print(f"[CMD]  Unknown command: {cmd}")
                print("[CMD]  Type 'help' for available commands")

        except (ValueError, IndexError) as e:
            print(f"[CMD]  !! Parse error: {e}")
        except Exception as e:
            print(f"[CMD]  !! Error: {e}")


# --- MAIN ----------------------------------------------------------------

def main():
    print("=" * 70)
    print("  L2J Interlude Packet Relay + Command Injector")
    print("=" * 70)

    threading.Thread(target=login_listener, daemon=True).start()
    threading.Thread(target=game_listener, daemon=True).start()

    time.sleep(0.3)
    command_loop()

    print("\n[DONE]  Proxy stopped")


if __name__ == "__main__":
    main()
