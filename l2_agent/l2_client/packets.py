"""
Lineage2 C6 Interlude -- packet builders and parsers for all known server packets.

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
"""

import re
import struct
from enum import IntEnum

# ---------------------------------------------------------------------------
# Opcodes
#
# Names and values mirror the server sources (L2J_Mobius_CT_0_Interlude):
# gameserver/network/ServerPackets.java, ClientPackets.java, ExClientPackets.java and
# loginserver/network/LoginClientPackets.java + loginserver/network/serverpackets/*.java.
# Login and game server use separate opcode spaces, and so do both directions, so the same
# byte means different packets in different enums (e.g. 0xA8 is the client's RequestNetPing
# but the server's AskJoinAlliance).
# ---------------------------------------------------------------------------


class LoginServerPacket(IntEnum):
    """Login server -> client."""

    INIT = 0x00
    LOGIN_FAIL = 0x01
    ACCOUNT_KICKED = 0x02
    LOGIN_OK = 0x03
    SERVER_LIST = 0x04
    PLAY_FAIL = 0x06
    PLAY_OK = 0x07
    GG_AUTH = 0x0B


class LoginClientPacket(IntEnum):
    """Client -> login server."""

    REQUEST_AUTH_LOGIN = 0x00
    REQUEST_SERVER_LOGIN = 0x02
    REQUEST_SERVER_LIST = 0x05
    AUTH_GAME_GUARD = 0x07


class GameServerPacket(IntEnum):
    """Game server -> client (the subset the client knows about)."""

    KEY_PACKET = 0x00
    CHAR_MOVE_TO_LOCATION = 0x01
    CHAR_INFO = 0x03
    USER_INFO = 0x04
    ATTACK = 0x05
    DIE = 0x06
    REVIVE = 0x07
    SPAWN_ITEM = 0x0B
    DROP_ITEM = 0x0C
    GET_ITEM = 0x0D
    STATUS_UPDATE = 0x0E
    NPC_HTML_MESSAGE = 0x0F
    SELL_LIST = 0x10
    BUY_LIST = 0x11
    DELETE_OBJECT = 0x12
    CHAR_SELECT_INFO = 0x13
    LOGIN_FAIL = 0x14
    CHAR_SELECTED = 0x15
    NPC_INFO = 0x16
    CHAR_TEMPLATES = 0x17
    CHAR_CREATE_OK = 0x19
    CHAR_CREATE_FAIL = 0x1A
    ITEM_LIST = 0x1B
    SUNRISE = 0x1C
    SUNSET = 0x1D
    SERVER_CLOSE = 0x26
    INVENTORY_UPDATE = 0x27
    TELEPORT_TO_LOCATION = 0x28
    CHANGE_WAIT_TYPE = 0x2F
    MAGIC_SKILL_USE = 0x48
    CREATURE_SAY = 0x4A
    SKILL_LIST = 0x58
    SYSTEM_MESSAGE = 0x64
    ABNORMAL_STATUS_UPDATE = 0x7F
    QUEST_LIST = 0x80
    TUTORIAL_SHOW_HTML = 0xA0
    TUTORIAL_SHOW_QUESTION_MARK = 0xA1
    TUTORIAL_ENABLE_CLIENT_EVENT = 0xA2
    TUTORIAL_CLOSE_HTML = 0xA3
    ASK_JOIN_ALLIANCE = 0xA8
    PET_INFO = 0xB1
    PET_STATUS_UPDATE = 0xB5
    PET_DELETE = 0xB6
    SKILL_COOL_TIME = 0xC1
    NET_PING = 0xD3  # only ever the answer to the client's RequestNetPing
    CONFIRM_DLG = 0xED
    ETC_STATUS_UPDATE = 0xF3
    GAME_GUARD_QUERY = 0xF9
    EX_PACKET = 0xFE


class GameClientPacket(IntEnum):
    """Client -> game server."""

    PROTOCOL_VERSION = 0x00
    MOVE_TO_LOCATION = 0x01
    ENTER_WORLD = 0x03
    ACTION = 0x04
    AUTH_LOGIN = 0x08
    ATTACK_REQUEST = 0x0A
    CHARACTER_CREATE = 0x0B
    CHARACTER_SELECT = 0x0D
    NEW_CHARACTER = 0x0E
    USE_ITEM = 0x14
    REQUEST_SELL_ITEM = 0x1E
    REQUEST_BUY_ITEM = 0x1F
    REQUEST_LINK_HTML = 0x20
    REQUEST_BYPASS_TO_SERVER = 0x21
    SEND_BYPASS_BUILD_CMD = 0x5B
    REQUEST_MAGIC_SKILL_USE = 0x2F
    REQUEST_ACTION_USE = 0x45
    APPEARING = 0x30
    SAY2 = 0x38
    REQUEST_QUEST_LIST = 0x63
    REQUEST_RESTART_POINT = 0x6D
    REQUEST_TUTORIAL_LINK_HTML = 0x7B
    REQUEST_TUTORIAL_PASS_CMD_TO_SERVER = 0x7C
    REQUEST_TUTORIAL_QUESTION_MARK = 0x7D
    REQUEST_TUTORIAL_CLIENT_EVENT = 0x7E
    REQUEST_SKILL_COOL_TIME = 0x9D
    REQUEST_ITEM_LIST = 0x0F
    NET_PING = 0xA8
    DLG_ANSWER = 0xC5
    GAME_GUARD_REPLY = 0xCA
    EX_PACKET = 0xD0


class GameClientExPacket(IntEnum):
    """Sub-opcodes (uint16) of :attr:`GameClientPacket.EX_PACKET`."""

    REQUEST_AUTO_SOUL_SHOT = 0x05


class SystemMessageId(IntEnum):
    """SystemMessage (0x64) ids the bots react to."""

    YOU_DID_DAMAGE = 35  # "You hit for $s1 damage."
    DAMAGE_TAKEN = 36  # "$s1 hit you for $s2 damage."
    DAMAGE_TAKEN_2 = 37  # same message for another attacker kind
    PET_DID_DAMAGE = 1015  # "Your pet hit for $s1 damage."


#: Game server opcode -> packet name, for logging.
PACKET_NAMES = {packet.value: packet.name for packet in GameServerPacket}

STATUS_ATTRIBUTE_NAMES = {
    0x01: "LEVEL", 0x02: "EXP", 0x03: "STR", 0x04: "DEX",
    0x05: "CON", 0x06: "INT", 0x07: "WIT", 0x08: "MEN",
    0x09: "CUR_HP", 0x0A: "MAX_HP", 0x0B: "CUR_MP", 0x0C: "MAX_MP",
    0x0D: "SP", 0x0E: "CUR_LOAD", 0x0F: "MAX_LOAD",
    0x11: "P_ATK", 0x12: "ATK_SPD", 0x13: "P_DEF", 0x14: "EVASION",
    0x15: "ACCURACY", 0x16: "CRITICAL", 0x17: "M_ATK", 0x18: "CAST_SPD",
    0x19: "M_DEF", 0x1A: "PVP_FLAG", 0x1B: "KARMA",
    0x21: "CUR_CP", 0x22: "MAX_CP",
}

PAPERDOLL_NAMES = [
    "UNDER", "REAR", "LEAR", "NECK", "RFINGER", "LFINGER",
    "HEAD", "RHAND", "LHAND", "GLOVES", "CHEST", "LEGS",
    "FEET", "CLOAK", "RHAND2", "HAIR", "FACE",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read_utf16le_string(data: bytes, offset: int) -> tuple[str, int]:
    """Read a null-terminated UTF-16-LE string, return (string, new_offset)."""
    chars = bytearray()
    pos = offset
    while pos + 2 <= len(data):
        c0, c1 = data[pos], data[pos + 1]
        pos += 2
        if c0 == 0 and c1 == 0:
            break
        chars += struct.pack("<BB", c0, c1)
    return chars.decode('utf-16-le', errors='replace'), pos


# ---------------------------------------------------------------------------
# Packet builders (client -> server)
# ---------------------------------------------------------------------------

def pack(p: bytearray) -> bytes:
    return struct.pack("<H", len(p) + 2) + p


def build_auth_game_guard(sid: int) -> bytes:
    return bytes(bytearray([LoginClientPacket.AUTH_GAME_GUARD]) + struct.pack("<I", sid) + b'\x00' * 16)


def build_request_auth_login(login: str, pw: str, sid: int, rsa_mod: bytes) -> bytes:
    r = bytearray(128)
    r[0] = 0x80
    r[0x5E:0x5E + 14] = login.encode('utf-8').ljust(14, b'\x00')
    r[0x6C:0x6C + 16] = pw.encode('utf-8').ljust(16, b'\x00')
    struct.pack_into("<I", r, 124, sid)
    m = int.from_bytes(rsa_mod, 'big')
    encrypted = pow(int.from_bytes(r, 'big'), 65537, m).to_bytes(128, 'big')
    return bytes(bytearray([LoginClientPacket.REQUEST_AUTH_LOGIN]) + encrypted + b'\x00' * 8)


def build_request_server_list(l1: int, l2: int) -> bytes:
    return bytes(bytearray([LoginClientPacket.REQUEST_SERVER_LIST]) + struct.pack("<II", l1, l2))


def build_request_server_login(l1: int, l2: int, sid: int) -> bytes:
    return bytes(bytearray([LoginClientPacket.REQUEST_SERVER_LOGIN]) + struct.pack("<II", l1, l2) + bytes([sid]))


def build_proto_version(v: int = 746) -> bytes:
    return pack(bytearray([GameClientPacket.PROTOCOL_VERSION]) + struct.pack("<I", v))


def build_auth_login_gs(name: str, pk2: int, pk1: int, lk1: int, lk2: int) -> bytes:
    p = bytearray([GameClientPacket.AUTH_LOGIN]) + name.encode('utf-16-le') + b'\x00\x00'
    p += struct.pack("<IIII", pk2, pk1, lk1, lk2)
    return pack(p)


def build_character_select(slot: int) -> bytes:
    p = bytearray([GameClientPacket.CHARACTER_SELECT]) + struct.pack("<I", slot)
    p += struct.pack("<H", 0) + struct.pack("<III", 0, 0, 0)
    return pack(p)


def build_enter_world() -> bytes:
    p = bytearray([GameClientPacket.ENTER_WORLD]) + b'\x00' * 32 + struct.pack("<IIII", 0, 0, 0, 0)
    p += b'\x00' * 32 + struct.pack("<I", 0) + b'\x00' * 20
    return pack(p)


def build_request_item_list() -> bytes:
    """RequestItemList (0x0F): ask the server for a fresh ItemList (0x1B)."""
    return pack(bytearray([GameClientPacket.REQUEST_ITEM_LIST]))


def build_request_skill_cool_time() -> bytes:
    """RequestSkillCoolTime (0x9D): ask the server for a SkillCoolTime (0xC1) answer."""
    return pack(bytearray([GameClientPacket.REQUEST_SKILL_COOL_TIME]))


def build_net_ping(pid: int) -> bytes:
    return pack(bytearray([GameClientPacket.NET_PING]) + struct.pack("<I", pid))


def build_appearing() -> bytes:
    """Appearing (0x30): tells the server the client finished a teleport (TeleportToLocation)."""
    return pack(bytearray([GameClientPacket.APPEARING]))


def build_game_guard_reply() -> bytes:
    """GameGuardReply (0xCA): answer to the server's GameGuardQuery (0xF9)."""
    return pack(bytearray([GameClientPacket.GAME_GUARD_REPLY]) + struct.pack("<I", 0))


def build_move_to_location(tx: int, ty: int, tz: int,
                           ox: int, oy: int, oz: int, mode: int = 1) -> bytes:
    return pack(bytearray([GameClientPacket.MOVE_TO_LOCATION]) + struct.pack("<iiiiiii", tx, ty, tz, ox, oy, oz, mode))


def build_say2(msg: str, chat_type: int = 0, target: str = "") -> bytes:
    p = bytearray([GameClientPacket.SAY2])
    p += msg.encode('utf-16-le') + b'\x00\x00'
    p += struct.pack("<I", chat_type)
    if chat_type == 2 and target:
        p += target.encode('utf-16-le') + b'\x00\x00'
    return pack(p)


def build_action(target_obj_id: int, x: int, y: int, z: int, shift_click: bool = False) -> bytes:
    """
    Packet 0x04: Action -- interact/pickup (simple click = 0, shift-click = 1).

    Format: c dddd c
    """
    p = bytearray([GameClientPacket.ACTION])
    p += struct.pack("<iiii", target_obj_id, x, y, z)
    p += struct.pack("<B", 1 if shift_click else 0)
    return pack(p)


def build_attack_request(target_obj_id: int, x: int, y: int, z: int, shift_click: bool = False) -> bytes:
    """
    Packet 0x0A: AttackRequest -- initiate auto-attack on a target.

    Format: c dddd c
    """
    p = bytearray([GameClientPacket.ATTACK_REQUEST])
    p += struct.pack("<iiii", target_obj_id, x, y, z)
    p += struct.pack("<B", 1 if shift_click else 0)
    return pack(p)


def build_request_action_use(action_id: int, ctrl_pressed: bool = False, shift_pressed: bool = False) -> bytes:
    """
    Packet 0x45: RequestActionUse -- an action of the action panel, e.g. a summon command (22 = attack,
    21 = follow, 23 = stop for a servitor; 16 / 15 / 17 for a pet) or a summon skill (43 = Hydro Screw ...).

    Format: c d d c
    """
    p = bytearray([GameClientPacket.REQUEST_ACTION_USE])
    p += struct.pack("<iiB", action_id, 1 if ctrl_pressed else 0, 1 if shift_pressed else 0)
    return pack(p)


def build_use_skill(skill_id: int, ctrl_pressed: bool = False, shift_pressed: bool = False) -> bytes:
    """
    Packet 0x2F: RequestMagicSkillUse -- use an active skill.

    Format: c ii B  (skillId int, ctrlPressed int, shiftPressed byte)
    """
    p = bytearray([GameClientPacket.REQUEST_MAGIC_SKILL_USE])
    p += struct.pack("<iiB", skill_id, 1 if ctrl_pressed else 0, 1 if shift_pressed else 0)
    return pack(p)


def build_use_item(object_id: int, ctrl_pressed: bool = False) -> bytes:
    """
    Packet 0x14: UseItem -- use an inventory item by its object ID.

    Format: c ii  (objectId int, ctrlPressed int)
    """
    p = bytearray([GameClientPacket.USE_ITEM])
    p += struct.pack("<ii", object_id, 1 if ctrl_pressed else 0)
    return pack(p)


def build_request_restart_point(requested_point_type: int = 0) -> bytes:
    """
    Packet 0x6D: RequestRestartPoint — return to town/clanhall/castle after death.

    Point types: 0 = town (default), 1 = clan hall, 2 = castle, 27 = jail.
    Format: c i
    """
    p = bytearray([GameClientPacket.REQUEST_RESTART_POINT])
    p += struct.pack("<i", requested_point_type)
    return pack(p)


def build_dlg_answer(message_id: int, answer: int, requester_id: int = 0) -> bytes:
    """
    Packet 0xC5: DlgAnswer — respond to a confirmation dialog.

    Used for accepting/rejecting resurrection, summons, etc.
    Format: c iii
    """
    p = bytearray([GameClientPacket.DLG_ANSWER])
    p += struct.pack("<iii", message_id, answer, requester_id)
    return pack(p)


def _utf16(text: str) -> bytes:
    return text.encode("utf-16-le") + b"\x00\x00"


def build_new_character() -> bytes:
    """NewCharacter (0x0E): open the character creation; the server answers with CharTemplates (0x17)."""
    return pack(bytearray([GameClientPacket.NEW_CHARACTER]))


def build_character_create(name: str, race: int, female: bool, class_id: int,
                           hair_style: int = 0, hair_color: int = 0, face: int = 0) -> bytes:
    """
    CharacterCreate (0x0B): a new character on the account. The server takes the base stats from the class
    template and ignores the six it reads here. Answer: CharCreateOk (0x19) + CharSelectInfo, or CharCreateFail.

    Format: c S d d d dddddd d d d
    """
    p = bytearray([GameClientPacket.CHARACTER_CREATE]) + _utf16(name)
    p += struct.pack("<iii", race, 1 if female else 0, class_id) + b"\x00" * 24
    p += struct.pack("<iii", hair_style, hair_color, face)
    return pack(p)


def build_bypass_to_server(command: str) -> bytes:
    """RequestBypassToServer (0x21): an ``<a action="bypass -h ...">`` link of a dialog (without ``bypass -h``)."""
    return pack(bytearray([GameClientPacket.REQUEST_BYPASS_TO_SERVER]) + _utf16(command))


def build_bypass_build_cmd(command: str) -> bytes:
    """SendBypassBuildCmd (0x5B): a ``//command`` of the chat line, run as ``admin_<command>`` (needs GM access)."""
    return pack(bytearray([GameClientPacket.SEND_BYPASS_BUILD_CMD]) + _utf16(command))


def build_link_html(link: str) -> bytes:
    """RequestLinkHtml (0x20): an ``<a action="link ...">`` link of a dialog (an .htm file under data/html)."""
    return pack(bytearray([GameClientPacket.REQUEST_LINK_HTML]) + _utf16(link))


def build_request_buy_item(list_id: int, items: list[tuple[int, int]]) -> bytes:
    """
    RequestBuyItem (0x1F): buy (item id, count) pairs from the targeted merchant's list ``list_id``.

    Format: c d d (d d)*
    """
    p = bytearray([GameClientPacket.REQUEST_BUY_ITEM]) + struct.pack("<ii", list_id, len(items))
    for item_id, count in items:
        p += struct.pack("<ii", item_id, count)
    return pack(p)


def build_request_sell_item(items: list[tuple[int, int, int]]) -> bytes:
    """
    RequestSellItem (0x1E): sell (object id, item id, count) to the targeted merchant (list id 0 = a sale).

    Format: c d d (d d d)*
    """
    p = bytearray([GameClientPacket.REQUEST_SELL_ITEM]) + struct.pack("<ii", 0, len(items))
    for object_id, item_id, count in items:
        p += struct.pack("<iii", object_id, item_id, count)
    return pack(p)


def build_request_quest_list() -> bytes:
    """RequestQuestList (0x63): the server answers with QuestList (0x80)."""
    return pack(bytearray([GameClientPacket.REQUEST_QUEST_LIST]))


def build_tutorial_link_html(bypass: str) -> bytes:
    """RequestTutorialLinkHtml (0x7B): a ``link`` of a tutorial window."""
    return pack(bytearray([GameClientPacket.REQUEST_TUTORIAL_LINK_HTML]) + _utf16(bypass))


def build_tutorial_pass_cmd(bypass: str) -> bytes:
    """RequestTutorialPassCmdToServer (0x7C): a ``bypass`` of a tutorial window (e.g. ``close``)."""
    return pack(bytearray([GameClientPacket.REQUEST_TUTORIAL_PASS_CMD_TO_SERVER]) + _utf16(bypass))


def build_tutorial_question_mark(number: int) -> bytes:
    """RequestTutorialQuestionMark (0x7D): a click on the tutorial question mark ``number``."""
    return pack(bytearray([GameClientPacket.REQUEST_TUTORIAL_QUESTION_MARK]) + struct.pack("<i", number))


def build_tutorial_client_event(event_id: int) -> bytes:
    """RequestTutorialClientEvent (0x7E): a client event the tutorial asked for (TutorialEnableClientEvent)."""
    return pack(bytearray([GameClientPacket.REQUEST_TUTORIAL_CLIENT_EVENT]) + struct.pack("<i", event_id))


def build_auto_soulshot(item_id: int, enable: bool = True) -> bytes:
    """
    ExPacket (client->server) 0xD0 0x05: RequestAutoSoulShot -- enable/disable auto soulshot use.

    Format: D0 (H)05 dd
    """
    p = bytearray([GameClientPacket.EX_PACKET])
    p += struct.pack("<H", GameClientExPacket.REQUEST_AUTO_SOUL_SHOT)
    p += struct.pack("<ii", item_id, 1 if enable else 0)
    return pack(p)


# ---------------------------------------------------------------------------
# Packet parsers (server -> client)
# ---------------------------------------------------------------------------

def parse_user_info(body: bytes) -> dict:
    """Parse UserInfo (0x04) packet -- comprehensive character info."""
    if len(body) < 13:
        return {}
    info = {}
    pos = 1  # skip packet ID

    info['x'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['y'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['z'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['vehicle_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['object_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['name'], pos = _read_utf16le_string(body, pos)

    info['race'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['sex'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['base_class'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['level'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['exp'] = struct.unpack_from("<q", body, pos)[0]; pos += 8
    info['STR'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['DEX'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['CON'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['INT'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['WIT'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['MEN'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['max_hp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['cur_hp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['max_mp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['cur_mp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['sp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['cur_load'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['max_load'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['weapon_slot_flag'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    # Paperdoll object IDs
    info['paperdoll'] = {}
    for slot_name in PAPERDOLL_NAMES:
        info['paperdoll'][slot_name] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    # Paperdoll item display IDs
    info['paperdoll_display'] = {}
    for slot_name in PAPERDOLL_NAMES:
        info['paperdoll_display'][slot_name] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    # c6 new h's block 1: 14 shorts
    pos += 14 * 2
    info['augment1'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    # c6 new h's block 2: 12 shorts
    pos += 12 * 2
    info['augment2'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    # c6 new h's block 3: 4 shorts
    pos += 4 * 2

    info['p_atk'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['p_atk_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['p_def'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['evasion'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['accuracy'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['critical'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['m_atk'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['m_atk_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    pos += 4  # duplicate pAtkSpd
    info['m_def'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['pvp_flag'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['karma'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['run_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['walk_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['swim_run_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['swim_walk_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['fly_run_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['fly_walk_spd'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    pos += 4  # duplicate flyRunSpd
    pos += 4  # duplicate flyWalkSpd

    info['move_multiplier'] = struct.unpack_from("<d", body, pos)[0]; pos += 8
    info['atk_spd_multiplier'] = struct.unpack_from("<d", body, pos)[0]; pos += 8

    info['collision_radius'] = struct.unpack_from("<d", body, pos)[0]; pos += 8
    info['collision_height'] = struct.unpack_from("<d", body, pos)[0]; pos += 8

    info['hair_style'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['hair_color'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['face'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['is_gm'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['title'], pos = _read_utf16le_string(body, pos)

    info['clan_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['clan_crest_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['ally_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['ally_crest_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['relation'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['mount_type'] = body[pos]; pos += 1
    info['private_store_type'] = body[pos]; pos += 1
    info['has_dwarven_craft'] = body[pos]; pos += 1

    info['pk_kills'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['pvp_kills'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    cubic_count = struct.unpack_from("<H", body, pos)[0]; pos += 2
    info['cubics'] = []
    for _ in range(cubic_count):
        info['cubics'].append(struct.unpack_from("<H", body, pos)[0]); pos += 2

    info['in_party_match_room'] = body[pos]; pos += 1
    info['abnormal_visual_effects'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['in_water'] = body[pos]; pos += 1
    info['clan_privileges'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['recom_left'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    info['recom_have'] = struct.unpack_from("<H", body, pos)[0]; pos += 2

    info['mount_npc_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['inventory_limit'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    info['class_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    pos += 4  # special effects

    info['max_cp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['cur_cp'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['enchant_effect'] = body[pos]; pos += 1
    info['team'] = body[pos]; pos += 1
    info['clan_crest_large_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['is_noble'] = body[pos]; pos += 1
    info['is_hero'] = body[pos]; pos += 1

    info['is_fishing'] = body[pos]; pos += 1
    info['fish_x'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['fish_y'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['fish_z'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['name_color'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    info['is_running'] = body[pos]; pos += 1
    info['pledge_class'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['pledge_type'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['title_color'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    info['cursed_weapon_level'] = struct.unpack_from("<i", body, pos)[0]; pos += 4

    return info


def _parse_item(body: bytes, pos: int) -> tuple[dict, int]:
    """One item of ItemList / InventoryUpdate (32 bytes) at ``pos``; returns it and the position after it."""
    item = {}
    item['type1'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['object_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    item['item_id'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    item['count'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    item['type2'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['custom_type1'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['equipped'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['body_part'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    item['enchant'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['custom_type2'] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item['augmentation_bonus'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    item['mana'] = struct.unpack_from("<i", body, pos)[0]; pos += 4
    return item, pos


def parse_item_list(body: bytes) -> dict:
    """Parse ItemList (0x1B) -- inventory contents."""
    if len(body) < 5:
        return {"show_window": False, "items": []}
    pos = 1
    show_window = struct.unpack_from("<H", body, pos)[0]; pos += 2
    item_count = struct.unpack_from("<H", body, pos)[0]; pos += 2
    items = []
    for _ in range(item_count):
        if pos + 32 > len(body):
            break
        item, pos = _parse_item(body, pos)
        items.append(item)
    return {"show_window": show_window != 0, "items": items}


def parse_inventory_update(body: bytes) -> list[dict]:
    """Parse InventoryUpdate (0x27): the changed items, each with ``change`` 1 = added, 2 = modified, 3 = removed."""
    if len(body) < 3:
        return []
    pos = 1
    count = struct.unpack_from("<H", body, pos)[0]; pos += 2
    items = []
    for _ in range(count):
        if pos + 34 > len(body):
            break
        change = struct.unpack_from("<H", body, pos)[0]; pos += 2
        item, pos = _parse_item(body, pos)
        item['change'] = change
        items.append(item)
    return items


def parse_change_wait_type(body: bytes) -> dict:
    """Parse ChangeWaitType (0x2F): ``wait_type`` 0 = sitting, 1 = standing, 2 / 3 = fake death start / stop."""
    if len(body) < 9:
        return {}
    object_id, wait_type = struct.unpack_from("<ii", body, 1)
    return {"object_id": object_id, "wait_type": wait_type}


def parse_skill_list(body: bytes) -> dict:
    """Parse SkillList (0x58)."""
    if len(body) < 5:
        return {"skills": []}
    pos = 1
    count = struct.unpack_from("<i", body, pos)[0]; pos += 4
    skills = []
    for _ in range(count):
        if pos + 13 > len(body):
            break
        passive = struct.unpack_from("<i", body, pos)[0]; pos += 4
        level = struct.unpack_from("<i", body, pos)[0]; pos += 4
        sid = struct.unpack_from("<i", body, pos)[0]; pos += 4
        disabled = body[pos]; pos += 1
        skills.append({
            "id": sid,
            "level": level,
            "passive": passive != 0,
            "disabled": disabled != 0,
        })
    return {"skills": skills}


def parse_status_update(body: bytes) -> dict:
    """Parse StatusUpdate (0x0E)."""
    if len(body) < 9:
        return {}
    pos = 1
    object_id = struct.unpack_from("<i", body, pos)[0]; pos += 4
    count = struct.unpack_from("<i", body, pos)[0]; pos += 4
    attrs = {}
    for _ in range(count):
        if pos + 8 > len(body):
            break
        attr_id = struct.unpack_from("<i", body, pos)[0]; pos += 4
        value = struct.unpack_from("<i", body, pos)[0]; pos += 4
        attrs[attr_id] = value
    return {"object_id": object_id, "attributes": attrs}


def parse_skill_cool_time(body: bytes) -> dict:
    """
    Parse SkillCoolTime (0xC1).

    Times are whole seconds as sent by the server, i.e. already scaled by its speed settings;
    only skills that are still cooling down are listed.
    """
    if len(body) < 5:
        return {"cooldowns": []}
    pos = 1
    count = struct.unpack_from("<i", body, pos)[0]; pos += 4
    cooldowns = []
    for _ in range(count):
        if pos + 16 > len(body):
            break
        skill_id, level, reuse, remaining = struct.unpack_from("<iiii", body, pos); pos += 16
        cooldowns.append({"skill_id": skill_id, "level": level, "reuse": reuse, "remaining": remaining})
    return {"cooldowns": cooldowns}


def parse_etc_status_update(body: bytes) -> dict:
    """Parse EtcStatusUpdate (0xF3)."""
    if len(body) < 29:
        return {}
    pos = 1
    return {
        "charges": struct.unpack_from("<i", body, pos)[0],
        "weight_penalty": struct.unpack_from("<i", body, pos + 4)[0],
        "message_refusal": struct.unpack_from("<i", body, pos + 8)[0],
        "danger_area": struct.unpack_from("<i", body, pos + 12)[0],
        "weapon_armor_penalty": struct.unpack_from("<i", body, pos + 16)[0],
        "charm_of_courage": struct.unpack_from("<i", body, pos + 20)[0],
        "death_penalty_buff_level": struct.unpack_from("<i", body, pos + 24)[0],
    }


def parse_creature_say(body: bytes) -> tuple[str, str]:
    """Parse CreatureSay (0x4A), return (speaker_name, message)."""
    # Format: opcode(1) + objectId(4) + chatType(4) + name(UTF16LE\0) + msg(UTF16LE\0)
    name, pos = _read_utf16le_string(body, 9)
    msg, _ = _read_utf16le_string(body, pos)
    return name, msg


def parse_spawn_item(body: bytes) -> dict:
    """Parse SpawnItem (0x0B) -- a dropped item appearing in the world."""
    if len(body) < 29:
        return {}
    return {
        "object_id": struct.unpack_from("<i", body, 1)[0],
        "item_id": struct.unpack_from("<i", body, 5)[0],
        "x": struct.unpack_from("<i", body, 9)[0],
        "y": struct.unpack_from("<i", body, 13)[0],
        "z": struct.unpack_from("<i", body, 17)[0],
        "stackable": struct.unpack_from("<i", body, 21)[0] != 0,
        "count": struct.unpack_from("<i", body, 25)[0],
    }


def parse_drop_item(body: bytes) -> dict:
    """Parse DropItem (0x0C) -- same format as SpawnItem but with dropper id first."""
    if len(body) < 33:
        return {}
    return {
        "dropper_id": struct.unpack_from("<i", body, 1)[0],
        "object_id": struct.unpack_from("<i", body, 5)[0],
        "item_id": struct.unpack_from("<i", body, 9)[0],
        "x": struct.unpack_from("<i", body, 13)[0],
        "y": struct.unpack_from("<i", body, 17)[0],
        "z": struct.unpack_from("<i", body, 21)[0],
        "stackable": struct.unpack_from("<i", body, 25)[0] != 0,
        "count": struct.unpack_from("<i", body, 29)[0],
    }


# NpcInfo binary layout -- offsets from packet opcode byte (index 0)
_NPCINFO_NAME_OFFSET = 122  # byte after opcode where the UTF-16-LE name starts


def parse_npc_info(body: bytes) -> dict:
    """Parse NpcInfo (0x16) -- extract objectId, position, heading, name, title."""
    if len(body) < _NPCINFO_NAME_OFFSET + 2:
        return {}
    obj_id = struct.unpack_from("<i", body, 1)[0]
    display_id = struct.unpack_from("<i", body, 5)[0]  # real = display_id + 1000000
    is_attackable = struct.unpack_from("<i", body, 9)[0]
    x = struct.unpack_from("<i", body, 13)[0]
    y = struct.unpack_from("<i", body, 17)[0]
    z = struct.unpack_from("<i", body, 21)[0]
    heading = struct.unpack_from("<i", body, 25)[0]  # radians * 182.044 = degrees
    is_alike_dead = body[120]  # 1 = dead or fake-dead
    is_in_combat = body[119]   # 1 = in combat

    name, pos = _read_utf16le_string(body, _NPCINFO_NAME_OFFSET)
    title, _ = _read_utf16le_string(body, pos)

    # Extract NPC level from title -- format like "Lv 18 [A] Orc Fighter"
    level = 0
    if title.startswith("Lv ") or " Lv " in title:
        match = re.search(r'Lv\s+(\d+)', title)
        if match:
            level = int(match.group(1))

    return {
        "object_id": obj_id,
        "display_id": display_id,
        "is_attackable": is_attackable != 0,
        "x": x, "y": y, "z": z,
        "heading": heading,
        "is_alike_dead": is_alike_dead != 0,
        "is_in_combat": is_in_combat != 0,
        "name": name,
        "title": title,
        "level": level,
    }


def parse_die(body: bytes) -> dict:
    """Parse Die (0x06) packet."""
    if len(body) < 9:
        return {}
    return {
        "object_id": struct.unpack_from("<i", body, 1)[0],
        "can_teleport": struct.unpack_from("<i", body, 5)[0] != 0,
    }


def parse_teleport_to_location(body: bytes) -> dict:
    """Parse TeleportToLocation (0x28): object id and destination."""
    if len(body) < 17:
        return {}
    object_id, x, y, z = struct.unpack_from("<iiii", body, 1)
    return {"object_id": object_id, "x": x, "y": y, "z": z}


def parse_pet_info(body: bytes) -> dict:
    """
    Parse PetInfo (0xB1): the character's summon appeared or changed.

    ``{summon_type (1 = servitor, 2 = pet), object_id, npc_id, x, y, z, in_combat, dead, cur_hp,
    max_hp, cur_mp, max_mp, level}``
    """
    try:
        pos = 1
        summon_type, object_id, display_id, _ = struct.unpack_from("<iiii", body, pos); pos += 16
        x, y, z, _heading, _ = struct.unpack_from("<iiiii", body, pos); pos += 20
        pos += 10 * 4 + 4 * 8 + 3 * 4  # speeds, multipliers and collision, weapon / armor
        _owner, _running, in_combat, dead, _value = struct.unpack_from("<BBBBB", body, pos); pos += 5
        _, pos = _read_utf16le_string(body, pos)  # name
        _, pos = _read_utf16le_string(body, pos)  # owner name
        pos += 4 * 5  # 1, pvp flag, karma, current / max fed
        cur_hp, max_hp, cur_mp, max_mp, _sp, level = struct.unpack_from("<iiiiii", body, pos)
    except struct.error:
        return {}
    return {"summon_type": summon_type, "object_id": object_id, "npc_id": display_id - 1000000, "x": x, "y": y,
            "z": z, "in_combat": bool(in_combat), "dead": bool(dead), "cur_hp": cur_hp, "max_hp": max_hp,
            "cur_mp": cur_mp, "max_mp": max_mp, "level": level}


def parse_pet_status_update(body: bytes) -> dict:
    """Parse PetStatusUpdate (0xB5): ``{summon_type, object_id, x, y, z, cur_hp, max_hp, cur_mp, max_mp, level}``."""
    try:
        pos = 1
        summon_type, object_id, x, y, z = struct.unpack_from("<iiiii", body, pos); pos += 20
        _, pos = _read_utf16le_string(body, pos)  # owner name
        pos += 8  # current / max fed
        cur_hp, max_hp, cur_mp, max_mp, level = struct.unpack_from("<iiiii", body, pos)
    except struct.error:
        return {}
    return {"summon_type": summon_type, "object_id": object_id, "x": x, "y": y, "z": z, "cur_hp": cur_hp,
            "max_hp": max_hp, "cur_mp": cur_mp, "max_mp": max_mp, "level": level}


def parse_pet_delete(body: bytes) -> dict:
    """Parse PetDelete (0xB6): ``{summon_type, object_id}``."""
    if len(body) < 9:
        return {}
    summon_type, object_id = struct.unpack_from("<ii", body, 1)
    return {"summon_type": summon_type, "object_id": object_id}


def parse_attack(body: bytes) -> dict:
    """
    Parse Attack (0x05): ``{"attacker_id": int, "hits": [(target_id, damage, flags), ...]}``.

    Layout: attacker, first hit (target, damage, flags byte), attacker x/y/z, the number of further
    hits (short), the further hits, target x/y/z. A polearm or an area attack hits several targets.
    """
    if len(body) < 1 + 4 + 9 + 12 + 2:
        return {}
    attacker = struct.unpack_from("<i", body, 1)[0]
    pos = 5
    hits = [struct.unpack_from("<iiB", body, pos)]
    pos += 9 + 12
    count = struct.unpack_from("<H", body, pos)[0]
    pos += 2
    for _ in range(count):
        if pos + 9 > len(body):
            break
        hits.append(struct.unpack_from("<iiB", body, pos))
        pos += 9
    return {"attacker_id": attacker, "hits": hits}


def parse_magic_skill_use(body: bytes) -> dict:
    """
    Parse MagicSkillUse (0x48): a creature starts casting.

    ``hit_time`` and ``reuse`` are milliseconds of *game* time (divide by the server ``TimeScale``
    for real time).
    """
    if len(body) < 25:
        return {}
    caster_id, target_id, skill_id, skill_level, hit_time, reuse = struct.unpack_from("<iiiiii", body, 1)
    return {"caster_id": caster_id, "target_id": target_id, "skill_id": skill_id,
            "skill_level": skill_level, "hit_time": hit_time, "reuse": reuse}


# SystemMessage parameter types (SystemMessage.TYPE_* on the server).
_SM_TEXT, _SM_INT, _SM_SKILL, _SM_LONG, _SM_ZONE, _SM_PLAYER = 0, 1, 4, 6, 7, 12


def parse_system_message(body: bytes) -> dict:
    """
    Parse SystemMessage (0x64): ``{"msg_id": int, "params": [...]}``.

    Text and player names become ``str``, numbers ``int``, skill names ``(skill_id, level)`` and zone
    names ``(x, y, z)``; every other parameter type is a single ``int`` (item / NPC ids, ...).
    """
    if len(body) < 9:
        return {}
    msg_id, count = struct.unpack_from("<ii", body, 1)
    pos, params = 9, []
    for _ in range(count):
        if pos + 4 > len(body):
            break
        ptype = struct.unpack_from("<i", body, pos)[0]; pos += 4
        if ptype in (_SM_TEXT, _SM_PLAYER):
            value, pos = _read_utf16le_string(body, pos)
        elif ptype == _SM_LONG:
            if pos + 8 > len(body):
                break
            value = struct.unpack_from("<q", body, pos)[0]; pos += 8
        elif ptype in (_SM_SKILL, _SM_ZONE):
            n = 2 if ptype == _SM_SKILL else 3
            if pos + 4 * n > len(body):
                break
            value = struct.unpack_from(f"<{n}i", body, pos); pos += 4 * n
        else:
            if pos + 4 > len(body):
                break
            value = struct.unpack_from("<i", body, pos)[0]; pos += 4
        params.append(value)
    return {"msg_id": msg_id, "params": params}


def parse_get_item(body: bytes) -> dict:
    """Parse GetItem (0x0D): who picked up which ground item (object id)."""
    if len(body) < 9:
        return {}
    picker_id, object_id = struct.unpack_from("<ii", body, 1)
    return {"picker_id": picker_id, "object_id": object_id}


def parse_revive(body: bytes) -> dict:
    """Parse Revive (0x07)."""
    if len(body) < 5:
        return {}
    return {"object_id": struct.unpack_from("<i", body, 1)[0]}


def parse_delete_object(body: bytes) -> dict:
    """Parse DeleteObject (0x12) packet."""
    if len(body) < 5:
        return {}
    return {
        "object_id": struct.unpack_from("<i", body, 1)[0],
    }


def parse_abnormal_status_update(body: bytes) -> dict:
    """
    Parse AbnormalStatusUpdate (0x7F) -- buff/debuff list.

    Note: the ``count`` field written by the server may be larger than the
    number of *actually serialised* entries because inactive effects are
    skipped inside the loop.  We parse until the packet runs out of bytes.
    """
    if len(body) < 3:
        return {"buffs": []}
    pos = 3
    end = len(body)
    buffs = []
    while pos + 10 <= end:
        skill_id = struct.unpack_from("<i", body, pos)[0]; pos += 4
        skill_level = struct.unpack_from("<H", body, pos)[0]; pos += 2
        duration = struct.unpack_from("<i", body, pos)[0]; pos += 4
        buffs.append({
            "skill_id": skill_id,
            "skill_level": skill_level,
            "duration": duration,
        })
    return {"buffs": buffs}


def parse_npc_html_message(body: bytes) -> dict:
    """NpcHtmlMessage (0x0F): ``npc_object_id`` (0 = no NPC), ``html``, ``item_id`` (a book / item dialog)."""
    if len(body) < 5:
        return {}
    npc_object_id = struct.unpack_from("<i", body, 1)[0]
    html, pos = _read_utf16le_string(body, 5)
    item_id = struct.unpack_from("<i", body, pos)[0] if pos + 4 <= len(body) else 0
    return {"npc_object_id": npc_object_id, "html": html, "item_id": item_id}


def parse_tutorial_show_html(body: bytes) -> dict:
    """TutorialShowHtml (0xA0): ``html`` of a tutorial window."""
    return {"html": _read_utf16le_string(body, 1)[0]}


def parse_tutorial_int(body: bytes) -> dict:
    """TutorialShowQuestionMark (0xA1) / TutorialEnableClientEvent (0xA2): the mark or event id."""
    return {"id": struct.unpack_from("<i", body, 1)[0]} if len(body) >= 5 else {}


_TRADE_ITEM = struct.Struct("<HiiiHHiHHHi")  # type1, object id, item id, count, type2, -, body part, enchant, -, -, price


def _parse_trade_items(body: bytes, pos: int) -> list[dict]:
    items = []
    while pos + _TRADE_ITEM.size <= len(body):
        type1, object_id, item_id, count, type2, _, body_part, enchant, _, _, price = _TRADE_ITEM.unpack_from(body, pos)
        pos += _TRADE_ITEM.size
        items.append({"object_id": object_id, "item_id": item_id, "count": count, "type1": type1, "type2": type2,
                      "body_part": body_part, "enchant": enchant, "price": price})
    return items


def parse_buy_list(body: bytes) -> dict:
    """
    BuyList (0x11): ``adena``, ``list_id`` and the ``items`` on sale with their ``price`` (tax included); ``count``
    0 = unlimited. The written item count also counts sold-out items that are left out, so read until the end.
    """
    if len(body) < 11:
        return {}
    adena, list_id = struct.unpack_from("<ii", body, 1)
    return {"adena": adena, "list_id": list_id, "items": _parse_trade_items(body, 11)}


def parse_sell_list(body: bytes) -> dict:
    """SellList (0x10): ``adena`` and the inventory ``items`` the merchant buys, ``price`` = what it pays per piece."""
    if len(body) < 11:
        return {}
    adena = struct.unpack_from("<i", body, 1)[0]
    return {"adena": adena, "items": _parse_trade_items(body, 11)}


def parse_quest_list(body: bytes) -> dict:
    """QuestList (0x80): ``quests`` = {quest id: cond} of the started quests (cond 0 = no state yet)."""
    if len(body) < 3:
        return {"quests": {}}
    count = struct.unpack_from("<H", body, 1)[0]
    quests, pos = {}, 3
    for _ in range(count):
        if pos + 8 > len(body):
            break
        quest_id, cond = struct.unpack_from("<ii", body, pos)
        pos += 8
        quests[quest_id] = cond
    return {"quests": quests}


def parse_char_create_fail(body: bytes) -> dict:
    """CharCreateFail (0x1A): ``reason`` (0 failed, 1 too many characters, 2 name exists, 3 16 chars, 4 bad name)."""
    return {"reason": struct.unpack_from("<i", body, 1)[0] if len(body) >= 5 else -1}
