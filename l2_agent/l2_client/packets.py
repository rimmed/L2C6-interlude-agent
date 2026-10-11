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

PACKET_NAMES = {
    0x00: "KeyPacket", 0x01: "MoveToLoc", 0x03: "CharInfo", 0x04: "UserInfo",
    0x0B: "SpawnItem", 0x0E: "StatusUpd", 0x13: "CharSelInfo", 0x14: "LoginFail",
    0x15: "CharSelected", 0x16: "NpcInfo", 0x1B: "ItemList",
    0x1C: "Sunrise", 0x1D: "Sunset", 0x26: "ServerClose",
    0x27: "InvUpdate", 0x4A: "CreatureSay", 0x58: "SkillList",
    0x64: "SysMsg", 0x7F: "AbnormalSt", 0x80: "QuestList",
    0xA8: "NetPing", 0xC1: "SkillCool", 0xF3: "EtcStatus",
    0xF9: "GGQuery", 0xFE: "ExPacket",
}

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
    return bytes(bytearray([0x07]) + struct.pack("<I", sid) + b'\x00' * 16)


def build_request_auth_login(login: str, pw: str, sid: int, rsa_mod: bytes) -> bytes:
    r = bytearray(128)
    r[0] = 0x80
    r[0x5E:0x5E + 14] = login.encode('utf-8').ljust(14, b'\x00')
    r[0x6C:0x6C + 16] = pw.encode('utf-8').ljust(16, b'\x00')
    struct.pack_into("<I", r, 124, sid)
    m = int.from_bytes(rsa_mod, 'big')
    return bytes(bytearray([0x00]) + pow(int.from_bytes(r, 'big'), 65537, m).to_bytes(128, 'big') + b'\x00' * 8)


def build_request_server_list(l1: int, l2: int) -> bytes:
    return bytes(bytearray([0x05]) + struct.pack("<II", l1, l2))


def build_request_server_login(l1: int, l2: int, sid: int) -> bytes:
    return bytes(bytearray([0x02]) + struct.pack("<II", l1, l2) + bytes([sid]))


def build_proto_version(v: int = 746) -> bytes:
    return pack(bytearray([0x00]) + struct.pack("<I", v))


def build_auth_login_gs(name: str, pk2: int, pk1: int, lk1: int, lk2: int) -> bytes:
    p = bytearray([0x08]) + name.encode('utf-16-le') + b'\x00\x00'
    p += struct.pack("<IIII", pk2, pk1, lk1, lk2)
    return pack(p)


def build_character_select(slot: int) -> bytes:
    p = bytearray([0x0D]) + struct.pack("<I", slot)
    p += struct.pack("<H", 0) + struct.pack("<III", 0, 0, 0)
    return pack(p)


def build_enter_world() -> bytes:
    p = bytearray([0x03]) + b'\x00' * 32 + struct.pack("<IIII", 0, 0, 0, 0)
    p += b'\x00' * 32 + struct.pack("<I", 0) + b'\x00' * 20
    return pack(p)


def build_request_skill_cool_time() -> bytes:
    """RequestSkillCoolTime (0x9D): ask the server for a SkillCoolTime (0xC1) answer."""
    return pack(bytearray([0x9D]))


def build_net_ping(pid: int) -> bytes:
    return pack(bytearray([0xA8]) + struct.pack("<I", pid))


def build_move_to_location(tx: int, ty: int, tz: int,
                           ox: int, oy: int, oz: int, mode: int = 1) -> bytes:
    return pack(bytearray([0x01]) + struct.pack("<iiiiiii", tx, ty, tz, ox, oy, oz, mode))


def build_say2(msg: str, chat_type: int = 0, target: str = "") -> bytes:
    p = bytearray([0x38])
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
    p = bytearray([0x04])
    p += struct.pack("<iiii", target_obj_id, x, y, z)
    p += struct.pack("<B", 1 if shift_click else 0)
    return pack(p)


def build_attack_request(target_obj_id: int, x: int, y: int, z: int, shift_click: bool = False) -> bytes:
    """
    Packet 0x0A: AttackRequest -- initiate auto-attack on a target.

    Format: c dddd c
    """
    p = bytearray([0x0A])
    p += struct.pack("<iiii", target_obj_id, x, y, z)
    p += struct.pack("<B", 1 if shift_click else 0)
    return pack(p)


def build_use_skill(skill_id: int, ctrl_pressed: bool = False, shift_pressed: bool = False) -> bytes:
    """
    Packet 0x2F: RequestMagicSkillUse -- use an active skill.

    Format: c ii B  (skillId int, ctrlPressed int, shiftPressed byte)
    """
    p = bytearray([0x2F])
    p += struct.pack("<iiB", skill_id, 1 if ctrl_pressed else 0, 1 if shift_pressed else 0)
    return pack(p)


def build_use_item(object_id: int, ctrl_pressed: bool = False) -> bytes:
    """
    Packet 0x14: UseItem -- use an inventory item by its object ID.

    Format: c ii  (objectId int, ctrlPressed int)
    """
    p = bytearray([0x14])
    p += struct.pack("<ii", object_id, 1 if ctrl_pressed else 0)
    return pack(p)


def build_request_restart_point(requested_point_type: int = 0) -> bytes:
    """
    Packet 0x6D: RequestRestartPoint — return to town/clanhall/castle after death.

    Point types: 0 = town (default), 1 = clan hall, 2 = castle, 27 = jail.
    Format: c i
    """
    p = bytearray([0x6D])
    p += struct.pack("<i", requested_point_type)
    return pack(p)


def build_dlg_answer(message_id: int, answer: int, requester_id: int = 0) -> bytes:
    """
    Packet 0xC5: DlgAnswer — respond to a confirmation dialog.

    Used for accepting/rejecting resurrection, summons, etc.
    Format: c iii
    """
    p = bytearray([0xC5])
    p += struct.pack("<iii", message_id, answer, requester_id)
    return pack(p)


def build_auto_soulshot(item_id: int, enable: bool = True) -> bytes:
    """
    ExPacket (client->server) 0xD0 0x05: RequestAutoSoulShot -- enable/disable auto soulshot use.

    Format: D0 (H)05 dd
    """
    p = bytearray([0xD0])
    p += struct.pack("<H", 0x05)  # sub-opcode
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
        items.append(item)
    return {"show_window": show_window != 0, "items": items}


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


def parse_delete_object(body: bytes) -> dict:
    """Parse DeleteObject (0x08) packet."""
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
