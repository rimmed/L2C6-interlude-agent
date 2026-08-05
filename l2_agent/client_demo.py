#!/usr/bin/env python3
# L2J Mobius Interlude client bot -- connects, enters world, reads full character data.


import random
import socket
import struct
import time

from l2_client import crypto, packets
from l2_client.character import L2Character
from l2_client.tools.data import item_data, npc_data, skill_data

# --- CONFIGURATION ---
SERVER_IP = "127.0.0.1"
LOGIN_PORT = 2106
ACCOUNT_LOGIN = ""
ACCOUNT_PASSWORD = ""
SERVER_ID = 0
CHAR_SLOT = 0


def _print_buffs(char: L2Character):
    """Print the active buff list for the character."""
    buffs = char.buffs
    if not buffs:
        print(f"[BUFFS]  (none)  limit={char.buff_limit}")
        return
    print(f"\n[BUFFS]  {len(buffs)} active (limit={char.buff_limit})")
    print(f"  {'Skill ID':>10} {'Lv':>4} {'Duration':>8}")
    print(f"  {'-'*10} {'-'*4} {'-'*8}")
    for b in sorted(buffs, key=lambda b: b["duration"], reverse=True):
        print(f"  {b['skill_id']:>10} {b['skill_level']:>4} {b['duration']:>7}s")


# -- Login server flow --

def login_server_flow(sock: socket.socket):
    di = crypto.decrypt_init_packet(crypto.recv_packet(sock), crypto.L2MOBIUS_STATIC_KEY)
    assert di[0] == 0x00
    sid = struct.unpack_from("<I", di, 1)[0]
    rm = bytearray(di[9:9 + 128])
    crypto.unscramble_modulus(rm)
    dk = di[9 + 128 + 16:9 + 128 + 16 + 16]
    print(f"[LS]  Session: {sid}")
    crypto.send_encrypted(sock, packets.build_auth_game_guard(sid), dk)
    crypto.decrypt_l2_packet(crypto.recv_packet(sock), dk)
    print("[LS]  GGAuth OK")
    crypto.send_encrypted(sock, packets.build_request_auth_login(ACCOUNT_LOGIN, ACCOUNT_PASSWORD, sid, rm), dk)
    dr = crypto.decrypt_l2_packet(crypto.recv_packet(sock), dk)
    if dr[0] == 0x01:
        raise RuntimeError(f"Login failed: {dr[1]}")
    assert dr[0] == 0x03
    l1 = struct.unpack_from("<I", dr, 1)[0]
    l2 = struct.unpack_from("<I", dr, 5)[0]
    print("[LS]  LoginOk!")
    crypto.send_encrypted(sock, packets.build_request_server_list(l1, l2), dk)
    ds = crypto.decrypt_l2_packet(crypto.recv_packet(sock), dk)
    assert ds[0] == 0x04
    sc = ds[1]
    pos = 3
    servers = []
    for i in range(sc):
        si = ds[pos]; pos += 1
        ib = ds[pos:pos + 4]; pos += 4
        port = struct.unpack_from("<I", ds, pos)[0]; pos += 4
        pos += 1; pos += 1; pos += 2; pos += 2; pos += 1; pos += 4; pos += 1
        servers.append({'id': si, 'ip': f"{ib[0]}.{ib[1]}.{ib[2]}.{ib[3]}", 'port': port})
    ch = servers[SERVER_ID]
    print(f"[LS]  Selected: {ch['ip']}:{ch['port']}")
    crypto.send_encrypted(sock, packets.build_request_server_login(l1, l2, ch['id']), dk)
    dp = crypto.decrypt_l2_packet(crypto.recv_packet(sock), dk)
    assert dp[0] == 0x07
    p1 = struct.unpack_from("<I", dp, 1)[0]
    p2 = struct.unpack_from("<I", dp, 5)[0]
    print("[LS]  PlayOk!")
    return (p1, p2, ch['ip'], ch['port'], ACCOUNT_LOGIN, l1, l2)


# -- GS flow --

def game_server_flow(ip: str, port: int, login_name: str,
                     play_ok1: int, play_ok2: int,
                     login_ok1: int, login_ok2: int):
    gs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gs.settimeout(60.0)
    try:
        print(f"\n[GS] Connecting to {ip}:{port}...")
        gs.connect((ip, port))

        crypto.send_raw(gs, packets.build_proto_version(746))
        kp = crypto.recv_packet(gs)
        assert kp[0] == 0x00
        print(f"[GS]  KeyPacket: key={bytes(kp[2:10]).hex()}")

        crypto.send_raw(gs, packets.build_auth_login_gs(login_name, play_ok2, play_ok1, login_ok1, login_ok2))
        print("[GS]  AuthLogin sent")

        csi = bytearray(crypto.recv_packet(gs))
        if csi[0] == 0x14:
            raise RuntimeError(f"GS Auth failed, reason: {csi[1]}")
        assert csi[0] == 0x13
        cc = struct.unpack_from("<I", csi, 1)[0]
        print(f"[GS]  Characters: {cc}")
        if cc == 0:
            raise RuntimeError("No characters")

        pos = 5
        for i in range(cc):
            nb = bytearray()
            while True:
                if pos + 2 > len(csi):
                    break
                ch = struct.unpack_from("<H", csi, pos)[0]; pos += 2
                if ch == 0:
                    break
                nb += struct.pack("<H", ch)
            name = nb.decode('utf-16-le', errors='replace')
            oid = struct.unpack_from("<I", csi, pos)[0]; pos += 4
            print(f"[GS]   Slot {i}: '{name}' (objId={oid})")
            if i >= CHAR_SLOT:
                break
            while True:
                if pos + 2 > len(csi):
                    break
                if struct.unpack_from("<H", csi, pos)[0] == 0:
                    pos += 2
                    break
                pos += 2
            pos += 200

        print(f"[GS] Selecting character slot {CHAR_SLOT}...")
        crypto.send_raw(gs, packets.build_character_select(CHAR_SLOT))
        assert crypto.recv_packet(gs)[0] == 0x15
        print("[GS]  CharSelected OK")

        print("[GS] Sending EnterWorld...")
        crypto.send_raw(gs, packets.build_enter_world())
        print("[GS]  Entered world!\n")

        # -- Collect character data into L2Character --
        char = L2Character()
        received_user_info = False
        received_item_list = False
        received_skill_list = False
        received_etc_status = False
        received_buffs = False
        data_dump_printed = False

        ping_counter = 0
        last_ping = time.time()
        cur_x, cur_y, cur_z = 0, 0, 0
        got_coords = False
        data_collection_timeout = 20.0
        enter_time = time.time()

        # -- Combat state --
        current_target_id: int = 0
        state: str = "IDLE"          # IDLE | MOVING | SELECT | ATTACKING | LOOTING | DEAD
        attack_attempts: int = 0
        skill_used_on_target: bool = False  # True once Power Strike was cast on current target
        last_action = time.time()
        pending_loot: list[dict] = []  # items to pick up during LOOTING
        looting_since: float = 0.0     # timestamp when LOOTING started
        kills: int = 0                 # monsters killed so far
        MAX_KILLS = 5                  # stop after this many kills
        char_dead_since: float = 0.0   # timestamp when character died
        asked_for_help: bool = False   # True once help message was sent after death

        while True:
            try:
                if not data_dump_printed:
                    gs.settimeout(min(2.0, data_collection_timeout - (time.time() - enter_time)))
                else:
                    gs.settimeout(2.0)

                body = bytearray(crypto.recv_packet(gs))
                if len(body) == 0:
                    continue
                pid = body[0]

                # -- Data-collection phase --
                if not data_dump_printed:
                    if pid == 0x04:  # UserInfo
                        ui = packets.parse_user_info(body)
                        char.apply_user_info(ui)
                        received_user_info = True
                        cur_x, cur_y, cur_z = char.x, char.y, char.z
                        got_coords = True
                        print(f"[GS]  <- UserInfo parsed: {char.name} lvl {char.level} "
                              f"HP {char.cur_hp:.0f}/{char.max_hp} "
                              f"MP {char.cur_mp:.0f}/{char.max_mp} "
                              f"CP {char.cur_cp:.0f}/{char.max_cp} "
                              f"SP {char.sp} "
                              f"pos ({cur_x},{cur_y},{cur_z})")

                    elif pid == 0x1B:  # ItemList
                        char.apply_item_list(packets.parse_item_list(body))
                        received_item_list = True
                        print(f"[GS]  <- ItemList parsed: {len(char.items)} items")

                    elif pid == 0x58:  # SkillList
                        char.apply_skill_list(packets.parse_skill_list(body))
                        received_skill_list = True
                        print(f"[GS]  <- SkillList parsed: {len(char.skills)} skills")

                    elif pid == 0xF3:  # EtcStatusUpdate
                        char.apply_etc_status_update(packets.parse_etc_status_update(body))
                        received_etc_status = True
                        print(f"[GS]  <- EtcStatusUpdate parsed: "
                              f"charges={char.charges} "
                              f"weight_penalty={char.weight_penalty}")

                    elif pid == 0x0E:  # StatusUpdate
                        su = packets.parse_status_update(body)
                        char.apply_status_update(su)
                        char.radar.update_from_status(su)

                    elif pid == 0x16:  # NpcInfo
                        npc = packets.parse_npc_info(body)
                        char.radar.add_or_update(npc)

                    elif pid == 0x06:  # Die
                        die = packets.parse_die(body)
                        if die:
                            char.radar.remove(die["object_id"])

                    elif pid == 0x08:  # DeleteObject
                        do = packets.parse_delete_object(body)
                        if do:
                            char.radar.remove(do["object_id"])

                    elif pid == 0x7F:  # AbnormalStatusUpdate -- buffs
                        ab = packets.parse_abnormal_status_update(body)
                        char.apply_abnormal_status_update(ab)
                        received_buffs = True
                        _print_buffs(char)

                    elif pid == 0x0B:  # SpawnItem -- queue for pickup
                        si = packets.parse_spawn_item(body)
                        if si:
                            pending_loot.append(si)
                            iname = item_data.get_item_name(si["item_id"])
                            print(f"[GS]  <- SpawnItem: {iname} ({si['item_id']}) "
                                  f"x{si['count']} objId={si['object_id']} "
                                  f"(queued #{len(pending_loot)})")

                    elif pid == 0x0C:  # DropItem -- queue for pickup
                        di = packets.parse_drop_item(body)
                        if di:
                            pending_loot.append(di)
                            iname = item_data.get_item_name(di["item_id"])
                            print(f"[GS]  <- DropItem: {iname} ({di['item_id']}) "
                                  f"x{di['count']} objId={di['object_id']} "
                                  f"(queued #{len(pending_loot)})")

                    elif pid == 0x0D:  # GetItem -- loot confirmation
                        if len(body) >= 13:
                            picker_id = struct.unpack_from("<i", body, 1)[0]
                            item_id = struct.unpack_from("<i", body, 5)[0]
                            iname = item_data.get_item_name(item_id)
                            print(f"[GS]  <- GetItem: picker={picker_id} {iname} ({item_id})")

                    elif pid == 0xA8:  # NetPing
                        crypto.send_raw(gs, packets.build_net_ping(struct.unpack_from("<I", body, 1)[0]))
                        ping_counter += 1
                        last_ping = time.time()

                    elif pid == 0xF9:  # GameGuard
                        gg = bytearray([0xCA]) + struct.pack("<I", 0)
                        crypto.send_raw(gs, packets.pack(gg))

                    elif pid == 0x26:  # ServerClose
                        print("[GS]  [!] Server closing!")
                        break

                    all_core = (received_user_info and received_item_list and
                                received_skill_list and received_etc_status and
                                received_buffs)
                    timed_out = (time.time() - enter_time) > data_collection_timeout
                    if all_core or timed_out:
                        if timed_out and not all_core:
                            print("[GS]  !! Data collection timed out, printing partial data.")
                        char.dump()
                        data_dump_printed = True
                        last_ping = time.time()
                        # Auto-enable soulshots
                        print(f"[GS]  [*] Weapon: {char.equipped_weapon_id} "
                              f"(paperdoll RHAND: {char.paperdoll_display.get('RHAND', '?')}, "
                              f"RHAND2: {char.paperdoll_display.get('RHAND2', '?')})")
                        ss_in_inv = [i["item_id"] for i in char.items
                                     if i["item_id"] in char.ALL_SOULSHOT_IDS and i["count"] > 0]
                        print(f"[GS]  [*] Soulshots in inv: {ss_in_inv}")
                        char.enable_auto_soulshot(gs)
                        print("\n[GS]  -- Game loop started --\n")
                    continue

                # -- Post-collection: game loop --

                if pid == 0x04 and len(body) >= 13:  # UserInfo update
                    cur_x = struct.unpack_from("<i", body, 1)[0]
                    cur_y = struct.unpack_from("<i", body, 5)[0]
                    cur_z = struct.unpack_from("<i", body, 9)[0]
                    char.x, char.y, char.z = cur_x, cur_y, cur_z
                    got_coords = True
                elif pid == 0x16:  # NpcInfo
                    npc = packets.parse_npc_info(body)
                    char.radar.add_or_update(npc)
                elif pid == 0x0E:  # StatusUpdate
                    su = packets.parse_status_update(body)
                    char.radar.update_from_status(su)
                elif pid == 0x06:  # Die
                    die = packets.parse_die(body)
                    if die:
                        oid = die["object_id"]
                        char.radar.remove(oid)
                        if oid == char.object_id:
                            # Character died
                            print("[GS]  [!] Character died!")
                            state = "DEAD"
                            char_dead_since = time.time()
                            asked_for_help = False
                            pending_loot.clear()
                            current_target_id = 0
                            attack_attempts = 0
                            skill_used_on_target = False
                            last_action = time.time()
                        elif oid == current_target_id:
                            print(f"[GS]  <- Die: target {current_target_id} killed!")
                            current_target_id = 0
                            state = "LOOTING"
                            looting_since = time.time()
                            kills += 1
                            attack_attempts = 0
                            last_action = time.time()
                            print(f"[GS]  Kills: {kills}/{MAX_KILLS}")
                            # Use Healing Potion (itemId=1061) if HP is below 70%
                            potion = next((i for i in char.items
                                          if i["item_id"] == 1061 and i["count"] > 0), None)
                            if potion and char.max_hp > 0 and char.cur_hp / char.max_hp < 0.7:
                                iname = item_data.get_item_name(1061)
                                print(f"[GS]  -> UseItem: {iname} (objectId={potion['object_id']}) "
                                      f"HP {char.cur_hp:.0f}/{char.max_hp}")
                                crypto.send_raw(gs, packets.build_use_item(potion["object_id"]))
                            if kills >= MAX_KILLS:
                                print(f"[GS]  [DONE] Reached {MAX_KILLS} kills -- stopping")
                                return
                elif pid == 0x08:  # DeleteObject
                    do = packets.parse_delete_object(body)
                    if do:
                        oid = do["object_id"]
                        char.radar.remove(oid)
                        if oid == current_target_id:
                            current_target_id = 0
                            state = "IDLE"
                elif pid == 0x7F:  # AbnormalStatusUpdate -- buffs
                    ab = packets.parse_abnormal_status_update(body)
                    char.apply_abnormal_status_update(ab)
                elif pid == 0x0B:  # SpawnItem -- queue for pickup
                    si = packets.parse_spawn_item(body)
                    if si:
                        pending_loot.append(si)
                        iname = item_data.get_item_name(si["item_id"])
                        print(f"[GS]  <- SpawnItem: {iname} ({si['item_id']}) "
                              f"x{si['count']} objId={si['object_id']} "
                              f"(queued #{len(pending_loot)})")
                elif pid == 0x0C:  # DropItem -- queue for pickup
                    di = packets.parse_drop_item(body)
                    if di:
                        pending_loot.append(di)
                        iname = item_data.get_item_name(di["item_id"])
                        print(f"[GS]  <- DropItem: {iname} ({di['item_id']}) "
                              f"x{di['count']} objId={di['object_id']} "
                              f"(queued #{len(pending_loot)})")
                elif pid == 0x0D:  # GetItem -- loot confirmation
                    if len(body) >= 13:
                        item_id = struct.unpack_from("<i", body, 5)[0]
                        iname = item_data.get_item_name(item_id)
                        print(f"[GS]  <- GetItem: {iname} ({item_id}) collected")

                # Protocol handlers
                if pid == 0xA8:
                    crypto.send_raw(gs, packets.build_net_ping(struct.unpack_from("<I", body, 1)[0]))
                elif pid == 0x26:
                    print("[GS]  [!] Server closing!"); break
                elif pid == 0xF9:
                    gg = bytearray([0xCA]) + struct.pack("<I", 0)
                    crypto.send_raw(gs, packets.pack(gg))
                elif pid == 0x4A:
                    name, msg = packets.parse_creature_say(body)
                    print(f"[GS]      [{name}]: {msg}")
                elif pid == 0x64 and len(body) >= 13:
                    # SysMsg damage log
                    msg_id = struct.unpack_from("<I", body, 1)[0]
                    param_count = struct.unpack_from("<I", body, 5)[0]
                    pos = 9
                    params = []
                    for _ in range(param_count):
                        if pos + 4 > len(body):
                            break
                        ptype = struct.unpack_from("<I", body, pos)[0]; pos += 4
                        if ptype == 1:  # TYPE_INT_NUMBER
                            if pos + 4 > len(body):
                                break
                            val = struct.unpack_from("<I", body, pos)[0]; pos += 4
                            params.append(str(val))
                        elif ptype == 12:  # TYPE_PLAYER_NAME
                            s, pos = packets._read_utf16le_string(body, pos)
                            params.append(s)
                        elif ptype == 0:  # TYPE_TEXT
                            s, pos = packets._read_utf16le_string(body, pos)
                            params.append(s)
                        elif ptype == 6:  # TYPE_LONG_NUMBER
                            if pos + 8 > len(body):
                                break
                            pos += 8  # skip
                        else:
                            if pos + 4 > len(body):
                                break
                            pos += 4  # skip int-based types
                    if msg_id == 35:
                        # "You hit for $s1 damage."
                        dmg = params[0] if params else "?"
                        print(f"[GS]  DMG dealt -> target: {dmg}")
                    elif msg_id in (36, 37):
                        # "$s1 hit you for $s2 damage."
                        name = params[0] if len(params) > 0 else "?"
                        dmg = params[1] if len(params) > 1 else "?"
                        print(f"[GS]  DMG taken <- {name}: {dmg}")
                elif pid == 0xED and state == "DEAD" and len(body) >= 5:
                    # ConfirmDlg: opcode at [0], message ID int at [1]
                    msg_id = struct.unpack_from("<I", body, 1)[0]
                    print(f"[GS]  <- ConfirmDlg id={msg_id} (len={len(body)})")
                    if msg_id == 1510 or msg_id == 322:
                        # 1510 = "$s1 is making an attempt at resurrection..."
                        # 322 = "Do you wish to be resurrected?"
                        print("[GS]     -> Accepting resurrection!")
                        crypto.send_raw(gs, packets.build_dlg_answer(msg_id, 1, 0))
                        state = "IDLE"
                        char_dead_since = 0.0
                elif pid == 0x64 and state == "DEAD" and len(body) >= 5:
                    # SysMsg (fallback): opcode at [0], message ID int at [1]
                    msg_id = struct.unpack_from("<I", body, 1)[0]
                    print(f"[GS]  SysMsg id={msg_id} (len={len(body)})")

                # Ping every 5s
                if time.time() - last_ping > 5:
                    crypto.send_raw(gs, packets.build_net_ping(ping_counter))
                    ping_counter += 1
                    last_ping = time.time()

                # -- Radar maintenance --
                if got_coords:
                    char.radar.prune(cur_x, cur_y)

                # -- Combat state machine (DEAD state runs regardless of got_coords) --
                if state == "DEAD":
                    dead_elapsed = time.time() - char_dead_since
                    if not asked_for_help and dead_elapsed > 1.0:
                        crypto.send_raw(gs, packets.build_say2("help me, i died! please resurrect!", chat_type=0))
                        print("[GS]  -> Chat: asked for help (all chat)")
                        asked_for_help = True
                        last_action = time.time()
                    elif dead_elapsed > 60:
                        print("[GS]  -> Returning to town...")
                        crypto.send_raw(gs, packets.build_request_restart_point(0))
                        return

                if got_coords and time.time() - last_action >= 1.0 and state != "DEAD":
                    if state == "LOOTING":
                        # Pick up queued drops: move to nearest, then send Action
                        if pending_loot:
                            nearest = min(pending_loot,
                                         key=lambda si: (cur_x - si["x"])**2 + (cur_y - si["y"])**2)
                            dist = ((cur_x - nearest["x"])**2 + (cur_y - nearest["y"])**2) ** 0.5
                            if dist < 80:
                                iname = item_data.get_item_name(nearest["item_id"])
                                print(f"[GS]  -> Pickup loot: {iname} ({nearest['item_id']}) "
                                      f"x{nearest['count']} objId={nearest['object_id']}")
                                crypto.send_raw(gs, packets.build_action(
                                    nearest["object_id"], nearest["x"], nearest["y"], nearest["z"],
                                    shift_click=False))
                                pending_loot.remove(nearest)
                                last_action = time.time()
                            else:
                                crypto.send_raw(gs, packets.build_move_to_location(
                                    nearest["x"], nearest["y"], nearest["z"],
                                    cur_x, cur_y, cur_z, 1))
                                cur_x, cur_y = nearest["x"], nearest["y"]
                                last_action = time.time()
                        elif time.time() - looting_since > 5:
                            state = "IDLE"

                    elif state == "MOVING":
                        if current_target_id and current_target_id in char.radar.entries:
                            t = char.radar.entries[current_target_id]
                            dist = ((cur_x - t["x"]) ** 2 + (cur_y - t["y"]) ** 2) ** 0.5
                            if dist < 60:
                                print(f"[GS]  -> Select target: '{t['name']}' objId={current_target_id}")
                                crypto.send_raw(gs, packets.build_action(
                                    current_target_id, cur_x, cur_y, cur_z, shift_click=False))
                                state = "SELECT"
                                last_action = time.time()
                            else:
                                crypto.send_raw(gs, packets.build_move_to_location(
                                    t["x"], t["y"], t["z"], cur_x, cur_y, cur_z, 1))
                                cur_x, cur_y = t["x"], t["y"]
                                last_action = time.time()
                        else:
                            current_target_id = 0
                            state = "IDLE"

                    elif state == "SELECT":
                        if time.time() - last_action > 0.3:
                            print(f"[GS]  -> AttackRequest on {current_target_id}")
                            crypto.send_raw(gs, packets.build_attack_request(
                                current_target_id, cur_x, cur_y, cur_z))
                            attack_attempts = 1
                            skill_used_on_target = False
                            state = "ATTACKING"
                            last_action = time.time()

                    elif state == "ATTACKING":
                        if current_target_id and current_target_id not in char.radar.entries:
                            current_target_id = 0
                            attack_attempts = 0
                            skill_used_on_target = False
                            state = "IDLE"
                        elif not skill_used_on_target and time.time() - last_action > 0.4:
                            # Cast Power Strike (skill id 3) once per target after starting attack
                            print(f"[GS]  -> UseSkill Power Strike (id=3) on {current_target_id}")
                            crypto.send_raw(gs, packets.build_use_skill(3, ctrl_pressed=True))
                            skill_used_on_target = True
                            last_action = time.time()
                        elif time.time() - last_action > 3:
                            attack_attempts += 1
                            if attack_attempts > 10:
                                print(f"[GS]  !! Giving up on {current_target_id} after 10 attempts")
                                current_target_id = 0
                                attack_attempts = 0
                                skill_used_on_target = False
                                state = "IDLE"
                            else:
                                print(f"[GS]  -> AttackRequest retry #{attack_attempts} on {current_target_id}")
                                crypto.send_raw(gs, packets.build_attack_request(
                                    current_target_id, cur_x, cur_y, cur_z))
                                last_action = time.time()

                    elif state == "IDLE":
                        # Find nearest mob with level < character level
                        my_lvl = char.level
                        best = None
                        best_dist = float("inf")
                        for oid, mob in char.radar.entries.items():
                            if mob["level"] <= 0 or mob["level"] >= my_lvl:
                                continue
                            d = (cur_x - mob["x"]) ** 2 + (cur_y - mob["y"]) ** 2
                            if d < best_dist:
                                best_dist = d
                                best = mob
                        if best:
                            current_target_id = best["object_id"]
                            print(f"[GS]  Targeting: '{best['name']}' lvl {best['level']} "
                                  f"objId={best['object_id']} dist={best_dist**0.5:.0f}")
                            state = "MOVING"
                            last_action = time.time()
                        else:
                            dx = random.randint(-400, 400)
                            dy = random.randint(-400, 400)
                            crypto.send_raw(gs, packets.build_move_to_location(
                                cur_x + dx, cur_y + dy, cur_z, cur_x, cur_y, cur_z, 1))
                            cur_x += dx
                            cur_y += dy
                            last_action = time.time()

            except socket.timeout:
                if data_dump_printed and time.time() - last_ping > 5:
                    crypto.send_raw(gs, packets.build_net_ping(ping_counter))
                    ping_counter += 1
                    last_ping = time.time()
                if not data_dump_printed and (time.time() - enter_time) > data_collection_timeout:
                    print("[GS]  !! Data collection timed out, printing partial data.")
                    char.dump()
                    data_dump_printed = True
                    last_ping = time.time()
                    print("\n[GS]  -- Game loop started --\n")
            except ConnectionError:
                print("[GS]  Connection lost."); break

    except KeyboardInterrupt:
        print("\n[GS]  Disconnected by user.")
    finally:
        gs.close()


# -- Main --

def main():
    npc_data.load_npc_names()
    item_data.load_item_data()
    skill_data.load_skill_data()
    ls_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls_sock.settimeout(10.0)
    try:
        print(f"[+] Connecting to {SERVER_IP}:{LOGIN_PORT} as '{ACCOUNT_LOGIN}'...")
        ls_sock.connect((SERVER_IP, LOGIN_PORT))
        p1, p2, gs_ip, gs_port, ln, l1, l2 = login_server_flow(ls_sock)
        ls_sock.close()
        game_server_flow(gs_ip, gs_port, ln, p1, p2, l1, l2)
    except Exception as e:
        print(f"[-] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ls_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
