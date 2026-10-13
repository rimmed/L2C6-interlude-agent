#!/usr/bin/env python3
# L2J Mobius Interlude client bot -- connects, enters world, reads full character data.


import argparse
import json
import logging
import random
import signal
import socket
import struct
import time

from l2_client import crypto, packets
from l2_client.character import L2Character
from l2_client.connection import GameConnection, login
from l2_client.packets import GameServerPacket
from l2_client.tools.data import item_data, npc_data, skill_data

# --- CONFIGURATION ---
SERVER_IP = "192.168.1.200"
LOGIN_PORT = 2106
ACCOUNT_LOGIN = "gather7"
ACCOUNT_PASSWORD = "S!lver=drag0n"
SERVER_ID = 0
CHAR_SLOT = 0

# --- MODEL MODE (--model) ---
HEALING_POTION_IDS = (1539, 1061, 1060)  # Greater, normal, Lesser Healing Potion (best first)
MODEL_ATTACK_REPEAT_S = 1.5    # don't re-send AttackRequest on the same target more often than this
MODEL_APPROACH_RANGE = 100     # "moved" walks to the target only when farther than this
MODEL_ROAM_STEP = 400          # "moved" with nothing to walk to: random step of up to this many units
MODEL_COOLDOWN_SYNC_S = 1.0    # RequestSkillCoolTime at most this often while a skill cools down
# Mystic classes (PlayerClass.isMage() on the server). For them a click on the already selected
# monster is a weapon hit, which the casters in the training data almost never do.
MAGE_CLASS_IDS = frozenset((10, 11, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42, 43,
                            49, 50, 51, 52, 94, 95, 96, 97, 98, 103, 104, 105, 110, 111, 112, 115, 116))
MODEL_CAST_MARGIN_S = 0.15     # extra wait after the server-announced cast time before the next decision
MODEL_PICKUP_COMMIT_S = 3.0    # after a pickup command, hold new decisions this long while the item is still there
MODEL_STUCK_TARGET_S = 15.0    # give up a target that took no hit for this long (unreachable / behind a wall)
MODEL_IGNORE_TARGET_S = 60.0   # ...and keep it off the radar for this long


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

def _find_healing_potion(char: L2Character) -> dict | None:
    """Best healing potion in the inventory, or ``None``."""
    for potion_id in HEALING_POTION_IDS:
        potion = next((i for i in char.items if i["item_id"] == potion_id and i["count"] > 0), None)
        if potion:
            return potion
    return None


# -- Model mode: decision executor --

def _execute_model_decision(gs: socket.socket, char: L2Character, policy, decision,
                            current_target_id: int, pending_loot: list[dict],
                            cur_x: int, cur_y: int, cur_z: int,
                            attack_memo: dict) -> tuple[int, int, int]:
    """
    Turn a :class:`l2_client.policy.Decision` into game packets.

    :param attack_memo: last AttackRequest (``target``/``at``) and pickup (``loot``/``loot_at``), used
        to avoid re-sending them on every tick (the model predicts ``attacked`` for every swing)
    :returns: updated ``(current_target_id, cur_x, cur_y)``
    """
    now = time.time()
    radar = char.radar.entries
    target_alive = bool(current_target_id) and current_target_id in radar
    mob_id = decision.mob_object_id
    action = decision.action

    def select(oid: int) -> int:
        mob = radar[oid]
        print(f"[MODEL] -> Select target: '{mob['name']}' lvl {mob['level']} objId={oid}")
        crypto.send_raw(gs, packets.build_action(oid, cur_x, cur_y, cur_z, shift_click=False))
        return oid

    if action == "selected_target" and mob_id is not None:
        if mob_id != current_target_id:
            current_target_id = select(mob_id)
        elif char.class_id in MAGE_CLASS_IDS:
            pass  # a caster keeps its target and waits for the next cast instead of meleeing
        elif attack_memo.get("target") != mob_id or now - attack_memo.get("at", 0.0) >= MODEL_ATTACK_REPEAT_S:
            # A click on the already selected monster is an attack in L2 - that is what the logged
            # mid-fight "selected_target" rows are.
            print(f"[MODEL] -> Action (attack) on selected target {mob_id}")
            crypto.send_raw(gs, packets.build_action(mob_id, cur_x, cur_y, cur_z, shift_click=False))
            attack_memo.update(target=mob_id, at=now)
            policy.note_combat(now)

    elif action == "attacked":
        target = current_target_id if target_alive else mob_id
        if target is not None:
            repeat = attack_memo.get("target") == target and now - attack_memo.get("at", 0.0) < MODEL_ATTACK_REPEAT_S
            if not repeat:
                print(f"[MODEL] -> AttackRequest on {target}")
                crypto.send_raw(gs, packets.build_attack_request(target, cur_x, cur_y, cur_z))
                attack_memo.update(target=target, at=now)
            policy.note_combat(now)
            current_target_id = target

    elif action == "used_skill" and decision.skill_id is not None:
        if not target_alive and mob_id is not None:
            current_target_id = select(mob_id)  # a skill needs a selected target
        name = skill_data.get_skill_name(decision.skill_id) or "?"
        print(f"[MODEL] -> UseSkill {name} (id={decision.skill_id}) on {current_target_id}")
        crypto.send_raw(gs, packets.build_use_skill(decision.skill_id, ctrl_pressed=True))
        policy.note_skill_used(decision.skill_id, now)
        policy.note_combat(now)
        crypto.send_raw(gs, packets.build_request_skill_cool_time())

    elif action == "picked_up" and pending_loot:
        # Action on an item makes the server walk there and pick it up (the logged pickups are
        # "select item" + "pickup" for that reason); the entry leaves the queue on DeleteObject.
        nearest = min(pending_loot, key=lambda si: (cur_x - si["x"]) ** 2 + (cur_y - si["y"]) ** 2)
        if attack_memo.get("loot") != nearest["object_id"] or now - attack_memo.get("loot_at", 0.0) > 2.0:
            iname = item_data.get_item_name(nearest["item_id"])
            print(f"[MODEL] -> Pickup loot: {iname} ({nearest['item_id']}) x{nearest['count']}")
            crypto.send_raw(gs, packets.build_action(
                nearest["object_id"], nearest["x"], nearest["y"], nearest["z"], shift_click=False))
            attack_memo.update(loot=nearest["object_id"], loot_at=now, pickup_until=now + MODEL_PICKUP_COMMIT_S)

    elif action == "used_item":
        potion = _find_healing_potion(char)
        if potion:
            print(f"[MODEL] -> UseItem: {item_data.get_item_name(potion['item_id'])} "
                  f"HP {char.cur_hp:.0f}/{char.max_hp}")
            crypto.send_raw(gs, packets.build_use_item(potion["object_id"]))

    elif action == "moved":
        goal_id = current_target_id if target_alive else mob_id
        goal = radar.get(goal_id) if goal_id is not None else None
        if goal and ((cur_x - goal["x"]) ** 2 + (cur_y - goal["y"]) ** 2) ** 0.5 > MODEL_APPROACH_RANGE:
            crypto.send_raw(gs, packets.build_move_to_location(
                goal["x"], goal["y"], goal["z"], cur_x, cur_y, cur_z, 1))
            cur_x, cur_y = goal["x"], goal["y"]
        elif not goal:
            dx = random.randint(-MODEL_ROAM_STEP, MODEL_ROAM_STEP)
            dy = random.randint(-MODEL_ROAM_STEP, MODEL_ROAM_STEP)
            crypto.send_raw(gs, packets.build_move_to_location(
                cur_x + dx, cur_y + dy, cur_z, cur_x, cur_y, cur_z, 1))
            cur_x += dx
            cur_y += dy

    return current_target_id, cur_x, cur_y


# -- GS flow --

def game_server_flow(conn: GameConnection,
                     policy=None, decision_interval: float = 0.5,
                     max_kills: int = 5, run_stats: dict | None = None):
    """
    Enter the world and run the combat loop.

    :param conn: game-server connection (``l2_client.connection``); the loop below reads its socket directly

    :param policy: a :class:`l2_client.policy.CombatPolicy`; when given, the neural model decides
        every combat action instead of the scripted state machine
    :param decision_interval: seconds between two model decisions
    :param max_kills: stop after this many kills
    :param run_stats: filled while playing (``kills``, ``deaths``), so the caller keeps them however
        the session ends
    """
    if run_stats is None:
        run_stats = {}
    run_stats.update(kills=0, deaths=0)
    gs = conn.sock
    try:
        conn.enter_world()
        gs.settimeout(60.0)
        print(f"[GS]  Entered world as '{conn.character_name}'!\n")

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
        MAX_KILLS = max_kills          # stop after this many kills
        char_dead_since: float = 0.0   # timestamp when character died
        asked_for_help: bool = False   # True once help message was sent after death

        # -- Model mode state --
        last_decision = 0.0
        last_cooldown_sync = 0.0
        attack_memo: dict = {}
        ignored_targets: dict[int, float] = {}  # object_id -> ignored until (model mode)

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

                # -- Server-side moves (both phases): an RL-environment server teleports agents to their
                # farm zone on login/respawn and revives them in place of the "return to town" dialog.
                if pid == GameServerPacket.TELEPORT_TO_LOCATION:
                    tp = packets.parse_teleport_to_location(body)
                    if tp and tp["object_id"] == char.object_id:
                        print(f"[GS]  <- Teleported to ({tp['x']},{tp['y']},{tp['z']})")
                        crypto.send_raw(gs, packets.build_appearing())
                        cur_x, cur_y, cur_z = tp["x"], tp["y"], tp["z"]
                        char.x, char.y, char.z = cur_x, cur_y, cur_z
                        got_coords = True
                        # Everything known about the old surroundings is stale now.
                        char.radar.entries.clear()
                        pending_loot.clear()
                        current_target_id = 0
                        attack_attempts = 0
                        if state != "DEAD":
                            state = "IDLE"
                        if policy is not None:
                            policy.reset_session()
                    continue
                if pid == GameServerPacket.REVIVE:
                    rv = packets.parse_revive(body)
                    if rv and rv["object_id"] == char.object_id and state == "DEAD":
                        print("[GS]  <- Revived")
                        state = "IDLE"
                        char_dead_since = 0.0
                        last_action = time.time()
                    continue

                # -- Data-collection phase --
                if not data_dump_printed:
                    if pid == GameServerPacket.USER_INFO:
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

                    elif pid == GameServerPacket.ITEM_LIST:
                        char.apply_item_list(packets.parse_item_list(body))
                        received_item_list = True
                        print(f"[GS]  <- ItemList parsed: {len(char.items)} items")

                    elif pid == GameServerPacket.SKILL_LIST:
                        char.apply_skill_list(packets.parse_skill_list(body))
                        received_skill_list = True
                        print(f"[GS]  <- SkillList parsed: {len(char.skills)} skills")

                    elif pid == GameServerPacket.ETC_STATUS_UPDATE:
                        char.apply_etc_status_update(packets.parse_etc_status_update(body))
                        received_etc_status = True
                        print(f"[GS]  <- EtcStatusUpdate parsed: "
                              f"charges={char.charges} "
                              f"weight_penalty={char.weight_penalty}")

                    elif pid == GameServerPacket.STATUS_UPDATE:
                        su = packets.parse_status_update(body)
                        char.apply_status_update(su)
                        char.radar.update_from_status(su)

                    elif pid == GameServerPacket.NPC_INFO:
                        npc = packets.parse_npc_info(body)
                        char.radar.add_or_update(npc)

                    elif pid == GameServerPacket.DIE:
                        die = packets.parse_die(body)
                        if die:
                            char.radar.remove(die["object_id"])

                    elif pid == GameServerPacket.DELETE_OBJECT:
                        do = packets.parse_delete_object(body)
                        if do:
                            char.radar.remove(do["object_id"])

                    elif pid == GameServerPacket.ABNORMAL_STATUS_UPDATE:  # buffs
                        ab = packets.parse_abnormal_status_update(body)
                        char.apply_abnormal_status_update(ab)
                        received_buffs = True
                        _print_buffs(char)

                    elif pid == GameServerPacket.SPAWN_ITEM:  # queue for pickup
                        si = packets.parse_spawn_item(body)
                        if si:
                            pending_loot.append(si)
                            iname = item_data.get_item_name(si["item_id"])
                            print(f"[GS]  <- SpawnItem: {iname} ({si['item_id']}) "
                                  f"x{si['count']} objId={si['object_id']} "
                                  f"(queued #{len(pending_loot)})")

                    elif pid == GameServerPacket.DROP_ITEM:  # queue for pickup
                        di = packets.parse_drop_item(body)
                        if di:
                            pending_loot.append(di)
                            iname = item_data.get_item_name(di["item_id"])
                            print(f"[GS]  <- DropItem: {iname} ({di['item_id']}) "
                                  f"x{di['count']} objId={di['object_id']} "
                                  f"(queued #{len(pending_loot)})")

                    elif pid == GameServerPacket.GET_ITEM:  # loot confirmation
                        if len(body) >= 13:
                            picker_id = struct.unpack_from("<i", body, 1)[0]
                            item_id = struct.unpack_from("<i", body, 5)[0]
                            iname = item_data.get_item_name(item_id)
                            print(f"[GS]  <- GetItem: picker={picker_id} {iname} ({item_id})")

                    elif pid == GameServerPacket.GAME_GUARD_QUERY:
                        crypto.send_raw(gs, packets.build_game_guard_reply())

                    elif pid == GameServerPacket.SERVER_CLOSE:
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
                        if policy is not None:
                            policy.bind_character(char)
                        print("\n[GS]  -- Game loop started --\n")
                    continue

                # -- Post-collection: game loop --

                if pid == GameServerPacket.USER_INFO and len(body) >= 13:
                    # Full refresh, not just the position: level, class and stats change while playing
                    # (an RL-environment server boosts and promotes agents right after login).
                    prev_level, prev_class = char.level, char.class_id
                    char.apply_user_info(packets.parse_user_info(body))
                    cur_x, cur_y, cur_z = char.x, char.y, char.z
                    got_coords = True
                    if (char.level, char.class_id) != (prev_level, prev_class):
                        print(f"[GS]  <- UserInfo: lvl {char.level} class {char.class_id} "
                              f"HP {char.cur_hp:.0f}/{char.max_hp} MP {char.cur_mp:.0f}/{char.max_mp}")
                elif pid == GameServerPacket.NPC_INFO:
                    npc = packets.parse_npc_info(body)
                    if ignored_targets.get(npc.get("object_id"), 0.0) <= time.time():
                        char.radar.add_or_update(npc)
                elif pid == GameServerPacket.ITEM_LIST:  # full inventory resend (e.g. new equipment)
                    char.apply_item_list(packets.parse_item_list(body))
                elif pid == GameServerPacket.SKILL_LIST:  # skills changed (level-up, class transfer)
                    char.apply_skill_list(packets.parse_skill_list(body))
                    if policy is not None:
                        policy.bind_character(char)
                elif pid == GameServerPacket.STATUS_UPDATE:
                    su = packets.parse_status_update(body)
                    char.radar.update_from_status(su)
                    if policy is not None:
                        policy.update_self_status(char, su)
                elif pid == GameServerPacket.MAGIC_SKILL_USE and policy is not None:
                    # Our own cast started: a new command before it ends is refused by the server.
                    msu = packets.parse_magic_skill_use(body)
                    if msu and msu["caster_id"] == char.object_id:
                        attack_memo["cast_until"] = (time.time() + msu["hit_time"] / 1000.0 / policy.server_speed
                                                     + MODEL_CAST_MARGIN_S)
                elif pid == GameServerPacket.SKILL_COOL_TIME and policy is not None:  # real cooldowns
                    policy.apply_skill_cool_time(packets.parse_skill_cool_time(body))
                elif pid == GameServerPacket.DIE:
                    die = packets.parse_die(body)
                    if die:
                        oid = die["object_id"]
                        char.radar.remove(oid)
                        if oid == char.object_id:
                            # Character died
                            print("[GS]  [!] Character died!")
                            run_stats["deaths"] += 1
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
                            if policy is not None:
                                policy.record_event(char, current_target_id, (cur_x, cur_y))
                            current_target_id = 0
                            state = "LOOTING"
                            looting_since = time.time()
                            kills += 1
                            attack_attempts = 0
                            last_action = time.time()
                            print(f"[GS]  Kills: {kills}/{MAX_KILLS}")
                            run_stats["kills"] = kills
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
                elif pid == GameServerPacket.DELETE_OBJECT:
                    do = packets.parse_delete_object(body)
                    if do:
                        oid = do["object_id"]
                        char.radar.remove(oid)
                        if oid == current_target_id:
                            current_target_id = 0
                            state = "IDLE"
                        if policy is not None:
                            # Picked up (by anyone) or despawned: stale loot would keep the model's
                            # session open and its pickup choice valid forever.
                            pending_loot[:] = [si for si in pending_loot if si["object_id"] != oid]
                elif pid == GameServerPacket.ABNORMAL_STATUS_UPDATE:  # buffs
                    ab = packets.parse_abnormal_status_update(body)
                    char.apply_abnormal_status_update(ab)
                elif pid == GameServerPacket.SPAWN_ITEM:  # queue for pickup
                    si = packets.parse_spawn_item(body)
                    if si:
                        pending_loot.append(si)
                        iname = item_data.get_item_name(si["item_id"])
                        print(f"[GS]  <- SpawnItem: {iname} ({si['item_id']}) "
                              f"x{si['count']} objId={si['object_id']} "
                              f"(queued #{len(pending_loot)})")
                elif pid == GameServerPacket.DROP_ITEM:  # queue for pickup
                    di = packets.parse_drop_item(body)
                    if di:
                        pending_loot.append(di)
                        iname = item_data.get_item_name(di["item_id"])
                        print(f"[GS]  <- DropItem: {iname} ({di['item_id']}) "
                              f"x{di['count']} objId={di['object_id']} "
                              f"(queued #{len(pending_loot)})")
                elif pid == GameServerPacket.GET_ITEM:  # loot confirmation
                    if len(body) >= 13:
                        item_id = struct.unpack_from("<i", body, 5)[0]
                        iname = item_data.get_item_name(item_id)
                        print(f"[GS]  <- GetItem: {iname} ({item_id}) collected")

                # Protocol handlers.  Keep-alive is our own RequestNetPing every 5 s below; the server's
                # NetPing (0xD3) is only the answer to it and must not be answered again.
                if pid == GameServerPacket.SERVER_CLOSE:
                    print("[GS]  [!] Server closing!"); break
                elif pid == GameServerPacket.GAME_GUARD_QUERY:
                    crypto.send_raw(gs, packets.build_game_guard_reply())
                elif pid == GameServerPacket.CREATURE_SAY:
                    name, msg = packets.parse_creature_say(body)
                    print(f"[GS]      [{name}]: {msg}")
                elif pid == GameServerPacket.SYSTEM_MESSAGE and len(body) >= 13:
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
                        attack_memo["hit_at"] = time.time()
                        if policy is not None:
                            policy.note_combat()
                            policy.record_event(char, current_target_id, (cur_x, cur_y))
                    elif msg_id in (36, 37):
                        # "$s1 hit you for $s2 damage."
                        name = params[0] if len(params) > 0 else "?"
                        dmg = params[1] if len(params) > 1 else "?"
                        print(f"[GS]  DMG taken <- {name}: {dmg}")
                        if policy is not None:
                            policy.note_combat()
                            policy.record_event(char, current_target_id, (cur_x, cur_y))
                elif pid == GameServerPacket.CONFIRM_DLG and state == "DEAD" and len(body) >= 5:
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
                elif pid == GameServerPacket.SYSTEM_MESSAGE and state == "DEAD" and len(body) >= 5:
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

                if policy is not None:
                    # -- Model mode: the neural policy replaces the scripted state machine below --
                    if policy.wants_cooldown_sync() and time.time() - last_cooldown_sync >= MODEL_COOLDOWN_SYNC_S:
                        crypto.send_raw(gs, packets.build_request_skill_cool_time())
                        last_cooldown_sync = time.time()
                    # A pickup makes the character walk to the item; the next decision would cancel that walk
                    # (casters kill from afar, so their loot lies far away). Let the walk finish first.
                    picking_up = (time.time() < attack_memo.get("pickup_until", 0.0)
                                  and any(si["object_id"] == attack_memo.get("loot") for si in pending_loot))
                    casting = time.time() < attack_memo.get("cast_until", 0.0)
                    if (got_coords and state != "DEAD" and not picking_up and not casting
                            and time.time() - last_decision >= decision_interval):
                        last_decision = time.time()
                        # A target that takes no hit for a while is unreachable (geodata, walls): drop it,
                        # otherwise the policy keeps attacking it forever.
                        if attack_memo.get("engaged") != current_target_id:
                            attack_memo.update(engaged=current_target_id, engaged_at=last_decision)
                        last_progress = max(attack_memo.get("engaged_at", 0.0), attack_memo.get("hit_at", 0.0))
                        if current_target_id and last_decision - last_progress > MODEL_STUCK_TARGET_S:
                            print(f"[MODEL] !! No hit on {current_target_id} for {MODEL_STUCK_TARGET_S:.0f}s "
                                  f"-- ignoring it for {MODEL_IGNORE_TARGET_S:.0f}s")
                            ignored_targets[current_target_id] = last_decision + MODEL_IGNORE_TARGET_S
                            char.radar.remove(current_target_id)
                            current_target_id = 0
                            policy.reset_session()
                        has_potion = _find_healing_potion(char) is not None
                        decision = policy.decide(char, current_target_id, (cur_x, cur_y),
                                                 has_loot=bool(pending_loot), has_potion=has_potion)
                        print(f"[MODEL]  {decision.describe()}")
                        current_target_id, cur_x, cur_y = _execute_model_decision(
                            gs, char, policy, decision, current_target_id, pending_loot,
                            cur_x, cur_y, cur_z, attack_memo)
                        last_action = time.time()

                elif got_coords and time.time() - last_action >= 1.0 and state != "DEAD":
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
                    if policy is not None:
                        policy.bind_character(char)
                    print("\n[GS]  -- Game loop started --\n")
            except ConnectionError:
                print("[GS]  Connection lost."); break

    except KeyboardInterrupt:
        print("\n[GS]  Disconnected by user.")
    finally:
        gs.close()


# -- Main --

def parse_args():
    parser = argparse.ArgumentParser(description="L2J Mobius Interlude headless client bot.")
    parser.add_argument("--server-ip", default=SERVER_IP, help=f"Login server address (default: {SERVER_IP})")
    parser.add_argument("--account", default=ACCOUNT_LOGIN,
                        help=f"Account name (default: {ACCOUNT_LOGIN}); with AutoCreateAccounts a new name creates the account")
    parser.add_argument("--password", default=ACCOUNT_PASSWORD, help="Account password")
    parser.add_argument("--model", metavar="ONNX",
                        help="Drive combat with the trained policy (.onnx from l2_NN/export_onnx.py, "
                             "its .json must sit next to it) instead of the scripted state machine")
    parser.add_argument("--decision-interval", type=float, default=0.5,
                        help="Seconds between two model decisions (default: 0.5)")
    parser.add_argument("--server-speed", type=float, default=1.0,
                        help="Server TimeScale (General.ini); must match the server: converts the server's "
                             "game-time cooldowns and cast times to real time (default: 1 = no acceleration)")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="Where the model runs: cpu or gpu (CUDA) (default: cpu)")
    parser.add_argument("--max-kills", type=int, default=5, help="Stop after this many kills (default: 5)")
    parser.add_argument("--stats-out", metavar="JSON",
                        help="Write the run statistics (kills, deaths, model latency) to this file")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    npc_data.load_npc_names()
    item_data.load_item_data()
    skill_data.load_skill_data()

    policy = None
    if args.model:
        from l2_client.policy import CombatPolicy  # onnxruntime is only needed in model mode
        policy = CombatPolicy(args.model, server_speed=args.server_speed, device=args.device)
        print(f"[MODEL] Loaded {args.model} on {policy.device.upper()} {policy.session.get_providers()} "
              f"(seq_len {policy.seq_len}, trained epoch {policy.meta['training'].get('epoch')})")
    run_stats: dict = {}
    started = time.time()
    try:
        print(f"[+] Connecting to {args.server_ip}:{LOGIN_PORT} as '{args.account}'...")
        ticket = login(args.server_ip, args.account, args.password, port=LOGIN_PORT, server_index=SERVER_ID)
        game_server_flow(GameConnection(ticket, char_slot=CHAR_SLOT),
                         policy=policy, decision_interval=args.decision_interval,
                         max_kills=args.max_kills, run_stats=run_stats)
    except Exception as e:
        print(f"[-] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _report_run(args, policy, run_stats, time.time() - started)


def _report_run(args, policy, run_stats: dict, seconds: float):
    """Print (and optionally save) kills, deaths and the model latency of this run."""
    # A second Ctrl+C (or a signal forwarded by a wrapper such as `uv run`) must not lose the report.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    report = {"account": args.account, "device": args.device if policy else None,
              "max_kills": args.max_kills, "seconds": round(seconds, 1), **run_stats}
    if policy is not None:
        report["model"] = policy.timing_summary()
        inf, dec = report["model"]["inference_ms"], report["model"]["decide_ms"]
        print(f"\n[STATS] {policy.device.upper()}: {run_stats.get('kills', 0)} kills, "
              f"{run_stats.get('deaths', 0)} deaths in {seconds / 60:.1f} min")
        if inf.get("count"):
            print(f"[STATS] inference (session.run): mean {inf['mean']:.3f} ms, median {inf['median']:.3f}, "
                  f"p95 {inf['p95']:.3f}, p99 {inf['p99']:.3f}  (n={inf['count']}, "
                  f"first call {report['model']['first_call_ms']:.1f} ms)")
            print(f"[STATS] whole decision (features + inference): mean {dec['mean']:.3f} ms, "
                  f"median {dec['median']:.3f}, p95 {dec['p95']:.3f}")
    if args.stats_out:
        with open(args.stats_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[STATS] written to {args.stats_out}")


if __name__ == "__main__":
    main()
