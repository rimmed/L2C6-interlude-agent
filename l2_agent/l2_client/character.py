"""
Lineage2 C6 Interlude -- character data container and pretty-printer.

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

from . import packets
from .packets import PAPERDOLL_NAMES, STATUS_ATTRIBUTE_NAMES
from .radar import Radar
from .tools.data import item_data, skill_data


class L2Character:
    """Holds all character data received from the server."""

    def __init__(self):
        # -- UserInfo fields --
        self.name: str = ""
        self.title: str = ""
        self.object_id: int = 0
        self.class_id: int = 0
        self.base_class: int = 0
        self.race: int = 0
        self.sex: int = 0
        self.is_gm: int = 0
        self.is_noble: int = 0
        self.is_hero: int = 0
        self.level: int = 0
        self.exp: int = 0
        self.cur_hp: float = 0.0
        self.max_hp: int = 0
        self.cur_mp: float = 0.0
        self.max_mp: int = 0
        self.cur_cp: float = 0.0
        self.max_cp: int = 0
        self.sp: int = 0
        self.cur_load: int = 0
        self.max_load: int = 0
        self.inventory_limit: int = 0
        self.STR: int = 0
        self.DEX: int = 0
        self.CON: int = 0
        self.INT: int = 0
        self.WIT: int = 0
        self.MEN: int = 0
        self.p_atk: int = 0
        self.p_atk_spd: int = 0
        self.p_def: int = 0
        self.m_atk: int = 0
        self.m_atk_spd: int = 0
        self.m_def: int = 0
        self.evasion: int = 0
        self.accuracy: int = 0
        self.critical: int = 0
        self.run_spd: int = 0
        self.walk_spd: int = 0
        self.swim_run_spd: int = 0
        self.swim_walk_spd: int = 0
        self.fly_run_spd: int = 0
        self.fly_walk_spd: int = 0
        self.move_multiplier: float = 1.0
        self.atk_spd_multiplier: float = 1.0
        self.x: int = 0
        self.y: int = 0
        self.z: int = 0
        self.collision_radius: float = 0.0
        self.collision_height: float = 0.0
        self.is_running: int = 0
        self.is_fishing: int = 0
        self.fish_x: int = 0
        self.fish_y: int = 0
        self.fish_z: int = 0
        self.pvp_kills: int = 0
        self.pk_kills: int = 0
        self.pvp_flag: int = 0
        self.karma: int = 0
        self.clan_id: int = 0
        self.clan_crest_id: int = 0
        self.clan_crest_large_id: int = 0
        self.ally_id: int = 0
        self.ally_crest_id: int = 0
        self.clan_privileges: int = 0
        self.pledge_class: int = 0
        self.mount_type: int = 0
        self.mount_npc_id: int = 0
        self.private_store_type: int = 0
        self.recom_left: int = 0
        self.recom_have: int = 0
        self.cubics: list[int] = []
        self.hair_style: int = 0
        self.hair_color: int = 0
        self.face: int = 0
        self.name_color: int = 0
        self.title_color: int = 0

        # Paperdoll
        self.paperdoll: dict[str, int] = {}
        self.paperdoll_display: dict[str, int] = {}

        # -- Other packet data --
        self.items: list[dict] = []
        self.skills: list[dict] = []
        self.status_updates: list[dict] = []
        # -- EtcStatus fields --
        self.charges: int = 0
        self.weight_penalty: int = 0
        self.message_refusal: int = 0
        self.danger_area: int = 0
        self.weapon_armor_penalty: int = 0
        self.charm_of_courage: int = 0
        self.death_penalty_buff_level: int = 0

        # -- Radar (nearby NPCs) --
        self.radar = Radar()

        # -- Buff tracking --
        # Base buff limit = 20, +4 from Divine Inspiration (3rd class) = 24 max
        self.buff_limit: int = 20
        self.buffs: list[dict] = []  # [{skill_id, skill_level, duration_secs}, ...]

    # ------------------------------------------------------------------
    # Populate from parsed packet dicts
    # ------------------------------------------------------------------

    def apply_user_info(self, ui: dict):
        """Fill fields from a parsed UserInfo (0x04) dict."""
        if not ui:
            return
        for key in (
            "object_id", "class_id", "base_class", "race", "sex",
            "is_gm", "is_noble", "is_hero", "level", "exp",
            "max_hp", "max_mp", "sp", "cur_load", "max_load",
            "inventory_limit", "STR", "DEX", "CON", "INT", "WIT", "MEN",
            "p_atk", "p_atk_spd", "p_def", "m_atk", "m_atk_spd", "m_def",
            "evasion", "accuracy", "critical",
            "run_spd", "walk_spd", "swim_run_spd", "swim_walk_spd",
            "fly_run_spd", "fly_walk_spd",
            "x", "y", "z", "is_running", "is_fishing",
            "fish_x", "fish_y", "fish_z",
            "pvp_kills", "pk_kills", "pvp_flag", "karma",
            "clan_id", "clan_crest_id", "clan_crest_large_id",
            "ally_id", "ally_crest_id", "clan_privileges", "pledge_class",
            "mount_type", "mount_npc_id", "private_store_type",
            "recom_left", "recom_have",
            "hair_style", "hair_color", "face",
            "name_color", "title_color",
        ):
            setattr(self, key, ui.get(key, getattr(self, key, 0)))

        # Floats / strings
        self.name = ui.get("name", "")
        self.title = ui.get("title", "")
        self.cur_hp = ui.get("cur_hp", 0.0)
        self.cur_mp = ui.get("cur_mp", 0.0)
        self.cur_cp = ui.get("cur_cp", 0.0)
        self.max_cp = ui.get("max_cp", 0)
        self.move_multiplier = ui.get("move_multiplier", 1.0)
        self.atk_spd_multiplier = ui.get("atk_spd_multiplier", 1.0)
        self.collision_radius = ui.get("collision_radius", 0.0)
        self.collision_height = ui.get("collision_height", 0.0)
        self.cubics = ui.get("cubics", [])
        self.paperdoll = ui.get("paperdoll", {})
        self.paperdoll_display = ui.get("paperdoll_display", {})

    def apply_item_list(self, pl: dict):
        """Store items from a parsed ItemList (0x1B) dict."""
        self.items = pl.get("items", [])

    def apply_skill_list(self, sl: dict):
        """Store skills from a parsed SkillList (0x58) dict."""
        self.skills = sl.get("skills", [])

    def apply_status_update(self, su: dict):
        """Record a StatusUpdate (0x0E)."""
        self.status_updates.append(su)

    def apply_etc_status_update(self, es: dict):
        """Store fields from EtcStatusUpdate (0xF3)."""
        if not es:
            return
        self.charges = es.get("charges", 0)
        self.weight_penalty = es.get("weight_penalty", 0)
        self.message_refusal = es.get("message_refusal", 0)
        self.danger_area = es.get("danger_area", 0)
        self.weapon_armor_penalty = es.get("weapon_armor_penalty", 0)
        self.charm_of_courage = es.get("charm_of_courage", 0)
        self.death_penalty_buff_level = es.get("death_penalty_buff_level", 0)

    def apply_abnormal_status_update(self, ab: dict):
        """Replace the active buff list from AbnormalStatusUpdate (0x7F)."""
        self.buffs = ab.get("buffs", [])

    # ------------------------------------------------------------------
    # Auto soulshot
    # ------------------------------------------------------------------

    # All soulshot item IDs (no-grade through S)
    ALL_SOULSHOT_IDS = [1835, 1463, 1464, 1465, 1466, 1467]

    # Weapon grade -> recommended soulshot ID
    SOULSHOT_BY_GRADE = {
        0: 1835, 1: 1463, 2: 1464, 3: 1465, 4: 1466, 5: 1467,
    }

    @staticmethod
    def _item_grade(item_id: int) -> int:
        """
        Return crystal grade from the XML item datapack (0-5).

        Items without a ``crystal_type`` declaration default to grade 0
        (no-grade"), which covers beginner weapons and quest items.
        """
        g = item_data.get_item_grade(item_id)
        return g if g >= 0 else 0

    @property
    def equipped_weapon_id(self) -> int:
        return self.paperdoll_display.get("RHAND", 0) or self.paperdoll_display.get("RHAND2", 0)

    def enable_auto_soulshot(self, sock) -> bool:
        """
        Enable auto soulshot for the equipped weapon's grade.

        Uses the XML item datapack to determine the weapon's crystal grade.
        Prints diagnostic messages explaining why activation failed.
        """
        weapon_id = self.equipped_weapon_id
        if weapon_id <= 0:
            print("[GS]  !! No weapon equipped -- soulshot skipped")
            return False

        grade = self._item_grade(weapon_id)
        weapon_name = item_data.get_item_name(weapon_id) or f"id={weapon_id}"
        target_ss = self.SOULSHOT_BY_GRADE.get(grade)
        if not target_ss:
            print(f"[GS]  !! Unknown grade for weapon {weapon_id} (grade {grade})")
            return False

        # Check if the required soulshot type is in inventory
        has_correct = any(item["item_id"] == target_ss and item["count"] > 0 for item in self.items)
        if not has_correct:
            available = [item["item_id"] for item in self.items
                         if item["item_id"] in self.ALL_SOULSHOT_IDS and item["count"] > 0]
            if available:
                print(f"[GS]  !! Weapon '{weapon_name}' ({weapon_id}) is grade {grade} -- "
                      f"needs soulshot itemId={target_ss}")
                print(f"[GS]     You have soulshots: {available} -- wrong grade for this weapon")
            else:
                print(f"[GS]  !! No soulshots in inventory "
                      f"(weapon='{weapon_name}', needs itemId={target_ss})")
            return False

        sock.sendall(packets.build_auto_soulshot(target_ss, enable=True))
        print(f"[GS]  -> Auto soulshot enabled "
              f"(itemId={target_ss}, weapon='{weapon_name}', grade={grade})")
        return True

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def dump(self):
        print("\n" + "=" * 70)
        print("  CHARACTER DATA DUMP")
        print("=" * 70)

        # -- Basic Info --
        print(f"\n  Name:           {self.name}")
        print(f"  Title:          {self.title}")
        print(f"  Object ID:      {self.object_id}")
        print(f"  Class ID:       {self.class_id}")
        print(f"  Base Class:     {self.base_class}  (Race: {self.race}, Sex: {self.sex})")
        print(f"  Is GM:          {self.is_gm}")
        print(f"  Is Noble:       {self.is_noble}")
        print(f"  Is Hero:        {self.is_hero}")

        # -- Level & XP --
        print(f"\n  Level:          {self.level}")
        print(f"  EXP:            {self.exp}")

        # -- HP / MP / CP / SP --
        print(f"\n  HP:             {self.cur_hp:.0f} / {self.max_hp}")
        print(f"  MP:             {self.cur_mp:.0f} / {self.max_mp}")
        print(f"  CP:             {self.cur_cp:.0f} / {self.max_cp}")
        print(f"  SP:             {self.sp}")

        # -- Load --
        print(f"\n  Load:           {self.cur_load} / {self.max_load}")
        print(f"  Inventory Limit:{self.inventory_limit} slots")

        # -- Stats --
        print(f"\n  STR: {self.STR}   DEX: {self.DEX}   CON: {self.CON}")
        print(f"  INT: {self.INT}   WIT: {self.WIT}   MEN: {self.MEN}")

        # -- Combat Stats --
        print(f"\n  P.Atk:          {self.p_atk}")
        print(f"  P.Atk.Spd:      {self.p_atk_spd}")
        print(f"  P.Def:          {self.p_def}")
        print(f"  M.Atk:          {self.m_atk}")
        print(f"  M.Atk.Spd:      {self.m_atk_spd}")
        print(f"  M.Def:          {self.m_def}")
        print(f"  Evasion:        {self.evasion}")
        print(f"  Accuracy:       {self.accuracy}")
        print(f"  Critical:       {self.critical}")

        # -- Speed --
        print(f"\n  Run Spd:        {self.run_spd}")
        print(f"  Walk Spd:       {self.walk_spd}")
        print(f"  Swim Run Spd:   {self.swim_run_spd}")
        print(f"  Swim Walk Spd:  {self.swim_walk_spd}")
        print(f"  Move Multiplier:{self.move_multiplier:.4f}")
        print(f"  Atk.Spd Mult:   {self.atk_spd_multiplier:.4f}")

        # -- Position --
        print(f"\n  Position:       ({self.x}, {self.y}, {self.z})")
        print(f"  Collision:      radius={self.collision_radius:.1f}, height={self.collision_height:.1f}")
        print(f"  Running:        {self.is_running}")
        print(f"  Fishing:        {self.is_fishing}  (fish pos: {self.fish_x}, {self.fish_y}, {self.fish_z})")

        # -- PvP / Karma --
        print(f"\n  PvP Kills:      {self.pvp_kills}")
        print(f"  PK Kills:       {self.pk_kills}")
        print(f"  Karma:          {self.karma}")
        print(f"  PvP Flag:       {self.pvp_flag}")

        # -- Clan --
        print(f"\n  Clan ID:        {self.clan_id}")
        print(f"  Clan Crest:     {self.clan_crest_id}")
        print(f"  Large Crest:    {self.clan_crest_large_id}")
        print(f"  Ally ID:        {self.ally_id}")
        print(f"  Ally Crest:     {self.ally_crest_id}")
        print(f"  Clan Privileges:{self.clan_privileges}")
        print(f"  Pledge Class:   {self.pledge_class}")

        # -- Mount / Store --
        print(f"\n  Mount Type:     {self.mount_type}")
        print(f"  Mount NPC ID:   {self.mount_npc_id}")
        print(f"  Private Store:  {self.private_store_type}")

        # -- Recoms --
        print(f"\n  Recommends:     left={self.recom_left}, received={self.recom_have}")

        # -- Cubics --
        if self.cubics:
            print(f"\n  Cubics:         {self.cubics}")

        # -- Appearance --
        print(f"\n  Hair Style:     {self.hair_style}")
        print(f"  Hair Color:     {self.hair_color}")
        print(f"  Face:           {self.face}")
        print(f"  Name Color:     0x{self.name_color:08X}")
        print(f"  Title Color:    0x{self.title_color:08X}")

        # -- Etc Status --
        print("\n  [EtcStatusUpdate]")
        print(f"  Charges:        {self.charges}")
        print(f"  Weight Penalty: {self.weight_penalty}")
        print(f"  Msg Refusal:    {self.message_refusal}")
        print(f"  Danger Area:    {self.danger_area}")
        print(f"  Wpn/Arm Penalty:{self.weapon_armor_penalty}")
        print(f"  CharmOfCourage: {self.charm_of_courage}")
        print(f"  Death Penalty:  lvl {self.death_penalty_buff_level}")

        # -- Status Updates --
        if self.status_updates:
            for su in self.status_updates:
                attrs = su.get("attributes", {})
                if attrs:
                    print(f"\n  [StatusUpdate (objId={su.get('object_id')})]")
                    for aid, val in sorted(attrs.items()):
                        name = STATUS_ATTRIBUTE_NAMES.get(aid, f"0x{aid:02X}")
                        print(f"    {name}: {val}")

        # -- Equipped Items (Paperdoll) --
        equipped = []
        for slot_name in PAPERDOLL_NAMES:
            obj_id = self.paperdoll.get(slot_name, 0)
            disp_id = self.paperdoll_display.get(slot_name, 0)
            if obj_id != 0 or disp_id != 0:
                equipped.append(f"{slot_name}: objId={obj_id}, itemId={disp_id}")
        if equipped:
            print("\n  [Equipped Items (Paperdoll)]")
            for e in equipped:
                print(f"    {e}")

        # -- Buffs --
        print(f"\n  [Active Buffs -- {len(self.buffs)} / {self.buff_limit} slots]")
        if self.buffs:
            print(f"  {'Skill ID':>10} {'Lv':>4} {'Duration':>8}")
            print(f"  {'-'*10} {'-'*4} {'-'*8}")
            for b in sorted(self.buffs, key=lambda b: b["duration"], reverse=True):
                print(f"  {b['skill_id']:>10} {b['skill_level']:>4} {b['duration']:>7}s")
        else:
            print("    (none)")

        # -- Inventory --
        print(f"\n  [Inventory -- {len(self.items)} items]")
        if self.items:
            print(f"  {'ID':>10} {'Name':<38} {'ObjId':>10} {'Count':>7} {'Ench':>4} {'Eq':>2} {'Slot':>8}  {'Type':>6}")
            print(f"  {'-'*10} {'-'*38} {'-'*10} {'-'*7} {'-'*4} {'-'*2} {'-'*8}  {'-'*6}")
            for item in sorted(self.items, key=lambda i: (i['equipped'], i['item_id']), reverse=True):
                equip = "E" if item['equipped'] else "-"
                iname = item_data.get_item_name(item["item_id"])
                print(f"  {item['item_id']:>10} {iname:<38} {item['object_id']:>10} {item['count']:>7} {item['enchant']:>4} {equip:>2} 0x{item['body_part']:06X}  0x{item['type2']:04X}")
        else:
            print("    (empty)")

        # -- Skill List --
        print(f"\n  [Skill List -- {len(self.skills)} skills]")
        if self.skills:
            passive_count = sum(1 for s in self.skills if s['passive'])
            active_count = len(self.skills) - passive_count
            print(f"  Active: {active_count}, Passive: {passive_count}")
            print(f"  {'Skill ID':>10} {'Name':<38} {'Lv':>3} {'Type':>8} {'Disabled':>8}")
            print(f"  {'-'*10} {'-'*38} {'-'*3} {'-'*8} {'-'*8}")
            for sk in sorted(self.skills, key=lambda s: (s['passive'], s['id'])):
                stype = "passive" if sk['passive'] else "active"
                disabled = "YES" if sk['disabled'] else "no"
                sname = skill_data.get_skill_name(sk["id"])
                print(f"  {sk['id']:>10} {sname:<38} {sk['level']:>3} {stype:>8} {disabled:>8}")
        else:
            print("    (none)")

        print("=" * 70)
