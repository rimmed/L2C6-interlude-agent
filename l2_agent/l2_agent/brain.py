"""
The agent's plan for a new character on Talking Island, scripted for now. Each round picks the first thing that
is due, as a player starting the game would:

1. the tutorial errands (the gemstone to the Newbie Helper, the recommendation to the guild master);
2. the newbie buffs of the Newbie Guide when they are gone;
3. selling the loot and buying potions, shots and better gear when the bag is full or the potions run out;
4. the next quest of :data:`QUEST_ORDER` its level allows;
5. hunting at the hunting ground of its level otherwise.

What the character has done is kept in a small JSON file per character, so a restart goes on where it stopped.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from l2_client import packets

from .combat import Fighter
from .dialog import Talker
from .geomap import GeoMap
from .knowledge import DATA_DIR, TALKING_ISLAND, Area, Point
from .navigation import Navigator
from .quests import QuestRunner, run_plan
from .session import Session
from .trade import Trader
from .tutorial import Tutorial

log = logging.getLogger(__name__)

GEODATA = DATA_DIR / "l2jc6_geodata.parquet"
NEWBIE_HELPER = 30009  # Cedric's Training Hall (Human Fighter start)
NEWBIE_GUIDE = 30598  # Talking Island village
GRAND_MASTER_ROIEN = 30008
GROCER = 30004  # Katerina: potions and shots
WEAPON_MERCHANT = 30001  # Lector: no-grade fighter / mystic equipment

BLUE_GEM = 6353
RECOMMENDATION = 1067
LESSER_HEALING_POTION = 1060
SOULSHOT_NO_GRADE = 1835
SOULSHOT_BEGINNER = 5789
SPIRITSHOT_BEGINNER = 5790
SPIRITSHOT_NO_GRADE = 2509
#: the Newbie Guide's buffs (handlers/bypass/npc/SupportMagic.java)
NEWBIE_BUFF_LEVELS = (8, 24)
NEWBIE_BUFF_SKILLS = {4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331}

QUEST_ORDER = (1, 258, 257)
#: items the agent never sells
KEEP = {57, BLUE_GEM, RECOMMENDATION, LESSER_HEALING_POTION, SOULSHOT_NO_GRADE, SOULSHOT_BEGINNER, SPIRITSHOT_BEGINNER,
        SPIRITSHOT_NO_GRADE, 5588}


@dataclass
class BrainConfig:
    """Thresholds of the plan: when to shop and how much to buy."""

    area: Area = TALKING_ISLAND
    state_dir: Path = Path("checkpoints/agent")
    #: potions bought when fewer are left
    potions_min: int = 5
    potions_buy: int = 20
    shots_buy: int = 500
    #: adena kept back when buying gear (for potions and shots)
    reserve: int = 300
    #: a hunt lasts this long before the agent looks at its plan again
    hunt_round_s: float = 600.0


@dataclass
class State:
    """What the character has done, kept between runs."""

    done_quests: list[int] = field(default_factory=list)
    failed: dict[str, int] = field(default_factory=dict)


class Brain:
    """Picks and does the next step of the plan, round after round."""

    def __init__(self, session: Session, cfg: BrainConfig | None = None):
        self.session = session
        self.cfg = cfg or BrainConfig()
        self.nav = Navigator(session, geomap=GeoMap() if GEODATA.exists() else None)
        self.talker = Talker(session, self.nav, self.cfg.area)
        self.fighter = Fighter(session)
        self.trader = Trader(session, self.talker)
        self.quests = QuestRunner(session, self.talker, self.fighter, self.cfg.area)
        self.tutorial = Tutorial(session)
        self.state_path = self.cfg.state_dir / f"state_{session.cfg.account}.json"
        self.state = self._load()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _load(self) -> State:
        if self.state_path.exists():
            return State(**json.loads(self.state_path.read_text()))
        return State()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.__dict__, indent=1))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_minutes: float = 0.0) -> None:
        s = self.session
        self.tutorial.settle(3.0)
        self._wear_best()
        self._auto_shots()
        end = time.time() + max_minutes * 60 if max_minutes else float("inf")
        while time.time() < end:
            self.quests.refresh()
            step = self.next_step()
            log.info("=== %s (level %d, %d adena, %d potions, at %s)", step, s.char.level, s.adena,
                     s.item_count(LESSER_HEALING_POTION), s.position)
            ok = getattr(self, "do_" + step[0])(*step[1:])
            if not ok:
                key = "/".join(str(x) for x in step)
                self.state.failed[key] = self.state.failed.get(key, 0) + 1
                self._save()

    def next_step(self) -> tuple:
        s, st = self.session, self.state
        if s.item_count(BLUE_GEM):
            return ("errand", NEWBIE_HELPER)
        if s.item_count(RECOMMENDATION) and st.failed.get(f"errand/{GRAND_MASTER_ROIEN}", 0) < 3:
            return ("errand", GRAND_MASTER_ROIEN)
        if NEWBIE_BUFF_LEVELS[0] <= s.char.level <= NEWBIE_BUFF_LEVELS[1] and not self._has_newbie_buffs() \
                and st.failed.get("buffs", 0) < 3:
            return ("buffs",)
        if self._needs_shopping():
            return ("shop",)
        for quest_id in QUEST_ORDER:
            if quest_id in st.done_quests or st.failed.get(f"quest/{quest_id}", 0) >= 2:
                continue
            if s.char.level >= self.quests.book.facts(quest_id).min_level:
                return ("quest", quest_id)
        return ("hunt",)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def do_errand(self, npc_id: int) -> bool:
        """Talk to ``npc_id`` and follow its quest link: the tutorial NPCs take the item and give the reward."""
        s, before = self.session, list(self.session.char.items)
        dialog = self.talker.talk(npc_id)
        for _ in range(3):
            # A reply that is an event of a quest script ("I brought the recommendation") before the quest menu.
            link = self.talker.choose(dialog, r"^Script Q\d+\w* \S", r"_Quest Q\d+", r"_Quest$|^Script$|^Quest")
            if link is None:
                break
            dialog = self.talker.follow(link)
        s.idle(1.0)
        changed = sorted(i["item_id"] for i in s.char.items) != sorted(i["item_id"] for i in before)
        return changed

    def do_buffs(self) -> bool:
        s = self.session
        dialog = self.talker.talk(NEWBIE_GUIDE)
        for _ in range(3):
            # The page link first leads to the page with the actual "_SupportMagic" bypass.
            link = self.talker.choose(dialog, r"_SupportMagic$", r"SupportMagic")
            if link is None:
                break
            asked = time.time()
            dialog = self.talker.follow(link, expect=lambda: self._has_newbie_buffs()
                                        or (s.dialog is not None and s.dialog.received >= asked)) or dialog
            if self._has_newbie_buffs():
                break
        s.wait_until(self._has_newbie_buffs, 3.0)
        log.info("Buffs: %s", sorted(b["skill_id"] for b in s.char.buffs))
        return self._has_newbie_buffs()

    def do_shop(self) -> bool:
        s, cfg = self.session, self.cfg
        self._wear_best()  # rewards and drops first, so the better pieces are not sold
        self.trader.sell(GROCER, KEEP | self._equipped_ids())
        wanted = {}
        if s.item_count(LESSER_HEALING_POTION) < cfg.potions_min:
            wanted[LESSER_HEALING_POTION] = cfg.potions_buy
        shot = self._shot_item()
        if shot and s.item_count(shot) + s.item_count(SOULSHOT_BEGINNER) + s.item_count(SPIRITSHOT_BEGINNER) < 100:
            wanted[shot] = cfg.shots_buy
        if wanted:
            self.trader.buy(GROCER, wanted)
        self._upgrade_gear()
        self._auto_shots()
        return True

    def do_quest(self, quest_id: int) -> bool:
        ok = run_plan(self.quests, quest_id)
        if ok:
            log.info("Quest %d finished", quest_id)
            self.state.done_quests.append(quest_id)
            self._save()
        return ok

    def do_hunt(self) -> bool:
        s = self.session
        grounds = s.knowledge.hunting_grounds(self.cfg.area, s.char.level)
        if grounds.empty:
            log.warning("No hunting ground for level %d", s.char.level)
            return False
        x, y, _ = s.position
        grounds = grounds.assign(d=((grounds.cx - x) ** 2 + (grounds.cy - y) ** 2) ** 0.5)
        best = grounds.sort_values(["is_aggressive", "d"]).iloc[0]
        home = Point(int(best.cx), int(best.cy), int(best.cz) if best.cz == best.cz else s.position[2])
        log.info("Hunting %s (level %d) at %s", best["name"], best["level"], home)
        self.nav.move_to(home, arrive=600, timeout=900)
        level = s.char.level
        result = self.fighter.hunt(home, until=lambda: s.char.level > level or self._needs_shopping(),
                                   timeout=self.cfg.hunt_round_s, max_deaths=3)
        log.info("Hunt: %s", result)
        return result.deaths < 3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_newbie_buffs(self) -> bool:
        return any(b["skill_id"] in NEWBIE_BUFF_SKILLS for b in self.session.char.buffs)

    def _needs_shopping(self) -> bool:
        s = self.session
        bag = sum(1 for i in s.char.items if i["item_id"] not in KEEP and not i.get("equipped"))
        if self.state.failed.get("shop", 0) >= 3:
            return False
        low_potions = s.item_count(LESSER_HEALING_POTION) < self.cfg.potions_min and s.adena >= 500
        return bag >= 15 or low_potions

    def _equipped_ids(self) -> set[int]:
        return {i["item_id"] for i in self.session.char.items if i.get("equipped")}

    def _shot_item(self) -> int | None:
        s = self.session
        weapon = next((i for i in s.char.items if i.get("equipped") and i["item_id"] in s.knowledge.items.index
                       and s.knowledge.items.at[i["item_id"], "type"] == "Weapon"), None)
        if weapon is None:
            return None
        return SPIRITSHOT_NO_GRADE if _is_mage(s.char.class_id) else SOULSHOT_NO_GRADE

    def _auto_shots(self) -> None:
        s = self.session
        for shot in (SOULSHOT_BEGINNER, SOULSHOT_NO_GRADE, SPIRITSHOT_BEGINNER, SPIRITSHOT_NO_GRADE):
            if s.item_count(shot):
                s.send(packets.build_auto_soulshot(shot, True))

    def _wear_best(self) -> None:
        """Wear the strongest piece of the bag for each slot (quest rewards, drops) when it beats the worn one."""
        s, k = self.session, self.session.knowledge
        mage = _is_mage(s.char.class_id)
        worn = self._worn(mage)
        best: dict[str, tuple[float, dict]] = {}
        for item in s.char.items:
            if item.get("equipped") or item["item_id"] not in k.items.index:
                continue
            row = k.items.loc[item["item_id"]]
            if not _fits(row, mage):
                continue
            slot, strength = _slot(row), _strength(row, mage)
            if strength > worn.get(slot, 0.0) and strength > best.get(slot, (0.0, None))[0]:
                best[slot] = (strength, item)
        for slot, (_, item) in best.items():
            s.send(packets.build_use_item(item["object_id"]))
            log.info("Wearing %s (%s)", k.item_name(item["item_id"]), slot)
            s.idle(0.5)

    def _worn(self, mage: bool) -> dict[str, float]:
        k = self.session.knowledge
        # A worn piece the agent would not choose (a bow without arrows) counts as an empty slot.
        return {_slot(k.items.loc[i["item_id"]]): _strength(k.items.loc[i["item_id"]], mage)
                for i in self.session.char.items if i.get("equipped") and i["item_id"] in k.items.index
                and _fits(k.items.loc[i["item_id"]], mage)}

    def _upgrade_gear(self) -> None:
        """Buy the no-grade piece that adds most strength per adena over the worn one, then wear it."""
        s, k = self.session, self.session.knowledge
        mage = _is_mage(s.char.class_id)
        lists = k.buylists_of(WEAPON_MERCHANT)
        list_id = int(lists["list_id"].max() if mage else lists["list_id"].min())
        offer = lists[lists["list_id"] == list_id].merge(k.items, left_on="item_id", right_index=True)
        offer = offer[offer["crystal_type"].isna()]
        budget = s.adena - self.cfg.reserve
        worn = self._worn(mage)
        best = None
        for row in offer.itertuples():
            item = k.items.loc[row.item_id]
            if not _fits(item, mage):
                continue
            price = row.price_x if not pd.isna(row.price_x) else row.price_y
            gain = _strength(item, mage) - worn.get(_slot(item), 0.0)
            if price <= budget and gain > 0 and (best is None or gain / price > best[2] / best[1]):
                best = (int(row.item_id), int(price), gain)
        if best is None:
            return
        bought = self.trader.buy(WEAPON_MERCHANT, {best[0]: 1}, list_id)
        if bought.get(best[0]):
            self._wear_best()


#: one-handed weapon types a fighter of the starting classes uses (no bow without arrows, no two-handed)
FIGHTER_WEAPONS = {"SWORD", "BLUNT"}
_SLOTS = {"rhand": "weapon", "lrhand": "weapon", "rear;lear": "ear", "rfinger;lfinger": "finger", "onepiece": "chest"}


def _is_mage(class_id: int) -> bool:
    return class_id in (10, 25, 38, 49)


def _slot(item) -> str:
    return _SLOTS.get(item["bodypart"], item["bodypart"])


def _fits(item, mage: bool) -> bool:
    """Whether the agent would wear ``item``: armor of the body slots, its archetype's weapons."""
    if item["type"] == "Weapon":
        if mage:
            return bool(item["is_magic_weapon"])
        return item["weapon_type"] in FIGHTER_WEAPONS and item["bodypart"] == "rhand"
    return item["type"] == "Armor" and item["bodypart"] not in ("hair", "hairall", "underwear", "back", "alldress")


def _strength(item, mage: bool) -> float:
    stats = item["stats"] if isinstance(item["stats"], dict) else {}
    if item["type"] == "Weapon":
        return float(stats.get("mAtk" if mage else "pAtk", 0.0) or 0.0)
    return float((stats.get("pDef", 0.0) or 0.0) + (stats.get("mDef", 0.0) or 0.0))
