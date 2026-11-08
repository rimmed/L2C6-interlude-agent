"""
The agent's plan for a new character from Talking Island to level 20, scripted for now. Each round picks the first
thing that is due, as a player starting the game would:

1. the tutorial errands (the gemstone to the Newbie Helper, the recommendation to the guild master);
2. the skills its class may learn at its level, at a trainer of the class;
3. the newbie buffs of the Newbie Guide (levels 8-24) when they are gone;
4. selling the loot (materials and recipes go to the warehouse instead) and buying potions, shots and better
   gear when the bag is full or the potions run out;
5. the quests that start where it is and its level allows (a plan of :data:`quests.PLANS`, else the generic run);
6. from level 15 on Talking Island, the teleport to Gludin, whose hunting grounds go on to level 20;
7. hunting at the hunting ground of its level otherwise.

It stops at level 20. What the character has done is kept in a small JSON file per character, so a restart goes on
where it stopped; ``status_<account>.json`` beside it tells what it is doing (for the fleet's report).
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
from .quests import PLANS, QuestRunner, run_generic, run_plan
from .session import Session
from .skills import SkillTrainer
from .trade import Trader
from .travel import Traveller
from .tutorial import Tutorial

log = logging.getLogger(__name__)

GEODATA = DATA_DIR / "l2jc6_geodata.parquet"
GLUDIN = Area("Gludin", -100000, -60000, 130000, 175000)
#: the tutorial's Newbie Helpers (fighter hall, mystic hall) and guild masters (Roien, Gallint)
#: the Newbie Helper and the guild master of the tutorial: fighters' in the fighter hall, mystics' in the school
NEWBIE_HELPER = {"fighter": 30009, "mystic": 30019}
GUILD_MASTER = {"fighter": 30008, "mystic": 30017}
NEWBIE_GUIDE = 30598  # Talking Island village
GATEKEEPER_TI = 30006  # Roxxy
TRAVEL_LEVEL = 15
GOAL_LEVEL = 20
#: quests the quest loop leaves alone: the tutorial (it runs them itself) and those needing what the agent cannot do
#: yet (426 Quest for Fishing Shot: fishing)
SKIPPED_QUESTS = {255, 999, 426}
GREMLIN = 18342

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
    deaths: int = 0
    started: float = 0.0
    #: the tutorial's errands are over (the guild master gave the recommendation's gift)
    tutorial_done: bool = False
    #: the Newbie Helper gave the gemstone task
    helper_met: bool = False
    #: the blue gemstone was brought to the Newbie Helper
    gem_done: bool = False
    #: Warehouse NPCs without a private warehouse (freightmen)
    no_warehouse: list = field(default_factory=list)


class Brain:
    """Picks and does the next step of the plan, round after round."""

    def __init__(self, session: Session, cfg: BrainConfig | None = None):
        self.session = session
        self.cfg = cfg or BrainConfig()
        self.nav = Navigator(session, geomap=GeoMap() if GEODATA.exists() else None)
        self.talker = Talker(session, self.nav, self.cfg.area)
        self.fighter = Fighter(session)
        self.nav.guard = self._guard
        self.fighter.nav = self.nav
        self._guarding = False
        self.trader = Trader(session, self.talker)
        self.quests = QuestRunner(session, self.talker, self.fighter, self.cfg.area)
        self.trainer = SkillTrainer(session, self.talker, self.cfg.area)
        self.traveller = Traveller(session, self.talker)
        self.tutorial = Tutorial(session)
        self.state_path = self.cfg.state_dir / f"state_{session.cfg.account}.json"
        self.status_path = self.cfg.state_dir / "status" / f"{session.cfg.account}.json"
        self.state = self._load()
        self.trader.no_warehouse = set(self.state.no_warehouse)
        if not self.state.started:
            self.state.started = time.time()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _load(self) -> State:
        if self.state_path.exists():
            return State(**json.loads(self.state_path.read_text()))
        return State()

    def _save(self) -> None:
        self.state.no_warehouse = sorted(self.trader.no_warehouse)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.__dict__, indent=1))

    def _status(self, step: tuple, result: str = "") -> None:
        s = self.session
        status = {"account": s.cfg.account, "name": s.conn.character_name, "class_id": s.char.class_id,
                  "level": s.char.level, "exp": getattr(s.char, "exp", 0), "sp": getattr(s.char, "sp", 0),
                  "adena": s.adena, "area": self.area.name, "position": list(s.position), "step": list(step),
                  "result": result, "deaths": self.state.deaths + self.fighter.deaths,
                  "done_quests": self.state.done_quests, "skills": len(s.char.skills), "time": time.time(),
                  "started": self.state.started}
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        tmp.replace(self.status_path)

    @property
    def area(self) -> Area:
        """Where the character is: Talking Island, else Gludin."""
        x, y, _ = self.session.position
        return TALKING_ISLAND if TALKING_ISLAND.contains(x, y) else GLUDIN

    def _set_area(self) -> None:
        area = self.area
        for part in (self.talker, self.quests, self.trainer):
            part.area = area

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
            if s.world.dead:
                self.fighter.revive()
            self._set_area()
            self.quests.refresh()
            step = self.next_step()
            log.info("=== %s (level %d, %d adena, %d potions, %s, at %s)", step, s.char.level, s.adena,
                     s.item_count(LESSER_HEALING_POTION), self.area.name, s.position)
            self._status(step)
            if step[0] == "finished":
                log.info("Level %d reached: done", s.char.level)
                return
            ok = getattr(self, "do_" + step[0])(*step[1:])
            self._status(step, "ok" if ok else "failed")
            if not ok:
                key = "/".join(str(x) for x in step)
                self.state.failed[key] = self.state.failed.get(key, 0) + 1
            self.state.deaths += self.fighter.deaths
            self.fighter.deaths = 0
            self._save()

    def _guard(self) -> bool:
        """On the way: a monster attacking is fought back, death ends in the town respawn (see Navigator.guard)."""
        if self._guarding:
            return False
        s = self.session
        self._guarding = True
        try:
            if s.world.dead:
                self.fighter.revive()
                return True
            if self.fighter.attackers():
                self.fighter.defend()
                if s.world.dead:
                    self.fighter.revive()
                return True
            return False
        finally:
            self._guarding = False

    def next_step(self) -> tuple:
        s, st = self.session, self.state
        level = s.char.level
        if level >= GOAL_LEVEL:
            return ("finished",)
        on_island = self.area is TALKING_ISLAND
        if on_island and level < 6 and not st.tutorial_done and not st.gem_done and not s.item_count(BLUE_GEM) \
                and not s.item_count(SOULSHOT_BEGINNER) and not s.item_count(SPIRITSHOT_BEGINNER) \
                and not s.item_count(RECOMMENDATION) and st.failed.get("gemstone", 0) < 3:
            return ("gemstone",)
        if s.item_count(BLUE_GEM) and on_island:
            return ("errand", self._helper())
        if s.item_count(RECOMMENDATION) and on_island and st.failed.get("errand/master", 0) < 3:
            return ("errand", "master")
        if self._books() and st.failed.get(f"books/{level}", 0) < 2:
            return ("books", level)
        if self.trainer.due() and st.failed.get(f"skills/{level}", 0) < 2:
            return ("skills", level)
        if NEWBIE_BUFF_LEVELS[0] <= level <= NEWBIE_BUFF_LEVELS[1] and on_island and not self._has_newbie_buffs() \
                and st.failed.get(f"buffs/{level}", 0) < 2:
            return ("buffs", level)
        if self._needs_shopping():
            return ("shop",)
        for quest_id in self._quest_order():
            return ("quest", quest_id)
        if on_island and level >= TRAVEL_LEVEL and st.failed.get(f"travel/{level}", 0) < 2 \
                and s.adena >= (self.traveller.fee(GATEKEEPER_TI, "The Village of Gludin") or 0) + self.cfg.reserve:
            return ("travel", level)
        return ("hunt",)

    def _quest_order(self) -> list[int]:
        """The quests that start in the area and the level allows, not done and not given up, lowest level first."""
        s, st = self.session, self.state
        quests = s.knowledge.quests_starting_in(self.area, s.char.level)
        ids = [int(q) for q in quests.sort_index().sort_values("min_level", kind="stable").index]
        return [q for q in ids if q not in SKIPPED_QUESTS and q not in st.done_quests
                and st.failed.get(f"quest/{q}", 0) < 2]

    def _nearest(self, npc_ids) -> int | None:
        """The NPC of ``npc_ids`` whose spawn in the area is nearest, ``None`` when none spawns there."""
        s = self.session
        x, y, _ = s.position
        spawns = {n: p for n in npc_ids if (p := s.knowledge.nearest_spawn(n, x, y, self.area)) is not None}
        return min(spawns, key=lambda n: spawns[n].distance(x, y)) if spawns else None

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def do_errand(self, npc_id) -> bool:
        """Talk to ``npc_id`` and follow its quest link: the tutorial NPCs take the item and give the reward."""
        s, before = self.session, list(self.session.char.items)
        master = npc_id == "master"
        if master:  # the recommendation goes to the guild master of the class's hall
            npc_id = GUILD_MASTER[self._kind()]
        gem = bool(s.item_count(BLUE_GEM))
        dialog = self.talker.talk(npc_id)
        for _ in range(3):
            # A reply that is an event of a quest script ("I brought the recommendation") before the quest menu.
            link = self.talker.choose(dialog, r"^Script Q\d+\w* \S", r"_Quest Q\d+", r"_Quest$|^Script$|^Quest")
            if link is None:
                break
            dialog = self.talker.follow(link)
        s.idle(1.0)
        changed = sorted(i["item_id"] for i in s.char.items) != sorted(i["item_id"] for i in before)
        if master and changed:
            self.state.tutorial_done = True
        if gem and changed and not s.item_count(BLUE_GEM):
            self.state.gem_done = True
        return changed

    def _kind(self) -> str:
        return "mystic" if _is_mage(self.session.char.class_id) else "fighter"

    def _helper(self) -> int:
        """The Newbie Helper of the character's hall: the other one sends it away."""
        return NEWBIE_HELPER[self._kind()]

    def do_gemstone(self) -> bool:
        """The tutorial's first task: gremlins in the training hall drop the blue gemstone for the Newbie Helper."""
        s = self.session
        if not self.state.helper_met:
            # The gremlins drop it only once the Newbie Helper has given the task (the tutorial quest starts).
            self.do_errand(self._helper())
            self.state.helper_met = True
            self._save()
        helper = s.knowledge.nearest_spawn(self._helper(), *s.position[:2], self.area)
        home = s.knowledge.nearest_spawn(GREMLIN, helper.x, helper.y, self.area) if helper else None
        if home is None:
            return False
        self.nav.move_to(home, arrive=400, timeout=300)
        result = self.fighter.hunt(home, until=lambda: s.item_count(BLUE_GEM) > 0, monsters={GREMLIN}, timeout=420,
                                   max_deaths=3)
        log.info("Gemstone hunt: %s", result)
        return s.item_count(BLUE_GEM) > 0

    def _books(self) -> dict[int, int]:
        """Spellbooks the skills of the level need, sold in the area: book -> its merchant (while adena last)."""
        s, k = self.session, self.session.knowledge
        out, adena = {}, s.adena - self.cfg.reserve
        for book in self.trainer.books_wanted():
            merchant = self._merchant(book)
            price = k.item_price(book) * 2  # the shop price with the tax stays below twice the reference
            if merchant is not None and price <= adena:
                out[book] = merchant
                adena -= price
        return out

    def do_books(self, _level: int) -> bool:
        """Buy the spellbooks the trainer will ask for, from the nearest merchant that sells them."""
        books = self._books()
        by_merchant: dict[int, dict[int, int]] = {}
        for book, merchant in books.items():
            by_merchant.setdefault(merchant, {})[book] = 1
        bought = {}
        lists = self.session.knowledge.buylists
        for merchant, wanted in by_merchant.items():
            for list_id in sorted({int(l) for l in lists[lists["item_id"].isin(list(wanted))
                                                          & lists["npc_ids"].apply(lambda ids, m=merchant: m in set(ids))]["list_id"]}):
                bought.update(self.trader.buy(merchant, wanted, list_id=list_id))
        if any(bought.values()):
            self.state.failed.pop(f"skills/{self.session.char.level}", None)  # the trainer may teach them now
            return True
        return False

    def _skill_items(self) -> set[int]:
        """What the class's skills consume when learned (spellbooks): kept, not sold."""
        tree = self.session.knowledge.skill_trees
        rows = tree[tree["class_id"] == self.session.char.class_id]["items"]
        return {int(i["item_id"]) for items in rows if items is not None for i in items}

    def do_skills(self, _level: int) -> bool:
        learned = self.trainer.learn()
        return bool(learned) or not self.trainer.due()

    def do_travel(self, _level: int) -> bool:
        return self.traveller.teleport(GATEKEEPER_TI, "The Village of Gludin")

    def do_buffs(self, _level: int = 0) -> bool:
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
        s, cfg, k = self.session, self.cfg, self.session.knowledge
        self._wear_best()  # rewards and drops first, so the better pieces are not sold
        grocer = self._merchant(LESSER_HEALING_POTION)
        if grocer is None:
            log.warning("No grocer in %s", self.area.name)
            return False
        stash = {i["item_id"] for i in s.char.items if i["item_id"] in k.items.index
                 and k.items.at[i["item_id"], "etcitem_type"] in ("MATERIAL", "RECIPE")}
        if stash:  # crafting materials and recipes are kept, as a player would, not sold
            for _ in range(2):  # the nearest may be a freightman: then the next one
                warehouse = self._nearest_type("Warehouse", exclude=self.trader.no_warehouse)
                if warehouse is None or self.trader.deposit(warehouse, stash) or warehouse not in self.trader.no_warehouse:
                    break
        self.trader.sell(grocer, KEEP | self._equipped_ids() | stash | self._skill_items())
        wanted = {}
        if s.item_count(LESSER_HEALING_POTION) < cfg.potions_min:
            wanted[LESSER_HEALING_POTION] = cfg.potions_buy
        shot = self._shot_item()
        if shot and s.item_count(shot) + s.item_count(SOULSHOT_BEGINNER) + s.item_count(SPIRITSHOT_BEGINNER) < 100:
            # at most a third of the spare adena: books, gear and the teleport need the rest
            price = max(int(k.item_price(shot) * 1.2), 1)  # with the tax
            count = min(cfg.shots_buy, max(s.adena - cfg.reserve, 0) // 3 // price)
            if count >= 50:
                wanted[shot] = count
        if wanted:
            self.trader.buy(grocer, wanted)
        self._upgrade_gear()
        self._auto_shots()
        return True

    def do_quest(self, quest_id: int) -> bool:
        if quest_id in PLANS:
            result = "done" if run_plan(self.quests, quest_id) else "stuck"
        else:
            result = run_generic(self.quests, quest_id)
        log.info("Quest %d %s: %s", quest_id, self.session.knowledge.quests.at[quest_id, "name"], result)
        if result == "done":
            self.state.done_quests.append(quest_id)
        elif result == "refused":  # not for this character: never again
            self.state.failed[f"quest/{quest_id}"] = 99
        self._save()
        return result == "done"

    def do_hunt(self) -> bool:
        s = self.session
        grounds = s.knowledge.hunting_grounds(self.area, s.char.level)
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

    def _merchant(self, item_id: int) -> int | None:
        """The nearest merchant of the area that sells ``item_id``."""
        sellers = self.session.knowledge.sellers_of(item_id, self.area)
        return self._nearest(sellers) if sellers else None

    def _nearest_type(self, npc_type: str, exclude: set[int] = frozenset()) -> int | None:
        npcs = self.session.knowledge.npcs_in(self.area, (npc_type,))
        return self._nearest(sorted({int(n) for n in npcs["npc_id"]} - set(exclude))) if len(npcs) else None

    def _has_newbie_buffs(self) -> bool:
        return any(b["skill_id"] in NEWBIE_BUFF_SKILLS for b in self.session.char.buffs)

    def _needs_shopping(self) -> bool:
        s = self.session
        bag = sum(1 for i in s.char.items if i["item_id"] not in KEEP and i["item_id"] not in self._skill_items()
                  and not i.get("equipped"))
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
        merchants = set(k.npcs_in(self.area, ("Merchant",))["npc_id"].astype(int))
        lists = k.buylists[k.buylists["npc_ids"].apply(lambda ids: bool(merchants & {int(n) for n in ids}))]
        offer = lists.merge(k.items, left_on="item_id", right_index=True)
        offer = offer[offer["crystal_type"].isna() & offer["type"].isin(("Weapon", "Armor"))]
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
                best = (int(row.item_id), int(price), gain, int(row.list_id),
                        self._nearest([int(n) for n in row.npc_ids if int(n) in merchants]))
        if best is None:
            return
        bought = self.trader.buy(best[4], {best[0]: 1}, best[3])
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
