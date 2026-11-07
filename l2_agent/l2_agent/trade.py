"""
Buying from and selling to NPC merchants: open the merchant's dialog, take its Buy / Sell link and answer the
BuyList / SellList the server shows. What a merchant sells and for how much comes from the buylist table; the live
list confirms it (the price there includes the tax).
"""

from __future__ import annotations

import logging
import re
import time

from l2_client import packets

from .dialog import Talker
from .session import Dialog, Session

log = logging.getLogger(__name__)

ADENA = 57


#: the server drops a buy or sale within FloodProtectorTransactionInterval (1 s) of the last one
TRANSACTION_INTERVAL_S = 1.2


class Trader:
    """Buys from and sells to NPC merchants."""

    def __init__(self, session: Session, talker: Talker):
        self.session = session
        self.talker = talker
        self._transaction_at = 0.0

    def _transaction(self, packet: bytes) -> None:
        wait = self._transaction_at + TRANSACTION_INTERVAL_S - time.time()
        if wait > 0:
            self.session.idle(wait)
        self.session.send(packet)
        self._transaction_at = time.time()

    def _open(self, npc_id: int, pattern: str, answer: str) -> dict | None:
        """Talk to the merchant and follow links matching ``pattern`` until the ``answer`` list arrives."""
        s, t = self.session, self.talker
        dialog: Dialog | None = t.talk(npc_id)
        asked = time.time()
        for _ in range(3):
            link = t.choose(dialog, pattern)
            if link is None:
                link = t.choose(dialog, r"^(buy|sell|trade|purchase)", r"merchant/")
            if link is None:
                log.warning("%s offers no %s link", s.knowledge.npc_name(npc_id), answer)
                return None
            def got(since: float = asked) -> bool:
                return (getattr(s, answer) or {}).get("received", 0) >= since

            dialog = t.follow(link, expect=lambda: got() or (s.dialog is not None and s.dialog.received >= asked))
            if got():
                return getattr(s, answer)
            asked = time.time()
        return None

    def open_list(self, npc_id: int, list_id: int) -> dict | None:
        """The live BuyList ``list_id`` of ``npc_id`` (prices with the tax), without buying."""
        return self._open(npc_id, rf"_Buy {list_id}$", "buy_list")

    def buy(self, npc_id: int, wanted: dict[int, int], list_id: int | None = None) -> dict[int, int]:
        """
        Buy ``wanted`` (item id -> count) from ``npc_id`` (its list ``list_id``, else the first) as far as the adena
        reach, in the order of ``wanted``. Returns what was bought.
        """
        s = self.session
        shop = self._open(npc_id, rf"_Buy {list_id}$" if list_id else r"_Buy\b", "buy_list")
        if shop is None:
            return {}
        prices = {i["item_id"]: i["price"] for i in shop["items"]}
        adena, order = s.adena, []
        for item_id, count in wanted.items():
            price = prices.get(item_id)
            if price is None:
                log.warning("%s does not sell %s", s.knowledge.npc_name(npc_id), s.knowledge.item_name(item_id))
                continue
            count = min(count, adena // max(price, 1))
            if count > 0:
                order.append((item_id, count))
                adena -= count * price
        if not order:
            return {}
        before = {i: s.item_count(i) for i, _ in order}
        self._transaction(packets.build_request_buy_item(shop["list_id"], order))
        s.wait_until(lambda: any(s.item_count(i) > before[i] for i, _ in order), 4.0)
        bought = {i: s.item_count(i) - before[i] for i, _ in order}
        log.info("Bought from %s: %s", s.knowledge.npc_name(npc_id),
                 {s.knowledge.item_name(i): c for i, c in bought.items()})
        return bought

    def sell(self, npc_id: int, keep: set[int]) -> int:
        """Sell everything the merchant buys except ``keep`` (item ids) and equipped items; return the adena earned."""
        s = self.session
        shop = self._open(npc_id, r"_Sell\b|\bsell\b", "sell_list")
        if shop is None:
            return 0
        equipped = {i.get("object_id") for i in s.char.items if i.get("equipped")}
        order = [(i["object_id"], i["item_id"], i["count"]) for i in shop["items"]
                 if i["item_id"] not in keep and i["item_id"] != ADENA and i["object_id"] not in equipped
                 and i["price"] > 0]
        if not order:
            return 0
        before = s.adena
        self._transaction(packets.build_request_sell_item(order))
        s.wait_until(lambda: s.adena > before, 4.0)
        log.info("Sold to %s: %s for %d adena", s.knowledge.npc_name(npc_id),
                 [s.knowledge.item_name(i) for _, i, _ in order], s.adena - before)
        return s.adena - before


def matches(text: str, *patterns: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)
