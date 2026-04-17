from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

PRODUCT = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 50
QUOTE_OFFSET = 4
TAKE_THRESHOLD = 1
DAY_UNKNOWN = -999

class Trader:
    def _get_fair_value(self, day: int, ts: int) -> float:
        return 10_000 + 1_000 * (day + 2) + 0.001 * ts

    def _extract_day(self, state: TradingState) -> int:
        od = state.order_depths.get(PRODUCT)
        if not od:
            return 0
        bids = od.buy_orders
        asks = od.sell_orders
        best_bid = max(bids.keys()) if bids else None
        best_ask = min(asks.keys()) if asks else None
        if best_bid is not None and best_ask is not None:
            mid_px = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid_px = best_bid
        elif best_ask is not None:
            mid_px = best_ask
        else:
            return 0
        return int(round((mid_px - 10_000) / 1_000) - 2)

    def _unpack(self, td: str) -> dict:
        if td:
            try:
                return json.loads(td)
            except Exception:
                return {}
        return {}

    def _pack(self, d: dict) -> str:
        return json.dumps(d)

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        stored = self._unpack(state.traderData)
        day = stored.get("day", DAY_UNKNOWN)
        if day == DAY_UNKNOWN:
            day = self._extract_day(state)
            stored["day"] = day
        fv = self._get_fair_value(day, state.timestamp)
        pos = state.position.get(PRODUCT, 0)
        buy_left = POSITION_LIMIT - pos
        sell_left = POSITION_LIMIT + pos
        orders = []
        od = state.order_depths.get(PRODUCT, OrderDepth())
        asks = sorted(od.sell_orders.items())
        bids = sorted(od.buy_orders.items(), reverse=True)
        for p, v in asks:
            if p > fv - TAKE_THRESHOLD or buy_left <= 0:
                break
            qty = min(-v, buy_left)
            orders.append(Order(PRODUCT, p, qty))
            buy_left -= qty
        for p, v in bids:
            if p < fv + TAKE_THRESHOLD or sell_left <= 0:
                break
            qty = min(v, sell_left)
            orders.append(Order(PRODUCT, p, -qty))
            sell_left -= qty
        mm_bid = math.floor(fv - QUOTE_OFFSET)
        mm_ask = math.ceil(fv + QUOTE_OFFSET)
        bid_size = min(10, buy_left)
        ask_size = min(10, sell_left)
        if bid_size > 0:
            orders.append(Order(PRODUCT, mm_bid, bid_size))
        if ask_size > 0:
            orders.append(Order(PRODUCT, mm_ask, -ask_size))
        resp = {PRODUCT: orders}
        conversions = 0
        new_data = self._pack(stored)
        return resp, conversions, new_data