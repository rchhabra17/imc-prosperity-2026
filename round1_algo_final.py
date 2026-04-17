from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


# ============================================================
#  CONFIGURABLE PARAMETERS — tweak these between runs
# ============================================================

# --- ASH_COATED_OSMIUM (stable, mean-reverting ~10000) -------
ACO              = "ASH_COATED_OSMIUM"
ACO_LIMIT        = 80

ACO_EWMA_SPAN    = 21      # higher = smoother fair estimate (filters bid-ask bounce)
ACO_TAKE_EDGE    = 0       # min ticks of edge to aggress a resting order (0 = take at fair)
ACO_QUOTE_OFFSET = 1       # primary quote distance from fair (each side)
ACO_QUOTE_SIZE   = 35      # primary quote size per side
ACO_L2_OFFSET    = 3       # 2nd-layer quote distance (0 = disabled)
ACO_L2_SIZE      = 20      # 2nd-layer quote size per side
ACO_SKEW         = 0.10    # fair shifts by  -SKEW * position  (flattens inventory)

# --- INTARIAN_PEPPER_ROOT (linear ramp: +1 XIREC / 1000 ts) -
IPR              = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT        = 80

IPR_SLOPE        = 0.001   # price increase per timestamp unit (1 per 1000 ts)
IPR_CORR_ALPHA   = 0.20    # EWMA smoothing for residual fallback correction
IPR_TAKE_EDGE    = 0       # min ticks of edge to aggress
IPR_QUOTE_OFFSET = 1       # primary quote distance from fair
IPR_QUOTE_SIZE   = 28      # primary quote size per side
IPR_L2_OFFSET    = 4       # 2nd-layer distance (0 = disabled)
IPR_L2_SIZE      = 16      # 2nd-layer size per side
IPR_SKEW         = 0.25    # inventory-flattening coefficient
IPR_INV_TARGET   = 75      # target resting position (positive = lean long into the trend)


# ============================================================
#  TRADER
# ============================================================

class Trader:

    # ---- state persistence (traderData is our only memory) ---

    def _load(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def _save(self, d: dict) -> str:
        return json.dumps(d)

    # ---- order-book helpers ---------------------------------

    def _mid(self, od: OrderDepth):
        """Simple mid-price, or None if either side is empty."""
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _detect_day_offset(self, mid: float) -> float:
        """Estimate IPR's 1000s-multiple offset from the first observed mid.
        day -2 → 0, day -1 → 1000, day 0 → 2000, day 1 → 3000, …"""
        return round((mid - 10000) / 1000) * 1000

    # ---- generic order builder ------------------------------

    def _build_orders(
        self, symbol: str, od: OrderDepth, pos: int,
        fair: float, limit: int,
        take_edge: float, q_off: float, q_sz: int,
        l2_off: float, l2_sz: int,
    ) -> List[Order]:

        orders: List[Order] = []
        buy_cap  = limit - pos          # room to buy  before hitting +limit
        sell_cap = limit + pos          # room to sell before hitting -limit

        # --- TAKE: sweep mispriced resting orders ---------------

        # buy against asks priced below our fair (ascending price)
        for px in sorted(od.sell_orders.keys()):
            if px > fair - take_edge or buy_cap <= 0:
                break
            vol = -od.sell_orders[px]           # sell qty is negative
            qty = min(vol, buy_cap)
            if qty > 0:
                orders.append(Order(symbol, px, qty))
                buy_cap -= qty

        # sell against bids priced above our fair (descending price)
        for px in sorted(od.buy_orders.keys(), reverse=True):
            if px < fair + take_edge or sell_cap <= 0:
                break
            vol = od.buy_orders[px]
            qty = min(vol, sell_cap)
            if qty > 0:
                orders.append(Order(symbol, px, -qty))
                sell_cap -= qty

        # --- QUOTE: passive layer 1 ----------------------------

        bid_px = math.floor(fair - q_off)
        ask_px = math.ceil(fair + q_off)

        bid_sz = min(q_sz, buy_cap)
        ask_sz = min(q_sz, sell_cap)

        if bid_sz > 0:
            orders.append(Order(symbol, bid_px, bid_sz))
            buy_cap -= bid_sz
        if ask_sz > 0:
            orders.append(Order(symbol, ask_px, -ask_sz))
            sell_cap -= ask_sz

        # --- QUOTE: passive layer 2 (deeper, optional) ---------

        if l2_off > 0 and l2_sz > 0:
            bid2_px = math.floor(fair - l2_off)
            ask2_px = math.ceil(fair + l2_off)
            bid2_sz = min(l2_sz, buy_cap)
            ask2_sz = min(l2_sz, sell_cap)
            if bid2_sz > 0:
                orders.append(Order(symbol, bid2_px, bid2_sz))
            if ask2_sz > 0:
                orders.append(Order(symbol, ask2_px, -ask2_sz))

        return orders

    # ---- ACO strategy ---------------------------------------

    def _trade_aco(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(ACO, OrderDepth())
        pos = state.position.get(ACO, 0)
        mid = self._mid(od)

        # update EWMA fair value (skip update if book is empty)
        alpha = 2.0 / (ACO_EWMA_SPAN + 1)
        prev  = data.get("aco_ewma")

        if mid is not None:
            ewma = mid if prev is None else alpha * mid + (1 - alpha) * prev
            data["aco_ewma"] = ewma
        else:
            ewma = prev
            if ewma is None:
                return []               # no fair estimate yet

        # position-skew: shift fair to lean toward flat
        fair = ewma - ACO_SKEW * pos

        return self._build_orders(
            ACO, od, pos, fair, ACO_LIMIT,
            ACO_TAKE_EDGE, ACO_QUOTE_OFFSET, ACO_QUOTE_SIZE,
            ACO_L2_OFFSET, ACO_L2_SIZE,
        )

    # ---- IPR strategy ---------------------------------------

    def _trade_ipr(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(IPR, OrderDepth())
        pos = state.position.get(IPR, 0)
        ts  = state.timestamp
        mid = self._mid(od)

        # --- determine day offset on first valid tick ----------
        if "ipr_doff" not in data:
            if mid is not None:
                data["ipr_doff"] = self._detect_day_offset(mid)
            else:
                data["ipr_doff"] = 3000         # default: assume day 1

        # analytical fair value
        base_fair = 10000 + data["ipr_doff"] + IPR_SLOPE * ts

        # --- residual correction (self-heals if formula drifts) -
        corr = data.get("ipr_corr", 0.0)
        if mid is not None:
            residual = mid - base_fair
            corr = IPR_CORR_ALPHA * residual + (1 - IPR_CORR_ALPHA) * corr
            data["ipr_corr"] = corr

        fair_raw = base_fair + corr

        # inventory-target skew
        adj_pos = pos - IPR_INV_TARGET
        fair    = fair_raw - IPR_SKEW * adj_pos

        return self._build_orders(
            IPR, od, pos, fair, IPR_LIMIT,
            IPR_TAKE_EDGE, IPR_QUOTE_OFFSET, IPR_QUOTE_SIZE,
            IPR_L2_OFFSET, IPR_L2_SIZE,
        )

    # ---- main entry point -----------------------------------

    def run(self, state: TradingState):
        data   = self._load(state.traderData)
        result = {}

        result[ACO] = self._trade_aco(state, data)
        result[IPR] = self._trade_ipr(state, data)

        conversions = 0
        return result, conversions, self._save(data)