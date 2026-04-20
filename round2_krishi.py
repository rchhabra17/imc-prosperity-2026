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

ACO_EWMA_SPAN    = 15      # higher = smoother fair estimate
ACO_TAKE_EDGE    = 0       # min ticks of edge to aggress
ACO_QUOTE_OFFSET = 1       # primary quote distance from fair
ACO_QUOTE_SIZE   = 20      # primary quote size per side
ACO_L2_OFFSET    = 3       # 2nd-layer quote distance
ACO_L2_SIZE      = 10      # 2nd-layer quote size per side
ACO_WALL_SIZE    = 0       # 3rd-layer: quote 1 inside detected wall
ACO_SKEW         = 0.20    # fair shifts by  -SKEW * position

# --- INTARIAN_PEPPER_ROOT (linear ramp) ----------------------
IPR              = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT        = 80

IPR_SLOPE        = 0.001   # expected price increase per ts unit
IPR_CORR_ALPHA   = 0.20    # EWMA smoothing for residual correction
IPR_TAKE_EDGE    = 0       # min ticks of edge to aggress
IPR_QUOTE_OFFSET = 1       # primary quote distance from fair
IPR_QUOTE_SIZE   = 28      # primary quote size per side
IPR_L2_OFFSET    = 4       # 2nd-layer distance
IPR_L2_SIZE      = 16      # 2nd-layer size per side
IPR_SKEW         = 0.25    # inventory-flattening coefficient

# --- ADAPTIVE TARGET PARAMS ----------------------------------
IPR_MAX_TARGET   = 75      # max long inventory when trend is strong
IPR_SLOPE_WINDOW = 30      # ticks of mid history to estimate slope
IPR_SLOPE_THRESH = 0.0003  # below this slope, target trends toward 0


# ============================================================
#  TRADER
# ============================================================

class Trader:
    
    # ---- MAF Auction Bid -------------------------------------
    
    def bid(self) -> int:
        """
        Bids for 25% extra market access volume in Round 2.
        2,500 is a Level-2 strategic bid designed to beat the median
        without severely impacting net PnL.
        """
        return 1500

    # ---- state persistence -----------------------------------

    def _load(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def _save(self, d: dict) -> str:
        return json.dumps(d)

    # ---- counterparty tracking -------------------------------
    
    def _track_counterparties(self, state: TradingState, data: dict):
        """
        Parses the trade tape to see who is taking our liquidity and
        maintains a persistent tally in the traderData state.
        """
        if "counterparty_volume" not in data:
            data["counterparty_volume"] = {}

        for product, trades in state.own_trades.items():
            for trade in trades:
                # Identify the counterparty. Usually, your bot is labeled "SUBMISSION"
                if trade.buyer == "SUBMISSION":
                    cp = trade.seller
                else:
                    cp = trade.buyer
                
                # Keep a running tally of absolute volume traded against each ID
                if cp not in ("", "SUBMISSION"):
                    current_vol = data["counterparty_volume"].get(cp, 0)
                    data["counterparty_volume"][cp] = current_vol + abs(trade.quantity)
        
        # Print to standard out so it appears in the sandbox logs
        print(f"[{state.timestamp}] Counterparty Volumes: {data['counterparty_volume']}")

    # ---- order-book helpers ----------------------------------

    def _mid(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _detect_day_offset(self, mid: float) -> float:
        return round((mid - 10000) / 1000) * 1000

    # ---- generic order builder -------------------------------

    def _build_orders(
        self, symbol: str, od: OrderDepth, pos: int,
        fair: float, limit: int,
        take_edge: float, q_off: float, q_sz: int,
        l2_off: float, l2_sz: int,
    ):

        orders: List[Order] = []
        buy_cap  = limit - pos
        sell_cap = limit + pos

        # --- TAKE: sweep mispriced resting orders ---------------
        for px in sorted(od.sell_orders.keys()):
            if px > fair - take_edge or buy_cap <= 0:
                break
            vol = -od.sell_orders[px]
            qty = min(vol, buy_cap)
            if qty > 0:
                orders.append(Order(symbol, px, qty))
                buy_cap -= qty

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

        # --- QUOTE: passive layer 2 ----------------------------
        if l2_off > 0 and l2_sz > 0:
            bid2_px = math.floor(fair - l2_off)
            ask2_px = math.ceil(fair + l2_off)
            bid2_sz = min(l2_sz, buy_cap)
            ask2_sz = min(l2_sz, sell_cap)
            if bid2_sz > 0:
                orders.append(Order(symbol, bid2_px, bid2_sz))
                buy_cap -= bid2_sz
            if ask2_sz > 0:
                orders.append(Order(symbol, ask2_px, -ask2_sz))
                sell_cap -= ask2_sz

        return orders, buy_cap, sell_cap

    # ---- ACO strategy ----------------------------------------

    def _trade_aco(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(ACO, OrderDepth())
        pos = state.position.get(ACO, 0)
        mid = self._mid(od)

        alpha = 2.0 / (ACO_EWMA_SPAN + 1)
        prev  = data.get("aco_ewma")

        if mid is not None:
            ewma = mid if prev is None else alpha * mid + (1 - alpha) * prev
            data["aco_ewma"] = ewma
        else:
            ewma = prev
            if ewma is None:
                return []

        skew = ACO_SKEW
        ts_remaining = 100000 - state.timestamp
        if ts_remaining < 2000:
            skew = ACO_SKEW * (2000.0 / max(ts_remaining, 1))

        fair = ewma - skew * pos

        orders, buy_cap, sell_cap = self._build_orders(
            ACO, od, pos, fair, ACO_LIMIT,
            ACO_TAKE_EDGE, ACO_QUOTE_OFFSET, ACO_QUOTE_SIZE,
            ACO_L2_OFFSET, ACO_L2_SIZE,
        )

        # --- LAYER 3: wall-aware quotes -----------------------
        if ACO_WALL_SIZE > 0 and od.buy_orders and od.sell_orders:
            # Wall = largest-volume level on each side
            wall_bid = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
            wall_ask = min(od.sell_orders.keys(), key=lambda p: abs(od.sell_orders[p]))

            wall_bid_px = wall_bid + 1
            wall_ask_px = wall_ask - 1

            # Only place if outside our L2 quotes (no collision)
            if wall_bid_px < math.floor(fair - ACO_L2_OFFSET):
                sz = min(ACO_WALL_SIZE, buy_cap)
                if sz > 0:
                    orders.append(Order(ACO, wall_bid_px, sz))
                    buy_cap -= sz

            if wall_ask_px > math.ceil(fair + ACO_L2_OFFSET):
                sz = min(ACO_WALL_SIZE, sell_cap)
                if sz > 0:
                    orders.append(Order(ACO, wall_ask_px, -sz))
                    sell_cap -= sz

        return orders

    # ---- IPR slope estimation --------------------------------

    def _estimate_slope(self, data: dict) -> float:
        history = data.get("ipr_mids", [])
        if len(history) < 10:
            return IPR_SLOPE

        recent = history[-IPR_SLOPE_WINDOW:]
        n = len(recent)
        if n < 10:
            return IPR_SLOPE

        # Linear regression: slope of mid vs tick index
        # Each tick step = 100 ts
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return IPR_SLOPE

        slope_per_tick = num / den
        slope_per_ts = slope_per_tick / 100.0
        return slope_per_ts

    def _adaptive_target(self, data: dict, ts: int = 0) -> float:
        corr = data.get("ipr_corr", 0.0)
        ticks_seen = len(data.get("ipr_mids", []))

        if corr > -10:
            raw_target = float(IPR_MAX_TARGET)
        elif corr < -20:
            raw_target = 0.0
        else:
            raw_target = IPR_MAX_TARGET * (20 + corr) / 10.0

        ramp = min(ticks_seen / 80.0, 1.0)
        raw_target *= ramp

        ts_remaining = 100000 - ts
        if ts_remaining < 2000:
            raw_target *= ts_remaining / 2000.0

        return raw_target

    # ---- IPR strategy ----------------------------------------

    def _trade_ipr(self, state: TradingState, data: dict) -> List[Order]:
        od  = state.order_depths.get(IPR, OrderDepth())
        pos = state.position.get(IPR, 0)
        ts  = state.timestamp
        mid = self._mid(od)

        # --- day offset detection ------------------------------
        if "ipr_doff" not in data:
            if mid is not None:
                data["ipr_doff"] = self._detect_day_offset(mid)
            else:
                data["ipr_doff"] = 3000

        # --- track mid history for slope estimation ------------
        if "ipr_mids" not in data:
            data["ipr_mids"] = []
        if mid is not None:
            data["ipr_mids"].append(mid)
            max_hist = IPR_SLOPE_WINDOW * 2
            if len(data["ipr_mids"]) > max_hist:
                data["ipr_mids"] = data["ipr_mids"][-max_hist:]

        # --- estimate slope and adapt target -------------------
        inv_target = self._adaptive_target(data, ts)
        data["ipr_target"] = inv_target

        # --- analytical fair value + correction ----------------
        base_fair = 10000 + data["ipr_doff"] + IPR_SLOPE * ts

        corr = data.get("ipr_corr", 0.0)
        if mid is not None:
            residual = mid - base_fair
            corr = IPR_CORR_ALPHA * residual + (1 - IPR_CORR_ALPHA) * corr
            data["ipr_corr"] = corr

        fair_raw = base_fair + corr

        # --- inventory-target skew -----------------------------
        adj_pos = pos - inv_target
        fair    = fair_raw - IPR_SKEW * adj_pos

        orders, _, _ = self._build_orders(
            IPR, od, pos, fair, IPR_LIMIT,
            IPR_TAKE_EDGE, IPR_QUOTE_OFFSET, IPR_QUOTE_SIZE,
            IPR_L2_OFFSET, IPR_L2_SIZE,
        )

        return orders

    # ---- main entry point ------------------------------------

    def run(self, state: TradingState):
        data   = self._load(state.traderData)
        result = {}

        # Update counterparty logs
        self._track_counterparties(state, data)

        result[ACO] = self._trade_aco(state, data)
        result[IPR] = self._trade_ipr(state, data)

        conversions = 0
        return result, conversions, self._save(data)