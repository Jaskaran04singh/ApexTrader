import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from .base_broker import BaseBroker, Position, OrderResult


class PaperBroker(BaseBroker):
    """Realistic paper trading broker simulator with slippage, fees, and SL/TP auto-triggers."""

    def __init__(
        self,
        initial_cash: float = 10000.0,
        slippage_pct: float = 0.05,
        fee_pct: float = 0.075
    ):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct

        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []

    def get_account_equity(self) -> float:
        """Total equity = Cash + Unrealized PnL of all open positions + Initial margin held."""
        positions_value = sum(
            (p.quantity * p.current_price) if p.side == "LONG" else (p.quantity * (2 * p.entry_price - p.current_price))
            for p in self.positions.values()
        )
        return self.cash + sum(p.unrealized_pnl for p in self.positions.values())

    def get_open_positions(self) -> List[Position]:
        return list(self.positions.values())

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        atr: float = 0.0
    ) -> OrderResult:
        """Executes a simulated market order with slippage, transaction fee, and trailing stop tracking."""
        if quantity <= 0 or price <= 0:
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                quantity=0.0,
                price=0.0,
                status="REJECTED",
                fee=0.0,
                message="Invalid order size or price."
            )

        # Apply slippage (Buyers pay slightly higher, Sellers receive slightly lower)
        slippage_factor = (1 + (self.slippage_pct / 100.0)) if side == "BUY" else (1 - (self.slippage_pct / 100.0))
        exec_price = price * slippage_factor

        notional = quantity * exec_price
        fee = notional * (self.fee_pct / 100.0)

        # Check cash balance for longs
        if side == "BUY" and (notional + fee) > self.cash:
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=exec_price,
                status="REJECTED",
                fee=0.0,
                message=f"Insufficient cash: Required ${round(notional + fee, 2)}, Available ${round(self.cash, 2)}"
            )

        self.cash -= fee
        order_id = str(uuid.uuid4())[:8]
        pos_id = f"pos_{symbol.replace('/', '_')}_{order_id}"

        position_side = "LONG" if side == "BUY" else "SHORT"
        
        pos = Position(
            position_id=pos_id,
            symbol=symbol,
            side=position_side,
            quantity=quantity,
            entry_price=exec_price,
            current_price=exec_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            opened_at=datetime.utcnow(),
            highest_price=exec_price,
            lowest_price=exec_price,
            atr=atr if atr > 0 else (exec_price * 0.015),
            trailing_stop_active=False
        )
        self.positions[pos_id] = pos

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=round(exec_price, 4),
            status="FILLED",
            fee=round(fee, 4),
            message=f"Paper order FILLED at ${round(exec_price, 4)} with ${round(fee, 2)} fee."
        )

    def close_position(self, position_id: str, current_price: float, reason: str = "MANUAL") -> OrderResult:
        """Closes an open position, realizes PnL, deducts closing fee."""
        pos = self.positions.get(position_id)
        if not pos:
            return OrderResult(
                order_id="",
                symbol="",
                side="",
                quantity=0.0,
                price=0.0,
                status="REJECTED",
                fee=0.0,
                message=f"Position {position_id} not found."
            )

        # Slippage on exit
        close_side = "SELL" if pos.side == "LONG" else "BUY"
        slippage_factor = (1 - (self.slippage_pct / 100.0)) if close_side == "SELL" else (1 + (self.slippage_pct / 100.0))
        exec_price = current_price * slippage_factor

        notional = pos.quantity * exec_price
        fee = notional * (self.fee_pct / 100.0)

        # Calculate Realized PnL
        if pos.side == "LONG":
            realized_pnl = (exec_price - pos.entry_price) * pos.quantity - fee
        else:
            realized_pnl = (pos.entry_price - exec_price) * pos.quantity - fee

        realized_pnl_pct = (realized_pnl / (pos.quantity * pos.entry_price)) * 100.0

        self.cash += (pos.quantity * exec_price) + realized_pnl if pos.side == "SHORT" else realized_pnl
        del self.positions[position_id]

        closed_record = {
            "position_id": position_id,
            "symbol": pos.symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": exec_price,
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(realized_pnl_pct, 2),
            "fee": round(fee, 4),
            "reason": reason,
            "opened_at": pos.opened_at.isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
        }
        self.closed_trades.append(closed_record)

        return OrderResult(
            order_id=str(uuid.uuid4())[:8],
            symbol=pos.symbol,
            side=close_side,
            quantity=pos.quantity,
            price=round(exec_price, 4),
            status="FILLED",
            fee=round(fee, 4),
            message=f"Position closed ({reason}): Realized PnL ${round(realized_pnl, 2)} ({round(realized_pnl_pct, 2)}%)"
        )

    def update_positions_mark_price(
        self,
        symbol: str,
        current_price: float,
        trailing_atr_mult: float = 1.5,
        breakeven_atr_mult: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Updates live mark prices, trails stop loss dynamically into profit, and executes SL/TP triggers."""
        closed_events = []
        to_close = []

        for pos_id, pos in self.positions.items():
            if pos.symbol != symbol:
                continue

            pos.current_price = current_price
            atr = pos.atr if pos.atr > 0 else (pos.entry_price * 0.015)

            if pos.side == "LONG":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
                pos.unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100.0

                # 1. Update peak price
                if current_price > pos.highest_price:
                    pos.highest_price = current_price

                # 2. Breakeven Lock: Once price gains >= 1.0 ATR, move stop to entry price (Risk-Free!)
                gain_distance = pos.highest_price - pos.entry_price
                if gain_distance >= (breakeven_atr_mult * atr):
                    breakeven_stop = pos.entry_price + (0.1 * atr)  # Cover fees
                    if pos.stop_loss < breakeven_stop:
                        pos.stop_loss = breakeven_stop
                        pos.trailing_stop_active = True

                # 3. Dynamic Trailing Stop: Trail behind the peak price
                if pos.trailing_stop_active:
                    dynamic_trail = pos.highest_price - (trailing_atr_mult * atr)
                    if dynamic_trail > pos.stop_loss:
                        pos.stop_loss = dynamic_trail

                # Check Exits
                if current_price <= pos.stop_loss:
                    reason = "TRAILING_STOP_TRIGGERED" if pos.trailing_stop_active else "STOP_LOSS_TRIGGERED"
                    to_close.append((pos_id, current_price, reason))
                elif current_price >= pos.take_profit:
                    to_close.append((pos_id, current_price, "TAKE_PROFIT_TRIGGERED"))

            elif pos.side == "SHORT":
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                pos.unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100.0

                # 1. Update lowest price seen
                if current_price < pos.lowest_price or pos.lowest_price == 0.0:
                    pos.lowest_price = current_price

                # 2. Breakeven Lock
                drop_distance = pos.entry_price - pos.lowest_price
                if drop_distance >= (breakeven_atr_mult * atr):
                    breakeven_stop = pos.entry_price - (0.1 * atr)
                    if pos.stop_loss > breakeven_stop:
                        pos.stop_loss = breakeven_stop
                        pos.trailing_stop_active = True

                # 3. Dynamic Trailing Stop
                if pos.trailing_stop_active:
                    dynamic_trail = pos.lowest_price + (trailing_atr_mult * atr)
                    if dynamic_trail < pos.stop_loss:
                        pos.stop_loss = dynamic_trail

                # Check Exits
                if current_price >= pos.stop_loss:
                    reason = "TRAILING_STOP_TRIGGERED" if pos.trailing_stop_active else "STOP_LOSS_TRIGGERED"
                    to_close.append((pos_id, current_price, reason))
                elif current_price <= pos.take_profit:
                    to_close.append((pos_id, current_price, "TAKE_PROFIT_TRIGGERED"))

        for pos_id, price, reason in to_close:
            res = self.close_position(pos_id, price, reason=reason)
            closed_events.append({
                "position_id": pos_id,
                "reason": reason,
                "result": res
            })

        return closed_events
