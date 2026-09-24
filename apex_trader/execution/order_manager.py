from typing import Dict, List, Any, Optional
from .base_broker import BaseBroker, Position, OrderResult
from ..risk.risk_manager import RiskDecision


class OrderManager:
    """Manages order dispatch, trade lifecycles, and portfolio synchronization."""

    def __init__(self, broker: BaseBroker):
        self.broker = broker

    def execute_risk_approved_trade(self, decision: RiskDecision) -> OrderResult:
        """Dispatches an approved trade proposal to the active broker."""
        if not decision.approved:
            return OrderResult(
                order_id="",
                symbol=decision.proposal.symbol,
                side=decision.proposal.action,
                quantity=0.0,
                price=0.0,
                status="REJECTED",
                fee=0.0,
                message=f"Order rejected by RiskManager: {decision.rejection_reason}"
            )

        p = decision.proposal
        return self.broker.submit_order(
            symbol=p.symbol,
            side=p.action,
            quantity=decision.quantity,
            price=p.target_entry_price,
            stop_loss=p.stop_loss_price,
            take_profit=p.take_profit_price
        )

    def sync_market_prices(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Syncs latest market price with active positions and checks for SL/TP executions."""
        return self.broker.update_positions_mark_price(symbol, current_price)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns snapshot of current portfolio equity, cash, and open positions."""
        equity = self.broker.get_account_equity()
        positions = self.broker.get_open_positions()
        
        pos_list = []
        for p in positions:
            pos_list.append({
                "id": p.position_id,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
            })

        return {
            "total_equity": round(equity, 2),
            "open_positions_count": len(positions),
            "positions": pos_list
        }
