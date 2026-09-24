from typing import Dict, Any


class PositionSizer:
    """Calculates volatility-adjusted position size based on portfolio risk limits."""

    def __init__(self, max_portfolio_risk_pct: float = 1.5, max_leverage: float = 1.0):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_leverage = max_leverage

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        confidence_score: float = 0.8
    ) -> Dict[str, Any]:
        """Calculates exact order quantity and notional capital allocation."""
        if account_equity <= 0 or entry_price <= 0:
            return {"quantity": 0.0, "notional_value": 0.0, "risk_amount": 0.0}

        risk_distance = abs(entry_price - stop_loss_price)
        if risk_distance <= 0:
            risk_distance = entry_price * 0.02  # 2% default buffer if zero

        # Base dollar risk allocation (e.g. 1.5% of $10,000 = $150)
        base_risk_dollars = account_equity * (self.max_portfolio_risk_pct / 100.0)
        
        # Scale risk slightly by agent confidence (0.5 to 1.0 multiplier)
        confidence_scale = max(0.5, min(1.0, confidence_score))
        risk_dollars = base_risk_dollars * confidence_scale

        # Quantity = Risk Dollars / Distance to Stop Loss
        quantity = risk_dollars / risk_distance
        notional_value = quantity * entry_price

        # Cap notional value to available equity * max_leverage with a 2% fee/slippage safety buffer
        max_notional = account_equity * self.max_leverage * 0.98
        if notional_value > max_notional:
            notional_value = max_notional
            quantity = notional_value / entry_price

        return {
            "quantity": round(quantity, 6),
            "notional_value": round(notional_value, 2),
            "risk_amount": round(risk_dollars, 2),
            "risk_pct": round((risk_dollars / account_equity) * 100.0, 2),
        }
