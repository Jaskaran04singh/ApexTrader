from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from ..agents.strategist_agent import TradeProposal
from ..data.order_flow import OrderFlowSnapshot
from .grounding_gate import GroundingGate, GroundingVerdict
from .position_sizer import PositionSizer


@dataclass
class RiskDecision:
    """Final risk approval verdict."""
    approved: bool
    rejection_reason: Optional[str]
    proposal: TradeProposal
    quantity: float
    notional_value: float
    risk_dollars: float


class RiskManager:
    """Master quantitative risk control engine and portfolio circuit breaker."""

    def __init__(
        self,
        max_portfolio_risk_pct: float = 1.5,
        max_open_positions: int = 3,
        daily_max_drawdown_pct: float = 3.0,
        confidence_threshold: float = 0.65,
        min_risk_reward: float = 1.5,
    ):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_open_positions = max_open_positions
        self.daily_max_drawdown_pct = daily_max_drawdown_pct
        self.confidence_threshold = confidence_threshold

        self.grounding_gate = GroundingGate(max_price_deviation_pct=2.0, min_risk_reward=min_risk_reward)
        self.position_sizer = PositionSizer(max_portfolio_risk_pct=max_portfolio_risk_pct)

        # State tracking
        self.starting_daily_equity: Optional[float] = None
        self.current_equity: float = 10000.0  # Default initial balance
        self.is_circuit_breaker_tripped: bool = False

    def evaluate_trade(
        self,
        proposal: TradeProposal,
        flow: OrderFlowSnapshot,
        open_positions_count: int,
        current_equity: float,
        technical_bias: Optional[str] = None,
        news_sentiment: Optional[float] = None,
        debate_winner: Optional[str] = None
    ) -> RiskDecision:
        """Evaluates trade proposal against all hard risk limits and 3-way multi-factor confluence."""
        self.current_equity = current_equity
        if self.starting_daily_equity is None:
            self.starting_daily_equity = current_equity

        # 1. Check Circuit Breaker (Daily Drawdown)
        daily_drawdown_pct = ((self.starting_daily_equity - current_equity) / (self.starting_daily_equity + 1e-10)) * 100.0
        if daily_drawdown_pct >= self.daily_max_drawdown_pct:
            self.is_circuit_breaker_tripped = True
            return RiskDecision(
                approved=False,
                rejection_reason=f"CIRCUIT BREAKER TRIPPED: Daily drawdown ({round(daily_drawdown_pct, 2)}%) exceeds limit ({self.daily_max_drawdown_pct}%). Trading halted for today.",
                proposal=proposal,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        if proposal.action == "HOLD":
            return RiskDecision(
                approved=False,
                rejection_reason="No trade action requested (HOLD).",
                proposal=proposal,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        # 2. Check Agent Confidence Threshold
        if proposal.confidence_score < self.confidence_threshold:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Low confidence ({round(proposal.confidence_score, 2)} < {self.confidence_threshold}).",
                proposal=proposal,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        # 3. Check 3-Way Multi-Factor Confluence Gate
        if proposal.action == "BUY":
            # Flow must not be strong sell pressure
            if flow.flow_regime == "STRONG_SELL_PRESSURE":
                return RiskDecision(
                    approved=False,
                    rejection_reason="Confluence Gate Failed: Order book shows heavy sell wall dominance.",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )
            # News must not be heavily bearish
            if news_sentiment is not None and news_sentiment < -0.35:
                return RiskDecision(
                    approved=False,
                    rejection_reason=f"Confluence Gate Failed: News sentiment is strongly negative ({news_sentiment}).",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )
            # Debate should not be won by Bear
            if debate_winner == "BEAR":
                return RiskDecision(
                    approved=False,
                    rejection_reason="Confluence Gate Failed: Bearish researcher won the dialectic debate.",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )

        elif proposal.action == "SELL":
            if flow.flow_regime == "STRONG_BUY_PRESSURE":
                return RiskDecision(
                    approved=False,
                    rejection_reason="Confluence Gate Failed: Order book shows heavy buy wall dominance.",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )
            if news_sentiment is not None and news_sentiment > 0.35:
                return RiskDecision(
                    approved=False,
                    rejection_reason=f"Confluence Gate Failed: News sentiment is strongly positive ({news_sentiment}).",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )
            if debate_winner == "BULL":
                return RiskDecision(
                    approved=False,
                    rejection_reason="Confluence Gate Failed: Bullish researcher won the dialectic debate.",
                    proposal=proposal,
                    quantity=0.0,
                    notional_value=0.0,
                    risk_dollars=0.0
                )

        # 4. Check Max Open Positions
        if open_positions_count >= self.max_open_positions:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Max open positions reached ({open_positions_count}/{self.max_open_positions}).",
                proposal=proposal,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        # 5. Anti-Hallucination Grounding Audit
        verdict = self.grounding_gate.audit(proposal, flow)
        if not verdict.is_grounded:
            return RiskDecision(
                approved=False,
                rejection_reason=verdict.reason,
                proposal=proposal,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        # 6. Position Sizing
        adjusted = verdict.adjusted_proposal
        sizing = self.position_sizer.calculate_position_size(
            account_equity=current_equity,
            entry_price=adjusted.target_entry_price,
            stop_loss_price=adjusted.stop_loss_price,
            confidence_score=adjusted.confidence_score
        )

        if sizing["quantity"] <= 0:
            return RiskDecision(
                approved=False,
                rejection_reason="Calculated order quantity is 0 (insufficient margin or zero risk tolerance).",
                proposal=adjusted,
                quantity=0.0,
                notional_value=0.0,
                risk_dollars=0.0
            )

        return RiskDecision(
            approved=True,
            rejection_reason=None,
            proposal=adjusted,
            quantity=sizing["quantity"],
            notional_value=sizing["notional_value"],
            risk_dollars=sizing["risk_amount"]
        )
