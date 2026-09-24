from dataclasses import dataclass
from typing import Tuple, Dict, Any
from ..agents.strategist_agent import TradeProposal
from ..data.order_flow import OrderFlowSnapshot


@dataclass
class GroundingVerdict:
    """Result of the anti-hallucination and price sanity audit."""
    is_grounded: bool
    reason: str
    adjusted_proposal: TradeProposal


class GroundingGate:
    """Anti-hallucination guard that audits LLM proposed prices against live order book quotes."""

    def __init__(self, max_price_deviation_pct: float = 2.0, min_risk_reward: float = 1.5):
        self.max_price_deviation_pct = max_price_deviation_pct
        self.min_risk_reward = min_risk_reward

    def audit(self, proposal: TradeProposal, flow: OrderFlowSnapshot) -> GroundingVerdict:
        """Audits the trade proposal against live market order flow quotes."""
        if proposal.action == "HOLD":
            return GroundingVerdict(
                is_grounded=True,
                reason="HOLD action requires no price execution audit.",
                adjusted_proposal=proposal
            )

        mid_price = (flow.best_bid + flow.best_ask) / 2.0
        
        # 1. Check entry price deviation against real market mid-price
        entry_deviation_pct = abs(proposal.target_entry_price - mid_price) / (mid_price + 1e-10) * 100.0
        if entry_deviation_pct > self.max_price_deviation_pct:
            return GroundingVerdict(
                is_grounded=False,
                reason=f"Hallucination rejected: Proposed entry ({proposal.target_entry_price}) deviates {round(entry_deviation_pct, 2)}% from live mid-price ({round(mid_price, 2)}).",
                adjusted_proposal=proposal
            )

        # 2. Check logical stop loss and take profit orientation
        if proposal.action == "BUY":
            if proposal.stop_loss_price >= proposal.target_entry_price:
                return GroundingVerdict(
                    is_grounded=False,
                    reason=f"Logic error: BUY stop loss ({proposal.stop_loss_price}) must be strictly BELOW entry ({proposal.target_entry_price}).",
                    adjusted_proposal=proposal
                )
            if proposal.take_profit_price <= proposal.target_entry_price:
                return GroundingVerdict(
                    is_grounded=False,
                    reason=f"Logic error: BUY take profit ({proposal.take_profit_price}) must be strictly ABOVE entry ({proposal.target_entry_price}).",
                    adjusted_proposal=proposal
                )

        elif proposal.action == "SELL":
            if proposal.stop_loss_price <= proposal.target_entry_price:
                return GroundingVerdict(
                    is_grounded=False,
                    reason=f"Logic error: SELL stop loss ({proposal.stop_loss_price}) must be strictly ABOVE entry ({proposal.target_entry_price}).",
                    adjusted_proposal=proposal
                )
            if proposal.take_profit_price >= proposal.target_entry_price:
                return GroundingVerdict(
                    is_grounded=False,
                    reason=f"Logic error: SELL take profit ({proposal.take_profit_price}) must be strictly BELOW entry ({proposal.target_entry_price}).",
                    adjusted_proposal=proposal
                )

        # 3. Check Risk-to-Reward Ratio
        risk_dist = abs(proposal.target_entry_price - proposal.stop_loss_price)
        reward_dist = abs(proposal.take_profit_price - proposal.target_entry_price)
        rr_ratio = reward_dist / (risk_dist + 1e-10)

        if rr_ratio < self.min_risk_reward:
            return GroundingVerdict(
                is_grounded=False,
                reason=f"Sub-optimal trade: Risk-to-reward ratio ({round(rr_ratio, 2)}) is below minimum requirement ({self.min_risk_reward}).",
                adjusted_proposal=proposal
            )

        # Snap entry price to best ask for BUY, best bid for SELL to guarantee instant fill
        snapped_entry = flow.best_ask if proposal.action == "BUY" else flow.best_bid
        adjusted_proposal = TradeProposal(
            symbol=proposal.symbol,
            action=proposal.action,
            target_entry_price=snapped_entry,
            stop_loss_price=proposal.stop_loss_price,
            take_profit_price=proposal.take_profit_price,
            risk_reward_ratio=rr_ratio,
            confidence_score=proposal.confidence_score,
            reasoning=proposal.reasoning
        )

        return GroundingVerdict(
            is_grounded=True,
            reason="Grounded & validated against live market order book.",
            adjusted_proposal=adjusted_proposal
        )
