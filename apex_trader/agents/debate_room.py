from dataclasses import dataclass
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentOutput


@dataclass
class DebateOutcome:
    """Consolidated outcome of the Bull vs. Bear debate."""
    symbol: str
    bull_case: str
    bear_case: str
    winning_side: str  # "BULL", "BEAR", "NEUTRAL"
    key_vulnerability: str
    conviction_delta: float  # -1.0 (strongly bear) to +1.0 (strongly bull)
    debate_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "winning_side": self.winning_side,
            "key_vulnerability": self.key_vulnerability,
            "conviction_delta": round(self.conviction_delta, 2),
            "debate_summary": self.debate_summary,
        }


class BullBearDebateRoom:
    """Orchestrates an adversarial dialectic debate between Bull and Bear researchers."""

    BULL_PROMPT = """You are a Senior Bullish Investment Researcher.
Your job is to identify every upside catalyst, strong momentum indicator, order book buying support, and reason why the asset should trade higher.
Be aggressive, rigorous, and point out why sellers are trapped."""

    BEAR_PROMPT = """You are a Senior Bearish Risk Analyst & Short-Seller.
Your job is to stress-test the bullish thesis, spot liquidity traps, exhaustion volume, overhead resistance, and reasons why this trade could fail.
Be skeptical, risk-conscious, and point out downside traps."""

    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        self.bull_agent = BaseAgent("BullResearcher", self.BULL_PROMPT, provider, model_name)
        self.bear_agent = BaseAgent("BearResearcher", self.BEAR_PROMPT, provider, model_name)

    def conduct_debate(
        self,
        symbol: str,
        news_out: AgentOutput,
        flow_out: AgentOutput,
        tech_out: AgentOutput
    ) -> DebateOutcome:
        """Executes a structured debate round between Bull and Bear researchers."""
        context = f"""Asset: {symbol}
News Analysis: {news_out.summary} (Sentiment: {news_out.details.get('sentiment_score', 0)})
Order Flow: {flow_out.summary} (Bias: {flow_out.details.get('flow_bias', 'BALANCED')})
Technical Setup: {tech_out.summary} (Bias: {tech_out.details.get('technical_bias', 'NEUTRAL')})"""

        # 1. Bull arguments
        bull_prompt = f"""{context}\n\nPresent your concise Bull Thesis (max 3 bullet points) emphasizing why we should buy or go long."""
        bull_case = self.bull_agent.call_llm(bull_prompt)
        if "{" in bull_case and "}" in bull_case:
            bull_case = self.bull_agent.extract_json(bull_case).get("summary", bull_case)

        # 2. Bear arguments (rebutting the bull)
        bear_prompt = f"""{context}\n\nBull arguments:\n{bull_case}\n\nRebut the bull thesis and present your concise Bear Case highlighting downside risks."""
        bear_case = self.bear_agent.call_llm(bear_prompt)
        if "{" in bear_case and "}" in bear_case:
            bear_case = self.bear_agent.extract_json(bear_case).get("summary", bear_case)

        # 3. Calculate conviction delta based on evidence scores
        sentiment_score = float(news_out.details.get("sentiment_score", 0.0))
        imbalance = float(flow_out.details.get("imbalance_ratio", 0.0))
        tech_bias = tech_out.details.get("technical_bias", "NEUTRAL")

        tech_score = 0.5 if "BULL" in tech_bias else (-0.5 if "BEAR" in tech_bias else 0.0)
        conviction_delta = (sentiment_score * 0.3) + (imbalance * 0.3) + (tech_score * 0.4)

        if conviction_delta > 0.15:
            winning_side = "BULL"
        elif conviction_delta < -0.15:
            winning_side = "BEAR"
        else:
            winning_side = "NEUTRAL"

        return DebateOutcome(
            symbol=symbol,
            bull_case=bull_case.strip()[:300],
            bear_case=bear_case.strip()[:300],
            winning_side=winning_side,
            key_vulnerability="Overhead resistance or sharp macro shift" if winning_side == "BULL" else "Strong buying support on dips",
            conviction_delta=conviction_delta,
            debate_summary=f"Debate resolved in favor of {winning_side} with conviction score {round(conviction_delta, 2)}."
        )
