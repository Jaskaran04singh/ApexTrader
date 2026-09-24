from typing import Dict, Any
from .base_agent import BaseAgent, AgentOutput
from ..data.order_flow import OrderFlowSnapshot


class OrderFlowAgent(BaseAgent):
    """Analyzes market microstructure, order book imbalance, and liquidity pressure."""

    SYSTEM_PROMPT = """You are a Market Microstructure & Order Flow Specialist.
Your job is to analyze Level-2 order book metrics (spread, microprice, bid/ask depth, and imbalance ratio) to determine short-term order book pressure.

Output strictly valid JSON with this format:
{
    "agent_name": "OrderFlowAgent",
    "flow_bias": "BULLISH_ACCUMULATION", // "BULLISH_ACCUMULATION", "BEARISH_DISTRIBUTION", "BALANCED", "LIQUIDITY_TRAP"
    "imbalance_significance": "HIGH",     // "LOW", "MEDIUM", "HIGH"
    "summary": "Short explanation of the order book dynamics",
    "confidence": 0.80                    // Float between 0.0 and 1.0
}"""

    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        super().__init__(
            name="OrderFlowAgent",
            system_prompt=self.SYSTEM_PROMPT,
            provider=provider,
            model_name=model_name
        )

    def analyze(self, flow_snapshot: OrderFlowSnapshot) -> AgentOutput:
        """Analyzes order flow snapshot."""
        prompt = f"""Order Flow Metrics:
{flow_snapshot.to_dict()}

Determine the short-term order book bias and supply/demand imbalances. Output the required JSON."""

        raw_resp = self.call_llm(prompt)
        parsed = self.extract_json(raw_resp)

        flow_bias = parsed.get("flow_bias", "BALANCED")
        significance = parsed.get("imbalance_significance", "MEDIUM")
        summary = parsed.get("summary", f"Order book shows {flow_snapshot.flow_regime} with spread of {flow_snapshot.spread_bps} bps.")
        confidence = parsed.get("confidence", 0.70)

        return AgentOutput(
            agent_name=self.name,
            summary=summary,
            confidence=float(confidence),
            details={
                "flow_bias": flow_bias,
                "imbalance_significance": significance,
                "imbalance_ratio": flow_snapshot.imbalance_ratio,
                "spread_bps": flow_snapshot.spread_bps,
                "microprice": flow_snapshot.microprice,
            }
        )

    def _heuristic_fallback(self, prompt: str) -> str:
        import json
        return json.dumps({
            "agent_name": self.name,
            "flow_bias": "BULLISH_ACCUMULATION",
            "imbalance_significance": "MEDIUM",
            "summary": "Order book indicates slight buy wall support near the best bid.",
            "confidence": 0.72
        })
