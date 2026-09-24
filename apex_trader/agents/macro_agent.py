from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentOutput


class MacroRegimeAgent(BaseAgent):
    """Evaluates macro environment, broad market risk regime, and systemic volatility."""

    SYSTEM_PROMPT = """You are a Global Macro Strategist & Risk Regime Analyst.
Your role is to assess the macro landscape and classify current market conditions into risk regimes.

Output strictly valid JSON with this format:
{
    "agent_name": "MacroRegimeAgent",
    "macro_regime": "RISK_ON",        // "RISK_ON", "RISK_OFF", "CHOPPY_RANGEBOUND", "HIGH_VOLATILITY_EXPANSION"
    "market_permission": "ALLOW_ALL", // "ALLOW_ALL", "ALLOW_LONGS_ONLY", "ALLOW_SHORTS_ONLY", "HALT_TRADING"
    "summary": "Concise 2-sentence macro synthesis",
    "confidence": 0.80                // Float between 0.0 and 1.0
}"""

    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        super().__init__(
            name="MacroRegimeAgent",
            system_prompt=self.SYSTEM_PROMPT,
            provider=provider,
            model_name=model_name
        )

    def analyze(self, asset_summaries: List[Dict[str, Any]]) -> AgentOutput:
        """Evaluates macro regime across watchlist assets."""
        prompt = f"""Watchlist Summary:
{asset_summaries}

Classify the overall macro market regime and determine trade permission."""

        raw_resp = self.call_llm(prompt)
        parsed = self.extract_json(raw_resp)

        regime = parsed.get("macro_regime", "RISK_ON")
        permission = parsed.get("market_permission", "ALLOW_ALL")
        summary = parsed.get("summary", "Macro conditions remain constructive for momentum and directional flow.")
        confidence = parsed.get("confidence", 0.75)

        return AgentOutput(
            agent_name=self.name,
            summary=summary,
            confidence=float(confidence),
            details={
                "macro_regime": regime,
                "market_permission": permission,
            }
        )

    def _heuristic_fallback(self, prompt: str) -> str:
        import json
        return json.dumps({
            "agent_name": self.name,
            "macro_regime": "RISK_ON",
            "market_permission": "ALLOW_ALL",
            "summary": "Macro environment remains stable with constructive liquidity backdrop.",
            "confidence": 0.75
        })
