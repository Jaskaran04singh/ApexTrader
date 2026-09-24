from typing import Dict, Any
from .base_agent import BaseAgent, AgentOutput
from ..data.indicators import IndicatorSnapshot


class TechnicalAgent(BaseAgent):
    """Analyzes multi-timeframe price action, momentum, and technical indicators."""

    SYSTEM_PROMPT = """You are an elite Quantitative Technical Analyst.
Your job is to evaluate price action, momentum indicators (RSI, MACD), volatility bands (Bollinger, ATR), moving averages, and key support/resistance levels.

Output strictly valid JSON with this format:
{
    "agent_name": "TechnicalAgent",
    "technical_bias": "BULLISH",         // "STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"
    "setup_quality": "A_GRADE",          // "A_GRADE", "B_GRADE", "NO_SETUP"
    "key_trigger": "RSI recovery from oversold with MACD bullish cross",
    "summary": "Concise technical reasoning",
    "confidence": 0.85                   // Float between 0.0 and 1.0
}"""

    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        super().__init__(
            name="TechnicalAgent",
            system_prompt=self.SYSTEM_PROMPT,
            provider=provider,
            model_name=model_name
        )

    def analyze(self, symbol: str, indicators: IndicatorSnapshot) -> AgentOutput:
        """Analyzes technical indicator snapshot."""
        prompt = f"""Symbol: {symbol}
Technical Indicators:
{indicators.to_dict()}

Evaluate the setup quality, trend momentum, and indicator confluence. Output the required JSON."""

        raw_resp = self.call_llm(prompt)
        parsed = self.extract_json(raw_resp)

        bias = parsed.get("technical_bias", indicators.trend_regime)
        setup = parsed.get("setup_quality", "B_GRADE")
        trigger = parsed.get("key_trigger", f"Price at {indicators.current_price} near EMA 21.")
        summary = parsed.get("summary", f"Technical analysis indicates {bias} momentum with RSI at {indicators.rsi_14}.")
        confidence = parsed.get("confidence", 0.75)

        return AgentOutput(
            agent_name=self.name,
            summary=summary,
            confidence=float(confidence),
            details={
                "technical_bias": bias,
                "setup_quality": setup,
                "key_trigger": trigger,
                "trend_regime": indicators.trend_regime,
                "volatility_regime": indicators.volatility_regime,
                "rsi_14": indicators.rsi_14,
                "support_level": indicators.support_level,
                "resistance_level": indicators.resistance_level,
            }
        )

    def _heuristic_fallback(self, prompt: str) -> str:
        import json
        return json.dumps({
            "agent_name": self.name,
            "technical_bias": "BULLISH",
            "setup_quality": "A_GRADE",
            "key_trigger": "Momentum holding above moving averages with healthy volume.",
            "summary": "Technical setup displays solid upward continuation potential.",
            "confidence": 0.78
        })
