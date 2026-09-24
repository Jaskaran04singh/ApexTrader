from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentOutput
from ..data.news_feed import NewsItem


class NewsSentimentAgent(BaseAgent):
    """Analyzes financial news headlines and extracts sentiment, catalysts, and urgency."""

    SYSTEM_PROMPT = """You are a veteran Wall Street Financial News & Catalyst Analyst.
Your job is to analyze incoming news items for a specific trading asset and provide a structured JSON assessment.

Output strictly valid JSON with this format:
{
    "agent_name": "NewsSentimentAgent",
    "sentiment_score": 0.35,          // Float between -1.0 (extremely bearish) and +1.0 (extremely bullish)
    "urgency": "MEDIUM",              // "LOW", "MEDIUM", "HIGH", "CRITICAL"
    "primary_catalyst": "Explanation of the biggest driver",
    "summary": "Concise 2-sentence summary of overall news tone",
    "confidence": 0.75                // Float between 0.0 and 1.0
}"""

    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        super().__init__(
            name="NewsSentimentAgent",
            system_prompt=self.SYSTEM_PROMPT,
            provider=provider,
            model_name=model_name
        )

    def analyze(self, symbol: str, news_items: List[NewsItem]) -> AgentOutput:
        """Analyzes a list of news items for the given symbol."""
        news_payload = [item.to_dict() for item in news_items]
        prompt = f"""Symbol: {symbol}
Recent News Items:
{news_payload}

Evaluate the sentiment, identify major catalysts, and output the required JSON."""

        raw_resp = self.call_llm(prompt)
        parsed = self.extract_json(raw_resp)

        sentiment_score = parsed.get("sentiment_score", 0.0)
        urgency = parsed.get("urgency", "MEDIUM")
        catalyst = parsed.get("primary_catalyst", "No major breaking catalysts")
        summary = parsed.get("summary", f"News analysis indicates neutral to balanced sentiment for {symbol}.")
        confidence = parsed.get("confidence", 0.65)

        return AgentOutput(
            agent_name=self.name,
            summary=summary,
            confidence=float(confidence),
            details={
                "sentiment_score": float(sentiment_score),
                "urgency": urgency,
                "primary_catalyst": catalyst,
                "news_count": len(news_items),
            }
        )

    def _heuristic_fallback(self, prompt: str) -> str:
        import json
        return json.dumps({
            "agent_name": self.name,
            "sentiment_score": 0.20,
            "urgency": "LOW",
            "primary_catalyst": "General macroeconomic steady flow",
            "summary": "Market headlines show moderate stability with no critical disruptions.",
            "confidence": 0.70
        })
