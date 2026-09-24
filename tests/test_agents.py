import pytest
from apex_trader.agents.base_agent import BaseAgent, AgentOutput
from apex_trader.agents.news_agent import NewsSentimentAgent
from apex_trader.agents.flow_agent import OrderFlowAgent
from apex_trader.agents.technical_agent import TechnicalAgent
from apex_trader.agents.macro_agent import MacroRegimeAgent
from apex_trader.agents.debate_room import BullBearDebateRoom, DebateOutcome
from apex_trader.agents.strategist_agent import MasterStrategistAgent, TradeProposal
from apex_trader.data.news_feed import NewsItem
from apex_trader.data.order_flow import OrderFlowSnapshot
from apex_trader.data.indicators import IndicatorSnapshot


def test_base_agent_json_extraction():
    agent = BaseAgent("TestAgent", "System prompt")
    raw_markdown = '```json\n{"key": "value", "number": 42}\n```'
    parsed = agent.extract_json(raw_markdown)
    assert parsed.get("key") == "value"
    assert parsed.get("number") == 42


def test_news_agent_analyze():
    agent = NewsSentimentAgent()
    news = [NewsItem("BTC Surges to New High", "Bullish momentum continues.", "CryptoNews", "2026-09-24", "http://...", 0.8)]
    out = agent.analyze("BTC/USDT", news)
    assert isinstance(out, AgentOutput)
    assert out.agent_name == "NewsSentimentAgent"
    assert -1.0 <= out.details.get("sentiment_score", 0.0) <= 1.0


def test_flow_agent_analyze():
    agent = OrderFlowAgent()
    flow = OrderFlowSnapshot(
        symbol="BTC/USDT",
        best_bid=65000.0,
        best_ask=65010.0,
        spread=10.0,
        spread_bps=1.5,
        microprice=65005.0,
        imbalance_ratio=0.35,
        total_bid_volume=150.0,
        total_ask_volume=80.0,
        flow_regime="MILD_BUY_PRESSURE"
    )
    out = agent.analyze(flow)
    assert isinstance(out, AgentOutput)
    assert "flow_bias" in out.details


def test_debate_room():
    room = BullBearDebateRoom()
    news_out = AgentOutput(agent_name="News", summary="Positive catalysts", confidence=0.8, details={"sentiment_score": 0.5})
    flow_out = AgentOutput(agent_name="Flow", summary="Order book accumulation", confidence=0.8, details={"imbalance_ratio": 0.3, "flow_bias": "BULLISH"})
    tech_out = AgentOutput(agent_name="Tech", summary="Golden cross breakout", confidence=0.85, details={"technical_bias": "BULLISH"})

    debate = room.conduct_debate("BTC/USDT", news_out, flow_out, tech_out)
    assert isinstance(debate, DebateOutcome)
    assert debate.winning_side in ["BULL", "BEAR", "NEUTRAL"]
    assert -1.0 <= debate.conviction_delta <= 1.0
