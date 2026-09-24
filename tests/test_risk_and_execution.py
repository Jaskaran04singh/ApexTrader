import pytest
from apex_trader.agents.strategist_agent import TradeProposal
from apex_trader.data.order_flow import OrderFlowSnapshot
from apex_trader.risk.grounding_gate import GroundingGate
from apex_trader.risk.position_sizer import PositionSizer
from apex_trader.risk.risk_manager import RiskManager
from apex_trader.execution.paper_broker import PaperBroker
from apex_trader.execution.order_manager import OrderManager


@pytest.fixture
def sample_flow():
    return OrderFlowSnapshot(
        symbol="BTC/USDT",
        best_bid=65000.0,
        best_ask=65010.0,
        spread=10.0,
        spread_bps=1.5,
        microprice=65005.0,
        imbalance_ratio=0.2,
        total_bid_volume=100.0,
        total_ask_volume=80.0,
        flow_regime="MILD_BUY_PRESSURE"
    )


def test_grounding_gate_rejects_hallucinated_price(sample_flow):
    gate = GroundingGate(max_price_deviation_pct=2.0)
    # Hallucinated entry price at 80,000 when market is at 65,000 (+23% off)
    hallucinated = TradeProposal(
        symbol="BTC/USDT",
        action="BUY",
        target_entry_price=80000.0,
        stop_loss_price=78000.0,
        take_profit_price=85000.0,
        risk_reward_ratio=2.5,
        confidence_score=0.9,
        reasoning="Hallucinated price test"
    )
    verdict = gate.audit(hallucinated, sample_flow)
    assert verdict.is_grounded is False
    assert "Hallucination rejected" in verdict.reason


def test_grounding_gate_rejects_inverted_stop_loss(sample_flow):
    gate = GroundingGate()
    # Inverted stop loss (BUY with stop loss above entry price)
    bad_stop = TradeProposal(
        symbol="BTC/USDT",
        action="BUY",
        target_entry_price=65000.0,
        stop_loss_price=66000.0,  # Invalid: above entry!
        take_profit_price=68000.0,
        risk_reward_ratio=3.0,
        confidence_score=0.8,
        reasoning="Inverted stop test"
    )
    verdict = gate.audit(bad_stop, sample_flow)
    assert verdict.is_grounded is False
    assert "Logic error" in verdict.reason


def test_position_sizer():
    sizer = PositionSizer(max_portfolio_risk_pct=1.0)
    # Account $10,000, 1% risk = $100. Entry $65,000, Stop $64,000 (distance = $1,000). Qty = 100/1000 = 0.1 BTC
    res = sizer.calculate_position_size(
        account_equity=10000.0,
        entry_price=65000.0,
        stop_loss_price=64000.0,
        confidence_score=1.0
    )
    assert round(res["quantity"], 2) == 0.10
    assert res["risk_amount"] == 100.0


def test_risk_manager_circuit_breaker(sample_flow):
    rm = RiskManager(daily_max_drawdown_pct=3.0)
    proposal = TradeProposal("BTC/USDT", "BUY", 65000.0, 64000.0, 67000.0, 2.0, 0.8, "Good trade")
    
    # Starting equity $10,000, dropped to $9,600 (4% loss > 3% threshold)
    rm.starting_daily_equity = 10000.0
    decision = rm.evaluate_trade(proposal, sample_flow, open_positions_count=0, current_equity=9600.0)
    assert decision.approved is False
    assert "CIRCUIT BREAKER" in decision.rejection_reason


def test_paper_broker_execution_and_tp_trigger():
    broker = PaperBroker(initial_cash=10000.0, slippage_pct=0.0, fee_pct=0.0)
    
    # 1. Submit BUY order
    order_res = broker.submit_order(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.1,
        price=65000.0,
        stop_loss=64000.0,
        take_profit=67000.0
    )
    assert order_res.status == "FILLED"
    assert len(broker.get_open_positions()) == 1

    # 2. Simulate price increase to $67,500 (hits Take-Profit at $67,000)
    triggers = broker.update_positions_mark_price("BTC/USDT", current_price=67500.0)
    assert len(triggers) == 1
    assert triggers[0]["reason"] == "TAKE_PROFIT_TRIGGERED"
    assert len(broker.get_open_positions()) == 0
    assert broker.cash > 10000.0  # Realized profit
