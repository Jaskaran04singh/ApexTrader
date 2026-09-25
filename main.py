import os
import sys
import time
import argparse
from typing import Dict, Any, List

from apex_trader.config import load_config
from apex_trader.data import (
    PriceFeedManager,
    OrderFlowAnalyzer,
    NewsFeedCollector,
    calculate_indicators,
)
from apex_trader.agents import (
    NewsSentimentAgent,
    OrderFlowAgent,
    TechnicalAgent,
    MacroRegimeAgent,
    BullBearDebateRoom,
    MasterStrategistAgent,
)
from apex_trader.risk import RiskManager
from apex_trader.execution import PaperBroker, OrderManager
from apex_trader.storage import DatabaseManager
from apex_trader.ui import TerminalDashboard


class ApexTraderEngine:
    """Master Orchestrator for the ApexTrader multi-agent autonomous trading system."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # 1. Initialize Data Feeds
        self.price_feed = PriceFeedManager(exchange_id="binance")
        self.order_flow = OrderFlowAnalyzer(exchange_id="binance")
        self.news_feed = NewsFeedCollector(finnhub_api_key=self.config.finnhub_api_key)

        # 2. Initialize Agent Swarm
        provider = self.config.agents.llm_provider
        model_name = self.config.agents.model_name

        self.news_agent = NewsSentimentAgent(provider=provider, model_name=model_name)
        self.flow_agent = OrderFlowAgent(provider=provider, model_name=model_name)
        self.tech_agent = TechnicalAgent(provider=provider, model_name=model_name)
        self.macro_agent = MacroRegimeAgent(provider=provider, model_name=model_name)
        self.debate_room = BullBearDebateRoom(provider=provider, model_name=model_name)
        self.strategist = MasterStrategistAgent(provider=provider, model_name=model_name)

        # 3. Initialize Risk Shield
        self.risk_manager = RiskManager(
            max_portfolio_risk_pct=self.config.risk.max_portfolio_risk_pct,
            max_open_positions=self.config.risk.max_open_positions,
            daily_max_drawdown_pct=self.config.risk.daily_max_drawdown_pct,
            confidence_threshold=self.config.agents.confidence_threshold,
            min_risk_reward=self.config.risk.risk_reward_ratio_min,
        )

        # 4. Initialize Execution & Storage
        self.broker = PaperBroker(initial_cash=10000.0)
        self.order_manager = OrderManager(broker=self.broker)
        self.db = DatabaseManager(db_path=self.config.system.db_path)
        self.dashboard = TerminalDashboard()

    def run_single_cycle(self) -> Dict[str, Any]:
        """Runs one full perception-reasoning-risk-execution cycle across the universe."""
        watchlist_summary: List[Dict[str, Any]] = []
        last_decision: Dict[str, Any] = None
        last_status_msg: str = "Scanning markets..."

        # Step 1: Collect Data for all assets in universe
        asset_snapshots = []
        for asset in self.config.trading.universe:
            symbol = asset.symbol
            ohlcv = self.price_feed.fetch_ohlcv(
                symbol=symbol,
                asset_type=asset.asset_type,
                timeframe=self.config.trading.timeframe,
                limit=self.config.trading.lookback_candles
            )
            flow = self.order_flow.fetch_order_flow(symbol=symbol)
            indicators = calculate_indicators(ohlcv.df)

            # Sync open positions against latest market price
            closed_triggers = self.order_manager.sync_market_prices(symbol, flow.best_bid)
            for trigger in closed_triggers:
                last_status_msg = f"Auto-closed: {trigger['reason']} for {symbol}"

            asset_snapshots.append({
                "symbol": symbol,
                "asset_type": asset.asset_type,
                "ohlcv": ohlcv,
                "flow": flow,
                "indicators": indicators
            })

            watchlist_summary.append({
                "symbol": symbol,
                "price": ohlcv.current_price,
                "trend": indicators.trend_regime,
                "rsi": indicators.rsi_14,
                "spread_bps": flow.spread_bps,
                "flow_bias": flow.flow_regime,
                "volatility": indicators.volatility_regime
            })

        # Step 2: Macro Regime Assessment
        macro_out = self.macro_agent.analyze(watchlist_summary)

        # Step 3: Deep Multi-Agent Analysis on the primary high-momentum asset
        for item in asset_snapshots:
            symbol = item["symbol"]
            flow = item["flow"]
            indicators = item["indicators"]
            current_price = item["ohlcv"].current_price

            # News Analysis
            news_items = self.news_feed.fetch_news(category=item["asset_type"], limit=4)
            news_out = self.news_agent.analyze(symbol, news_items)

            # Flow Analysis
            flow_out = self.flow_agent.analyze(flow)

            # Technical Analysis
            tech_out = self.tech_agent.analyze(symbol, indicators)

            # Adversarial Debate
            debate = self.debate_room.conduct_debate(symbol, news_out, flow_out, tech_out)

            # Master Strategist Proposal
            proposal = self.strategist.formulate_trade_proposal(
                symbol=symbol,
                current_price=current_price,
                indicators=indicators,
                news_out=news_out,
                flow_out=flow_out,
                tech_out=tech_out,
                macro_out=macro_out,
                debate=debate
            )

            # Risk Shield Audit
            current_equity = self.broker.get_account_equity()
            open_count = len(self.broker.get_open_positions())
            risk_decision = self.risk_manager.evaluate_trade(
                proposal=proposal,
                flow=flow,
                open_positions_count=open_count,
                current_equity=current_equity
            )

            # Execution (if approved)
            if risk_decision.approved:
                order_result = self.order_manager.execute_risk_approved_trade(risk_decision)
                last_status_msg = f"EXECUTED: {order_result.message}"
                self.db.log_trade({
                    "order_id": order_result.order_id,
                    "symbol": order_result.symbol,
                    "side": order_result.side,
                    "quantity": order_result.quantity,
                    "price": order_result.price,
                    "stop_loss": proposal.stop_loss_price,
                    "take_profit": proposal.take_profit_price,
                    "status": order_result.status,
                    "fee": order_result.fee
                })

            # Record Decision
            last_decision = {
                "symbol": symbol,
                "macro_regime": macro_out.details.get("macro_regime"),
                "news_sentiment": news_out.details.get("sentiment_score"),
                "news_summary": news_out.summary,
                "flow_bias": flow_out.details.get("flow_bias"),
                "flow_summary": flow_out.summary,
                "technical_bias": tech_out.details.get("technical_bias"),
                "tech_summary": tech_out.summary,
                "debate_winner": debate.winning_side,
                "debate_conviction": debate.conviction_delta,
                "bull_case": debate.bull_case,
                "bear_case": debate.bear_case,
                "proposed_action": proposal.action,
                "entry_price": proposal.target_entry_price,
                "stop_loss": proposal.stop_loss_price,
                "take_profit": proposal.take_profit_price,
                "confidence": proposal.confidence_score,
                "reasoning": proposal.reasoning,
                "risk_approved": risk_decision.approved,
                "rejection_reason": risk_decision.rejection_reason,
            }
            self.db.log_agent_decision(last_decision)
            break  # Deep analyze top priority candidate per cycle

        # Step 4: Record & Render Dashboard
        portfolio_summary = self.order_manager.get_portfolio_summary()
        self.db.log_equity_snapshot(
            equity=portfolio_summary["total_equity"],
            cash=self.broker.cash,
            open_positions_count=portfolio_summary["open_positions_count"]
        )

        self.dashboard.render_cycle(
            mode=self.config.system.mode,
            portfolio_summary=portfolio_summary,
            watchlist_data=watchlist_summary,
            agent_decision=last_decision,
            last_action=last_status_msg
        )

        return portfolio_summary

    def start_loop(self, iterations: int = 0):
        """Runs continuous event loop."""
        count = 0
        try:
            while True:
                count += 1
                self.run_single_cycle()
                if iterations > 0 and count >= iterations:
                    break
                time.sleep(self.config.system.poll_interval_seconds)
        except KeyboardInterrupt:
            print("\n[ApexTrader] Shutdown requested by user.")


    def print_performance_stats(self):
        """Displays formatted total profit/loss and performance statistics."""
        stats = self.db.get_performance_summary(initial_balance=10000.0)
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        import rich.box as box

        console = Console()
        table = Table(title="[PORTFOLIO PERFORMANCE & PnL SUMMARY]", box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="bold white", justify="right")

        pnl = stats["net_pnl"]
        pnl_pct = stats["net_pnl_pct"]
        pnl_style = "bold green" if pnl >= 0 else "bold red"
        pnl_text = f"${pnl:+,.2f} ({pnl_pct:+.2f}%)"

        table.add_row("Starting Balance", f"${stats['initial_balance']:,.2f}")
        table.add_row("Current Account Equity", f"${stats['current_equity']:,.2f}")
        table.add_row("Available Cash", f"${stats['cash']:,.2f}")
        table.add_row("Open Positions", str(stats["open_positions"]))
        table.add_row("Total Executed Trades", str(stats["total_trades"]))
        table.add_row("Closed Positions", str(stats["closed_trades"]))
        table.add_row("Win Rate", f"{stats['win_rate_pct']:.1f}% ({stats['win_count']} wins)")
        table.add_row("Total Fees Paid", f"${stats['total_fees_paid']:,.2f}")
        table.add_row("Net Overall PnL", Text(pnl_text, style=pnl_style))
        table.add_row("Profit Status", Text("PROFITABLE [GAIN]" if stats["is_profitable"] else "DRAWDOWN [LOSS]", style=pnl_style))

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="ApexTrader Autonomous Trading System")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle and exit")
    parser.add_argument("--iterations", type=int, default=0, help="Number of cycles to run (0 = infinite)")
    parser.add_argument("--stats", action="store_true", help="Display overall Profit/Loss and performance statistics from database")
    args = parser.parse_args()

    engine = ApexTraderEngine(config_path=args.config)
    if args.stats:
        engine.print_performance_stats()
    elif args.once:
        engine.run_single_cycle()
    else:
        engine.start_loop(iterations=args.iterations)


if __name__ == "__main__":
    main()
