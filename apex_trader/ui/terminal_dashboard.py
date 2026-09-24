import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box


class TerminalDashboard:
    """Rich interactive terminal dashboard for live agent observability and trading monitoring."""

    def __init__(self):
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        self.console = Console(highlight=False)

    def render_cycle(
        self,
        mode: str,
        portfolio_summary: Dict[str, Any],
        watchlist_data: List[Dict[str, Any]],
        agent_decision: Optional[Dict[str, Any]] = None,
        last_action: Optional[str] = None
    ):
        """Renders the comprehensive live terminal dashboard."""
        self.console.clear()

        # 1. Header Banner
        equity = portfolio_summary.get("total_equity", 10000.0)
        open_count = portfolio_summary.get("open_positions_count", 0)
        header_text = Text()
        header_text.append("[*] APEX TRADER ", style="bold cyan")
        header_text.append("| AUTONOMOUS MULTI-AGENT QUANT ENGINE | ", style="bold white")
        header_text.append(f"MODE: [{mode.upper()}] ", style="bold magenta")
        header_text.append(f"| EQUITY: ${equity:,.2f} ", style="bold green" if equity >= 10000 else "bold red")
        header_text.append(f"| POSITIONS: {open_count} ", style="bold yellow")
        header_text.append(f"| TIME: {datetime.utcnow().strftime('%H:%M:%S UTC')}", style="dim white")

        self.console.print(Panel(header_text, style="bold blue", box=box.ROUNDED))

        # 2. Watchlist & Market Flow Table
        market_table = Table(title="[MARKET] Real-Time Watchlist & Order Flow", box=box.SIMPLE_HEAVY, expand=True)
        market_table.add_column("Symbol", style="cyan", justify="left")
        market_table.add_column("Price", style="bold white", justify="right")
        market_table.add_column("Trend", justify="center")
        market_table.add_column("RSI (14)", justify="right")
        market_table.add_column("Spread (bps)", justify="right")
        market_table.add_column("Flow Bias", justify="center")
        market_table.add_column("Volatility", justify="center")

        for item in watchlist_data:
            trend = item.get("trend", "NEUTRAL")
            trend_style = "bold green" if "BULL" in trend else ("bold red" if "BEAR" in trend else "yellow")
            
            flow = item.get("flow_bias", "BALANCED")
            flow_style = "green" if "BUY" in flow else ("red" if "SELL" in flow else "white")

            rsi = item.get("rsi", 50.0)
            rsi_style = "bold red" if rsi > 70 else ("bold green" if rsi < 30 else "white")

            market_table.add_row(
                item.get("symbol", ""),
                f"${item.get('price', 0.0):,.2f}",
                Text(trend, style=trend_style),
                Text(f"{rsi:.1f}", style=rsi_style),
                f"{item.get('spread_bps', 0.0):.1f}",
                Text(flow, style=flow_style),
                item.get("volatility", "NORMAL")
            )

        self.console.print(market_table)

        # 3. Agent Reasoning & Debate Stream
        if agent_decision:
            decision_table = Table(box=box.MINIMAL, expand=True, show_header=False)
            decision_table.add_column("Agent / Role", style="bold yellow", width=22)
            decision_table.add_column("Analysis & Verdict", style="white")

            decision_table.add_row("[Macro]", f"{agent_decision.get('macro_regime', 'RISK_ON')}")
            decision_table.add_row("[News Sentiment]", f"Score: {agent_decision.get('news_sentiment', 0.0)} | {agent_decision.get('news_summary', '')}")
            decision_table.add_row("[Order Flow]", f"{agent_decision.get('flow_summary', '')}")
            decision_table.add_row("[Technical]", f"{agent_decision.get('tech_summary', '')}")
            decision_table.add_row(
                "[Bull vs Bear]",
                f"Winner: [bold cyan]{agent_decision.get('debate_winner', 'NEUTRAL')}[/bold cyan] (Conviction: {agent_decision.get('debate_conviction', 0.0)}) | Bull: {agent_decision.get('bull_case', '')[:80]}... | Bear: {agent_decision.get('bear_case', '')[:80]}..."
            )
            
            action = agent_decision.get("proposed_action", "HOLD")
            act_style = "bold green" if action == "BUY" else ("bold red" if action == "SELL" else "bold yellow")
            
            decision_table.add_row(
                "[CIO Decision]",
                Text(
                    f"ACTION: {action} @ ${agent_decision.get('entry_price', 0.0):,.2f} | SL: ${agent_decision.get('stop_loss', 0.0):,.2f} | TP: ${agent_decision.get('take_profit', 0.0):,.2f} | Confidence: {agent_decision.get('confidence', 0.0)*100:.0f}%",
                    style=act_style
                )
            )
            
            risk_status = "APPROVED [PASS]" if agent_decision.get("risk_approved") else f"REJECTED [FAIL] ({agent_decision.get('rejection_reason')})"
            decision_table.add_row("[Risk Shield]", risk_status)

            self.console.print(Panel(decision_table, title="[AI BRAIN] Multi-Agent Collective Intelligence & Debate", border_style="cyan"))

        # 4. Active Positions Table
        positions = portfolio_summary.get("positions", [])
        if positions:
            pos_table = Table(title="[PORTFOLIO] Active Positions", box=box.SIMPLE_HEAVY, expand=True)
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Side", justify="center")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Mark Price", justify="right")
            pos_table.add_column("Stop Loss", style="red", justify="right")
            pos_table.add_column("Take Profit", style="green", justify="right")
            pos_table.add_column("PnL ($)", justify="right")
            pos_table.add_column("PnL (%)", justify="right")

            for p in positions:
                side = p.get("side", "LONG")
                side_style = "bold green" if side == "LONG" else "bold red"
                pnl = p.get("unrealized_pnl", 0.0)
                pnl_pct = p.get("unrealized_pnl_pct", 0.0)
                pnl_style = "bold green" if pnl >= 0 else "bold red"

                pos_table.add_row(
                    p.get("symbol", ""),
                    Text(side, style=side_style),
                    f"{p.get('quantity', 0.0):.4f}",
                    f"${p.get('entry_price', 0.0):,.2f}",
                    f"${p.get('current_price', 0.0):,.2f}",
                    f"${p.get('stop_loss', 0.0):,.2f}",
                    f"${p.get('take_profit', 0.0):,.2f}",
                    Text(f"${pnl:+,.2f}", style=pnl_style),
                    Text(f"{pnl_pct:+.2f}%", style=pnl_style)
                )

            self.console.print(pos_table)
        else:
            self.console.print(Panel("No open positions. Scanning market for high-conviction setups...", style="dim yellow"))

        if last_action:
            self.console.print(f"[bold white]Status:[/bold white] {last_action}")
