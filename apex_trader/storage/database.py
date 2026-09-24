import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database storage for trade logs, agent decisions, and equity history."""

    def __init__(self, db_path: str = "storage/apex_trader.db"):
        self.db_path = db_path
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Agent Decisions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                macro_regime TEXT,
                news_sentiment REAL,
                flow_bias TEXT,
                technical_bias TEXT,
                debate_winner TEXT,
                debate_conviction REAL,
                proposed_action TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                reasoning TEXT,
                risk_approved INTEGER,
                rejection_reason TEXT
            )
            """)

            # 2. Executed Trades Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT NOT NULL,
                fee REAL NOT NULL,
                realized_pnl REAL DEFAULT 0.0,
                closed_at TEXT
            )
            """)

            # 3. Portfolio Equity Curve Snapshots
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_equity REAL NOT NULL,
                cash REAL NOT NULL,
                open_positions_count INTEGER NOT NULL
            )
            """)
            conn.commit()

    def log_agent_decision(self, record: Dict[str, Any]):
        """Logs a full multi-agent decision cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO agent_decisions (
                timestamp, symbol, macro_regime, news_sentiment, flow_bias,
                technical_bias, debate_winner, debate_conviction, proposed_action,
                entry_price, stop_loss, take_profit, confidence, reasoning,
                risk_approved, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                record.get("symbol"),
                record.get("macro_regime"),
                record.get("news_sentiment"),
                record.get("flow_bias"),
                record.get("technical_bias"),
                record.get("debate_winner"),
                record.get("debate_conviction"),
                record.get("proposed_action"),
                record.get("entry_price"),
                record.get("stop_loss"),
                record.get("take_profit"),
                record.get("confidence"),
                record.get("reasoning"),
                1 if record.get("risk_approved") else 0,
                record.get("rejection_reason"),
            ))
            conn.commit()

    def log_trade(self, trade_data: Dict[str, Any]):
        """Logs an executed trade."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO trades (
                order_id, timestamp, symbol, side, quantity,
                entry_price, stop_loss, take_profit, status, fee
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get("order_id"),
                datetime.utcnow().isoformat(),
                trade_data.get("symbol"),
                trade_data.get("side"),
                trade_data.get("quantity"),
                trade_data.get("price"),
                trade_data.get("stop_loss"),
                trade_data.get("take_profit"),
                trade_data.get("status"),
                trade_data.get("fee", 0.0),
            ))
            conn.commit()

    def log_equity_snapshot(self, equity: float, cash: float, open_positions_count: int):
        """Logs real-time portfolio equity snapshot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO equity_history (timestamp, total_equity, cash, open_positions_count)
            VALUES (?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                equity,
                cash,
                open_positions_count
            ))
            conn.commit()

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trade_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
