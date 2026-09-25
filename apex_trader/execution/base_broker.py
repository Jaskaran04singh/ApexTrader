from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class Position:
    """Represents an active open position with trailing stop tracking."""
    position_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: datetime
    highest_price: float = 0.0
    lowest_price: float = 0.0
    atr: float = 0.0
    trailing_stop_active: bool = False


@dataclass
class OrderResult:
    """Result of order dispatch."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str  # "FILLED", "REJECTED", "CANCELLED"
    fee: float
    message: str


class BaseBroker(ABC):
    """Abstract interface for all live and paper broker connectors."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Returns total current account equity (cash + open positions)."""
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Returns list of currently active open positions."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: float,
        price: float,
        stop_loss: float,
        take_profit: float
    ) -> OrderResult:
        """Submits and executes an order."""
        pass

    @abstractmethod
    def close_position(self, position_id: str, current_price: float, reason: str = "MANUAL") -> OrderResult:
        """Closes an active position."""
        pass

    @abstractmethod
    def update_positions_mark_price(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Updates unrealized PnL and executes stop-loss / take-profit triggers."""
        pass
