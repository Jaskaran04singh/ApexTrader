# ⚡ ApexTrader: Autonomous Multi-Agent Quantitative Trading System

ApexTrader is a production-grade, autonomous trading engine built on a **2-Tier Hybrid Architecture**:
1. **The Cognitive Brain (Multi-Agent Swarm):** Synthesizes live financial news, order book flow, technical indicators, and macroeconomic conditions; conducts adversarial (Bull vs. Bear) debates; and formulates grounded trade proposals.
2. **The Execution Muscle (Quant Risk & Fast Execution Engine):** Implements anti-hallucination price grounding, volatility-adjusted ATR position sizing, daily circuit breakers, and sub-second order routing to paper/live brokers.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph DataLayer ["1. Real-Time Data Ingestion Layer"]
        D1["Live Price & OHLCV Feed (CCXT / yfinance)"]
        D2["Order Flow Analyzer (L2 Spreads & Imbalance)"]
        D3["Live News & Macro Feed (RSS / Finnhub)"]
        D4["Quant Factor Engine (RSI, MACD, VWAP, ATR, Bands)"]
    end

    subgraph AgentSwarm ["2. Multi-Agent Reasoning Swarm"]
        A1["News & Sentiment Agent"]
        A2["Order Flow & Microstructure Agent"]
        A3["Technical & Quant Factor Agent"]
        A4["Macro Regime Agent"]

        Debate["Adversarial Debate Room (Bull vs. Bear)"]
        Strategist["Master Portfolio Strategist (CIO)"]
    end

    subgraph RiskLayer ["3. Quantitative Risk Shield"]
        Grounding["Grounding Gate (Anti-Hallucination Audit)"]
        RiskManager["Risk & Sizing Manager<br>• Daily Drawdown Circuit Breaker<br>• Volatility ATR Position Sizing<br>• Max Concurrent Position Limit"]
    end

    subgraph ExecutionLayer ["4. Execution & Storage Engine"]
        PaperBroker["Realistic Paper Trading Simulator (Slippage + Fees)"]
        LiveBroker["Live Broker Adapters (Alpaca / CCXT)"]
        DB[("SQLite Trade & Equity Store")]
    end

    subgraph UI ["5. Observability Dashboard"]
        CLI["Rich Terminal Live Ticker & Reasoning Stream"]
    end

    DataLayer --> AgentSwarm
    A1 & A2 & A3 & A4 --> Debate
    Debate --> Strategist
    Strategist --> Grounding
    Grounding --> RiskManager
    RiskManager --> PaperBroker & LiveBroker
    PaperBroker & LiveBroker --> DB
    DB --> CLI
```

---

## 🚀 Quick Start

### 1. Installation

Create a virtual environment and install dependencies:
```bash
uv venv
.venv\Scripts\activate      # On Windows (.venv/bin/activate on Linux/macOS)
uv pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and add your API keys (optional, fallback heuristic intelligence is active by default):
```bash
cp .env.example .env
```

Review `config.yaml` to customize your trading universe, risk parameters, and timeframe:
```yaml
trading:
  universe:
    - symbol: "BTC/USDT"
      asset_type: "crypto"
      exchange: "binance"
    - symbol: "ETH/USDT"
      asset_type: "crypto"
      exchange: "binance"
    - symbol: "SOL/USDT"
      asset_type: "crypto"
      exchange: "binance"
  timeframe: "5m"
  lookback_candles: 100

risk:
  max_portfolio_risk_pct: 1.5
  max_open_positions: 3
  daily_max_drawdown_pct: 3.0
  risk_reward_ratio_min: 1.5
```

---

## 💻 Running the System

### Single Decision Cycle (Dry Run):
```bash
python main.py --once
```

### Continuous 24/7 Live Event Loop:
```bash
python main.py --iterations 0
```

---

## 🧪 Running Unit & Integration Tests

Run the full pytest suite:
```bash
pytest -v
```

All 16 unit tests verify:
* Technical indicator math (RSI, MACD, Bollinger Bands, ATR, VWAP)
* Real-time order book flow and bid-ask spread imbalance
* Multi-agent debate resolution
* Anti-hallucination Grounding Gate
* Volatility-adjusted Kelly/ATR position sizing
* Daily drawdown circuit breaker and paper execution with auto-triggered Take-Profit/Stop-Loss
