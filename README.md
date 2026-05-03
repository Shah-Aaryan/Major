# ML & Quantitative Trading Research Platform

A robust Python-based research framework designed to study the effectiveness and limitations of machine-learning-assisted parameter optimization applied to human-defined algorithmic trading strategies.

## 🎯 Core Research Philosophy

**ML does NOT invent new strategies; it ONLY optimizes the parameters of user-defined rules.**

This project is built around three core research questions:
1. **WHEN** does ML optimization help a trading strategy?
2. **HOW** does ML optimization improve performance (e.g., parameter tuning vs. regime detection)?
3. **WHEN** does ML optimization fail or overfit the market?

## ✨ Key Features

- **Multiple Optimization Algorithms:** Supports Bayesian Optimization (`scikit-optimize`, `optuna`), Random Search, and Evolutionary Algorithms (`deap`).
- **Pre-built Strategies:** Includes `rsi_mean_reversion`, `ema_crossover`, and `bollinger_breakout` as baseline human-defined rules.
- **Walk-Forward Validation:** Robust backtesting framework with walk-forward capabilities to prevent and detect overfitting.
- **Paper Trading Simulator:** Test ML-optimized parameters in a simulated live environment using real-time websocket data feeds.
- **Hybrid Human + Live Optimization:** A specialized mode that seeds the optimizer with human intuitions, integrates live data from CoinGecko, and compares human performance vs. ML fine-tuning.
- **Natural Language Parsing:** Support for converting human-readable trading rules into structured parameters for the execution engine.

## 📁 Project Structure

```text
├── backtesting/      # Walk-forward validation and backtest engine
├── backend/          # Backend APIs and infrastructure 
├── config/           # Configuration structures (settings, experiment configs)
├── data/             # Historical OHLCV datasets
├── features/         # Feature engineering and live feature updating
├── optimization/     # ML optimization modules (Optuna, Scikit-Optimize, DEAP)
├── realtime/         # Simulated WebSocket and Paper Trading engine
├── strategies/       # Core strategy engine, base strategies, and human rule parser
├── main.py           # Main CLI entry point
├── hybrid_flow.py    # Hybrid human+live optimization workflow
└── research_pipeline.py # End-to-end backtesting & optimization pipeline
```

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create and activate a virtual environment (Recommended Python 3.10+):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

The platform is driven via a comprehensive CLI in `main.py`.

### 1. Backtesting & ML Optimization
Run optimization for a specific strategy on historical data:
```bash
python main.py --data ./data/btcusdt_1m.csv --strategy rsi_mean_reversion --trials 100
```

Run a walk-forward optimization test on all strategies across multiple timeframes:
```bash
python main.py --data ./data/ --all-strategies --timeframes 1m,5m,15m --walk-forward --wf-windows 5
```

### 2. Paper Trading Simulation
Test the best ML-optimized parameters in a simulated live environment:
```bash
python main.py --data ./data/btcusdt_1m.csv --paper-trade --symbol BTCUSDT --replay-speed 60
```

### 3. Hybrid Human+Live Optimization
Start with human parameters, pull the latest data from CoinGecko, and run optimization to see if ML can beat human intuition:
```bash
python main.py --data ./data/btcusdt_1m.csv --hybrid-live --symbol BTCUSDT \
  --human-param rsi_buy_threshold=30 \
  --human-param rsi_sell_threshold=70 \
  --coingecko-days 1
```

## 📊 Output & Reporting

All results are saved in the `research_output/` directory (or a custom directory specified by `--output`).
- **`session_*.json`**: Contains detailed JSON logs of all parameters, trials, and backtest results.
- **`reports/`**: Generates Markdown reports detailing performance, Sharpe ratio improvements, and ML recommendations.

## 📜 License

This project is licensed under the MIT License.
