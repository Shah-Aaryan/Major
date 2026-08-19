# Autonomous Algorithmic Trading & Optimization Pipeline
## Operations & User Guide

**Version:** 2.0 | **Prepared For:** Client Delivery | **Date:** August 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Running the Autonomous Pipeline](#3-running-the-autonomous-pipeline)
4. [Experiment Preset Packs](#4-experiment-preset-packs)
5. [Optimization Algorithms](#5-optimization-algorithms)
6. [Noise & Robustness Stress Testing](#6-noise--robustness-stress-testing)
7. [Nested Walk-Forward Validation](#7-nested-walk-forward-validation)
8. [Feature Intelligence & Quality Control](#8-feature-intelligence--quality-control)
9. [Explainability & Decision Reporting](#9-explainability--decision-reporting)
10. [Blockchain Audit Trail](#10-blockchain-audit-trail)
11. [Live Market Data (Binance)](#11-live-market-data-binance)
12. [Running Tests & Verification](#12-running-tests--verification)
13. [Output Files & Reports](#13-output-files--reports)
14. [Configuration Reference](#14-configuration-reference)
15. [Extending the System](#15-extending-the-system)

---

## 1. System Overview

The **Autonomous Algorithmic Trading & Optimization Pipeline** is a self-contained research and trading infrastructure that:

- Executes and auto-tunes 3 trading strategies using **15 different ML optimization algorithms**
- Generates **feature quality reports** to detect look-ahead bias and data issues
- Performs **robustness stress testing** using synthetic market noise injection
- Validates strategy performance using **nested walk-forward cross-validation** (the gold standard for financial ML)
- Produces **plain-language explanations** of every ML parameter decision
- Maintains a **cryptographic audit trail** of all signals and optimization decisions
- Connects to **live Binance market data** via REST and WebSocket
- Exports **HTML, CSV, JSON, and chart reports** automatically after every experiment

---

## 2. Prerequisites & Installation

### System Requirements

| Requirement | Minimum Version |
|---|---|
| Python | 3.10 or higher |
| Operating System | Windows 10+ / Linux / macOS |
| RAM | 4 GB (8 GB recommended for research preset) |
| Disk | 500 MB free |

---

### Step 1 — Set Up Python Environment

Open a terminal (PowerShell on Windows) in the project root folder and run:

```powershell
# Create a virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it (Linux / macOS)
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt confirming it is active.

---

### Step 2 — Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs all core and optional packages. To install manually:

```powershell
# Core (required)
pip install pandas numpy scipy scikit-learn matplotlib requests websockets pytest pytest-cov

# Optional — enables CMA-ES and DEAP genetic optimizers
pip install cmaes deap

# Optional — enables on-chain blockchain audit anchoring
pip install web3
```

---

### Step 3 — Verify Installation

```powershell
python -c "import pandas, numpy, sklearn, matplotlib; print('All dependencies OK')"
```

Expected output:
```
All dependencies OK
```

---

## 3. Running the Autonomous Pipeline

The pipeline is controlled entirely through the **`cli.py`** command-line interface. All features are accessible through this single entry point — no programming required.

### Command: `run-benchmark`

**Purpose:** Runs the full autonomous research pipeline — loads market data, generates features, optimizes strategy parameters using the selected ML algorithm, backtests performance, and exports reports.

**Basic usage:**

```powershell
python cli.py run-benchmark --data data/BTCUSDT_1h.csv
```

**Full syntax:**

```powershell
python cli.py run-benchmark [OPTIONS]
```

---

### Available Options

| Option | Default Value | Description |
|---|---|---|
| `--data` | `data/BTCUSDT_1h.csv` | Path to your CSV market data file |
| `--preset` | `balanced` | Experiment size: `fast`, `balanced`, or `research` |
| `--strategy` | `rsi_mean_reversion` | Trading strategy to optimize |
| `--optimizer` | *(preset default)* | ML optimization algorithm to use |
| `--iterations` | *(preset default)* | Number of optimizer search trials |
| `--cycles` | `5` | Number of pipeline cycles to run |
| `--symbol` | `BTCUSDT` | Market symbol (informational label) |
| `--timeframe` | `1h` | Candle timeframe (informational label) |
| `--output` | `./research_output` | Folder where all reports and charts are saved |
| `--audit-backend` | `local` | Audit mode: `local` (file-based) or `blockchain` (on-chain) |

---

### Example Commands

```powershell
# Quickest run — fast preset, RSI strategy, 5 cycles
python cli.py run-benchmark --preset fast --strategy rsi_mean_reversion --data data/BTCUSDT_1h.csv

# Balanced run — EMA crossover strategy with 10 cycles
python cli.py run-benchmark --preset balanced --strategy ema_crossover --data data/BTCUSDT_1h.csv --cycles 10

# Full research benchmark — all 50 indicators, 200 optimizer trials
python cli.py run-benchmark --preset research --strategy bollinger_breakout --data data/BTCUSDT_1h.csv --output ./full_research

# Override optimizer — use Particle Swarm with 100 trials
python cli.py run-benchmark --preset balanced --strategy rsi_mean_reversion --optimizer particle_swarm --iterations 100

# Save results to a named output folder
python cli.py run-benchmark --preset balanced --output ./client_demo_run_aug2026

# Run with blockchain audit enabled
python cli.py run-benchmark --preset fast --audit-backend blockchain --data data/BTCUSDT_1h.csv
```

---

### Available Strategies

| Strategy Name (`--strategy`) | Description |
|---|---|
| `rsi_mean_reversion` | RSI-based mean reversion — buys oversold, sells overbought |
| `ema_crossover` | Exponential Moving Average crossover trend follower |
| `bollinger_breakout` | Bollinger Band squeeze breakout momentum strategy |

---

## 4. Experiment Preset Packs

Presets are pre-configured bundles that set the feature complexity and optimizer trial budget with a single flag, making it easy to run controlled experiments at different scales.

| Preset (`--preset`) | Indicators Used | Optimizer Trials | Typical Run Time | Use Case |
|---|---|---|---|---|
| `fast` | ~10 | 15 | < 1 minute | CI/CD pipeline, quick sanity checks |
| `balanced` | ~25 | 50 | 3–5 minutes | Day-to-day research, default recommendation |
| `research` | All 50 | 200 | 15–30 minutes | Final benchmarks, publication-quality results |

**Commands:**

```powershell
# Fast — for quick iterations
python cli.py run-benchmark --preset fast --data data/BTCUSDT_1h.csv

# Balanced — recommended for most runs
python cli.py run-benchmark --preset balanced --data data/BTCUSDT_1h.csv

# Research — complete full benchmark
python cli.py run-benchmark --preset research --data data/BTCUSDT_1h.csv --output ./research_final
```

---

## 5. Optimization Algorithms

The system supports **15 different ML/metaheuristic optimization algorithms**, all operating under a unified interface so results are directly comparable.

| # | Algorithm (`--optimizer`) | Type | Description |
|---|---|---|---|
| 1 | `random_search` | Classical | Random sampling across parameter space |
| 2 | `grid_search` | Classical | Exhaustive grid over parameter bounds |
| 3 | `sobol` | Quasi-random | Low-discrepancy Sobol sequence sampling |
| 4 | `latin_hypercube` | Quasi-random | Latin Hypercube space-filling design |
| 5 | `bayesian_gp` | Bayesian | Gaussian Process surrogate model |
| 6 | `bayesian_tpe` | Bayesian | Tree-structured Parzen Estimator (TPE) |
| 7 | `cma_es` | Evolutionary | Covariance Matrix Adaptation Evolution Strategy |
| 8 | `differential_evolution` | Evolutionary | Differential Evolution (stochastic search) |
| 9 | `particle_swarm` | Swarm | Particle Swarm Optimization |
| 10 | `simulated_annealing` | Probabilistic | Simulated Annealing with cooling schedule |
| 11 | `genetic_algorithm` | Evolutionary | Genetic Algorithm with selection/crossover |
| 12 | `evolution_strategies` | Evolutionary | Evolution Strategies with self-adaptation |
| 13 | `nsga_ii` | Multi-objective | NSGA-II Pareto-front optimization |
| 14 | `nsga_iii` | Multi-objective | NSGA-III many-objective extension |
| 15 | `hyperband_asha` | Bandit | Hyperband / ASHA early-stopping scheduler |

**To benchmark all 15 algorithms back-to-back:**

```powershell
pytest tests/test_all_15_optimizers.py -v
```

**To run one specific optimizer via CLI:**

```powershell
python cli.py run-benchmark --preset balanced --optimizer bayesian_tpe --data data/BTCUSDT_1h.csv
python cli.py run-benchmark --preset balanced --optimizer cma_es --data data/BTCUSDT_1h.csv
python cli.py run-benchmark --preset balanced --optimizer nsga_ii --data data/BTCUSDT_1h.csv
```

---

## 6. Noise & Robustness Stress Testing

**Use Case:** After optimizing strategy parameters on clean historical data, stress testing evaluates how robust those parameters are when market conditions are realistically degraded — mimicking slippage, flash crashes, liquidity events, and volatility regime shifts.

### Command: `stress-test`

```powershell
python cli.py stress-test [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to clean CSV market data |
| `--strategy` | `rsi_mean_reversion` | Strategy to stress-test |
| `--noise` | `gaussian` | Type of noise to inject (see table below) |
| `--noise-sigma` | `0.001` | Gaussian noise strength (fraction of price per bar) |
| `--shock-prob` | `0.005` | Probability of a fat-tail shock event per bar |
| `--cycles` | `3` | Pipeline cycles to run on noisy data |
| `--output` | `./stress_output` | Output folder for stress-test results |
| `--seed` | `42` | Random seed for reproducibility |

### Noise Types

| `--noise` Value | Description | When to Use |
|---|---|---|
| `gaussian` | Small random price noise on every bar | Test sensitivity to tick-level noise |
| `fat_tail` | Rare but large price shocks (Student-t) | Test resilience to flash crashes |
| `regime_shift` | Mid-series volatility regime change (calm → turbulent) | Test parameter stability across market regimes |
| `full` | All noise types combined in sequence | Complete robustness validation |

### Example Commands

```powershell
# Gaussian noise stress test
python cli.py stress-test --strategy rsi_mean_reversion --data data/BTCUSDT_1h.csv --noise gaussian

# Fat-tail crash simulation
python cli.py stress-test --strategy rsi_mean_reversion --data data/BTCUSDT_1h.csv --noise fat_tail --shock-prob 0.01

# Volatility regime shift test
python cli.py stress-test --strategy ema_crossover --data data/BTCUSDT_1h.csv --noise regime_shift

# Complete stress test with all noise types, saved to named folder
python cli.py stress-test --strategy bollinger_breakout --data data/BTCUSDT_1h.csv \
    --noise full --output ./stress_results_aug2026 --seed 42
```

**What gets produced:**  
The noisy dataset is saved to `<output>/noisy_data.csv` and the full pipeline runs on it, producing the same comparison reports and charts as a standard benchmark run.

---

## 7. Nested Walk-Forward Validation

**Use Case:** The gold standard for evaluating whether ML parameter optimization actually improves real-world performance — or merely overfits to historical data. This uses a rigorous **two-level cross-validation** design:

- **Outer folds** — produce genuinely unbiased out-of-sample performance estimates
- **Inner folds** — tune hyperparameters on training data without touching the outer test set

### Command: `nested-wfv`

```powershell
python cli.py nested-wfv [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to CSV market data |
| `--strategy` | `rsi_mean_reversion` | Strategy to validate |
| `--outer` | `5` | Number of outer folds |
| `--inner` | `3` | Number of inner folds per outer fold |
| `--inner-trials` | `20` | Optimizer trials per inner fold |
| `--output` | `./nested_wfv_output` | Output folder for results |

### Example Commands

```powershell
# Standard nested walk-forward validation
python cli.py nested-wfv --strategy rsi_mean_reversion --data data/BTCUSDT_1h.csv

# More rigorous validation — more folds, more trials
python cli.py nested-wfv --strategy rsi_mean_reversion --data data/BTCUSDT_1h.csv \
    --outer 5 --inner 3 --inner-trials 50 --output ./nested_wfv_results

# Validate EMA crossover strategy
python cli.py nested-wfv --strategy ema_crossover --data data/BTCUSDT_1h.csv --outer 4 --inner 2
```

### Metrics Reported

The command prints a summary and saves `nested_wfv_result.json` to the output folder.

| Metric | Description |
|---|---|
| **Deflated Sharpe Ratio (DSR)** | Sharpe Ratio penalised for the number of optimization trials used — prevents inflated-by-search results |
| **Probability of Backtest Overfitting (PBO)** | Values above 0.50 indicate the optimizer is likely overfitting; below 0.50 means ML genuinely helps |
| **ML Consistency** | Percentage of outer folds where ML-optimized parameters outperformed the human baseline |
| **Aggregate Sharpe / Max Drawdown / Total Return** | Overall out-of-sample performance across all folds |

---

## 8. Feature Intelligence & Quality Control

The pipeline automatically runs a suite of data quality checks on every feature set. These tests are built into the pipeline and produce reports in the output folder. The modules and what they check are described below.

### Feature Leakage Detection

**Purpose:** Verifies that no indicator accidentally uses future data to predict past prices (look-ahead bias), which would make backtests unrealistically profitable.

**What is checked automatically on every run:**
- Whether any feature correlates suspiciously highly with next-bar returns (> 90% threshold)
- Whether derived indicators still contain missing values after the warm-up period
- Whether any indicator column has data before it should (possible back-fill error)

The leakage report is printed to the console during each pipeline run and included in the JSON audit log.

---

### Multicollinearity Filter

**Purpose:** Removes redundant features that carry near-identical information (correlation ≥ 0.90), keeping the feature set lean and reducing noise in the optimizer's signal.

**What it does:** Automatically compares all feature pairs. For each pair above the correlation threshold, the less "central" feature (with higher average correlation to all others) is dropped. OHLCV columns are always preserved.

This runs automatically during the pipeline. The number of features retained vs. dropped is logged at runtime.

---

### Automated Feature Ranking

**Purpose:** Ranks all indicators by how well they predict next-bar returns, using either Random Forest importance or Mutual Information scoring. Also computes importance separately per market regime.

**When to use:** Run the test suite to see the ranking for your dataset:

```powershell
pytest tests/test_feature_intelligence.py -v
```

This validates that all three quality control modules (leakage detector, correlation filter, feature selector) are functioning correctly on your system.

---

## 9. Explainability & Decision Reporting

**Purpose:** After each optimization cycle, the system automatically generates a plain-English explanation of every parameter change made by the ML algorithm, and whether it helped or hurt performance in each market regime.

### What Is Reported Automatically

Every pipeline run logs the following to the audit file and printed summary:

- **Parameter shift narratives** — e.g. *"In the 'volatile' regime, ML tightened the RSI oversold threshold from 30 → 22, improving Sharpe Ratio by 0.60 (0.80 → 1.40). RSI was at 25.0 at the time of optimization."*
- **Direction of each change** — whether a parameter was tightened, relaxed, increased, or decreased
- **Per-regime breakdown** — showing in which market regimes (trending bullish, trending bearish, ranging, volatile) ML optimization typically helps vs. hurts, and by how much on average

### Sample Console Output

```
=== Optimization Cycle 3 ===
Regime: volatile
ML improved Sharpe: 0.800 → 1.400 (+0.600)

Parameter Changes:
  rsi_buy_threshold : 30 → 22  (tightened, -26.7%)
  rsi_lookback      : 14 → 20  (increased, +42.9%)

Regime Breakdown (cumulative):
  volatile        : helped 3/4 cycles (75.0%), avg ΔSharpe = +0.45
  ranging         : helped 1/3 cycles (33.3%), avg ΔSharpe = -0.12
  trending_bullish: helped 2/2 cycles (100%), avg ΔSharpe = +0.71
```

No additional commands are needed — this output is generated automatically on every pipeline cycle.

---

## 10. Blockchain Audit Trail

**Purpose:** Every signal generated, parameter change, and optimization decision is cryptographically hashed and committed to an immutable audit trail. This ensures full reproducibility and tamper-proof record keeping.

### Local Mode (Default — No Setup Required)

All audit logs are written as JSON files to `<output_dir>/audit/`. No configuration needed.

```powershell
python cli.py run-benchmark --preset fast --audit-backend local --data data/BTCUSDT_1h.csv
```

---

### Blockchain Mode (On-Chain Verification)

Submits a **Merkle tree root hash** of all decisions to an Ethereum-compatible smart contract. Each batch of signals produces a single on-chain proof that can be independently verified by any third party.

#### Setup — Environment Variables

Before running in blockchain mode, set the following environment variables:

**Windows PowerShell:**

```powershell
$env:RPC_URL          = "https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
$env:PRIVATE_KEY      = "0xYOUR_WALLET_PRIVATE_KEY"
$env:CONTRACT_ADDRESS = "0xYOUR_DEPLOYED_CONTRACT_ADDRESS"
```

**Linux / macOS:**

```bash
export RPC_URL="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
export PRIVATE_KEY="0xYOUR_WALLET_PRIVATE_KEY"
export CONTRACT_ADDRESS="0xYOUR_DEPLOYED_CONTRACT_ADDRESS"
```

#### Run with Blockchain Auditing

```powershell
python cli.py run-benchmark --preset balanced --audit-backend blockchain --data data/BTCUSDT_1h.csv
```

Proof records are saved to `<output_dir>/audit/anchors/merkle_root_<timestamp>.json` alongside the on-chain transaction hash.

---

## 11. Live Market Data (Binance)

The pipeline supports two modes of live market data ingestion from Binance:

| Mode | Description |
|---|---|
| **REST (historical seeding)** | Downloads the last N candles via the Binance REST API to warm up indicators |
| **WebSocket (live streaming)** | Connects to the Binance WebSocket feed for real-time candle streaming with automatic reconnection |

### Running Against Live Binance Data

```powershell
python cli.py run-benchmark --strategy ema_crossover --preset fast \
    --symbol BTCUSDT --timeframe 1m --output ./live_output
```

> **Note:** When `--symbol` is specified without a `--data` CSV file, the pipeline automatically connects to the Binance live feed. Requires an active internet connection. No API key is required for public market data streams.

### Running Tests for Live Connectivity

```powershell
pytest tests/test_binance_live.py -v
```

This verifies REST kline seeding and WebSocket stream connection without requiring real funds.

---

## 12. Running Tests & Verification

The project includes **137 automated tests** across all modules. Run these after setup to confirm everything is working correctly.

### Full Test Suite

```powershell
# Run all 137 tests (recommended for initial verification)
pytest tests/ -q

# Run with detailed pass/fail output
pytest tests/ -v

# Run with code coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

Expected output:
```
137 passed in ~25s
```

---

### Run Individual Test Modules

```powershell
# Verify all 15 optimization algorithms
pytest tests/test_all_15_optimizers.py -v

# Verify feature intelligence (leakage, filter, selector, presets)
pytest tests/test_feature_intelligence.py -v

# Verify noise injection and early stopping
pytest tests/test_optimization_robustness.py -v

# Verify explainability engine and nested walk-forward
pytest tests/test_explainability_wfv.py -v

# Verify blockchain audit and Merkle proof integrity
pytest tests/test_hash_anchoring.py -v

# Verify Binance REST and WebSocket connectors
pytest tests/test_binance_live.py -v

# Verify core pipeline integration
pytest tests/test_pipeline_autonomous_loop.py -v

# Verify performance metrics (Sharpe, Sortino, drawdown)
pytest tests/test_evaluation_metrics.py -v

# Verify market data providers
pytest tests/test_market_data_provider.py -v
```

### Test Suite Coverage Map

| Test File | Tests | Module Covered |
|---|---|---|
| `test_all_15_optimizers.py` | 16 | All 15 optimization algorithms |
| `test_binance_live.py` | 3 | Binance REST + WebSocket |
| `test_evaluation_metrics.py` | 16 | Sharpe, Sortino, drawdown, win rate |
| `test_hash_anchoring.py` | 3 | Merkle proofs and audit logger |
| `test_legacy_backtest.py` | 2 | Legacy backtesting engine |
| `test_market_data_provider.py` | 22 | Dataset and synthetic providers |
| `test_pipeline_autonomous_loop.py` | 8 | Full autonomous pipeline |
| `test_feature_intelligence.py` | 19 | Leakage, filter, selector, presets |
| `test_optimization_robustness.py` | 16 | Early stopping and noise injection |
| `test_explainability_wfv.py` | 12 | Explainability engine and nested WFV |
| **Total** | **137** | **100% pass rate** |

---

## 13. Output Files & Reports

Every pipeline run automatically produces the following files in the `--output` directory:

```
research_output/
│
├── audit/
│   ├── audit_session_<timestamp>.json     Full event log: all signals, optimizations, regime changes
│   └── anchors/
│       └── merkle_root_<timestamp>.json   Blockchain proof records (blockchain mode only)
│
├── charts/
│   ├── equity_curve_cycle_001.png         Equity curve chart for cycle 1
│   ├── equity_curve_cycle_002.png         Equity curve chart for cycle 2
│   └── parameter_stability.png            Parameter drift visualization across cycles
│
├── reports/
│   ├── comparison_report.html             Interactive HTML summary (open in browser)
│   ├── comparison_report.csv              Flat metrics table (import into Excel)
│   └── comparison_report.json            Machine-readable structured benchmark data
│
└── nested_wfv_result.json                 Nested walk-forward results (nested-wfv command only)
```

### Opening the HTML Report

```powershell
# Windows — opens in default browser
Start-Process research_output/reports/comparison_report.html
```

---

## 14. Configuration Reference

### All CLI Commands — Quick Reference

| Command | Purpose |
|---|---|
| `python cli.py run-benchmark --preset fast --data <file>` | Quick pipeline run |
| `python cli.py run-benchmark --preset balanced --data <file>` | Standard research run |
| `python cli.py run-benchmark --preset research --data <file>` | Full benchmark run |
| `python cli.py stress-test --noise gaussian --data <file>` | Gaussian noise stress test |
| `python cli.py stress-test --noise fat_tail --data <file>` | Flash crash simulation |
| `python cli.py stress-test --noise regime_shift --data <file>` | Regime shift test |
| `python cli.py stress-test --noise full --data <file>` | Complete robustness test |
| `python cli.py nested-wfv --data <file>` | Nested walk-forward validation |
| `pytest tests/ -q` | Run all 137 tests |
| `pytest tests/ --cov=. --cov-report=term-missing` | Tests + coverage report |

---

### Environment Variables

| Variable | Required For | How to Set |
|---|---|---|
| `RPC_URL` | Blockchain audit anchoring | Infura / Alchemy / Quicknode endpoint URL |
| `PRIVATE_KEY` | Blockchain audit anchoring | Ethereum wallet private key (keep secret) |
| `CONTRACT_ADDRESS` | Blockchain audit anchoring | Deployed smart contract address |

---

### Data File Format

The `--data` CSV file must contain the following columns:

| Column | Description |
|---|---|
| `timestamp` | ISO datetime or Unix timestamp (used as row index) |
| `open` | Bar open price |
| `high` | Bar high price |
| `low` | Bar low price |
| `close` | Bar close price |
| `volume` | Bar trading volume |

Example first rows of a valid CSV:
```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.0,42350.0,41900.0,42200.0,1532.4
2024-01-01 01:00:00,42200.0,42500.0,42100.0,42400.0,1210.8
```

---

## 15. Extending the System

### Adding a New Trading Strategy

1. Create a new Python file inside `strategies/` implementing the strategy logic.
2. The class must inherit from `BaseStrategy` and implement three methods: `generate_signals()`, `get_parameter_bounds()`, and `get_default_parameters()`.
3. Register the strategy name in `pipeline/autonomous_loop.py` under `STRATEGY_FACTORY`.
4. The strategy will then be accessible via `--strategy <name>` in all CLI commands.

### Adding a New Optimization Algorithm

1. Create a new optimizer class inside `optimization/` inheriting from `BaseOptimizer`.
2. Register the algorithm keyword inside `MLParameterAdjuster._create_optimizer()` in `optimization/ml_parameter_adjuster.py`.
3. The optimizer will then be accessible via `--optimizer <name>`.

### Adding a New Experiment Preset

Open `features/preset_packs.py` and add a new entry to the `_PRESETS` dictionary specifying the `FeatureConfig` (indicator selection) and optimizer keyword arguments. The new preset will then be accessible via `--preset <name>`.

---

*For benchmark results, production readiness assessment, and research validation metrics, refer to `RESEARCH_VALIDATION_REPORT.md` in the project root.*
