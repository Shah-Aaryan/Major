"""
ML & Quantitative Trading Research Project

Main Entry Point
================

This research project studies the effectiveness and limitations of
machine-learning-assisted parameter optimization for human-defined
trading strategies.

Key Research Questions:
1. WHEN does ML optimization help?
2. HOW does ML optimization help?
3. WHEN does ML optimization fail?

IMPORTANT: ML does NOT invent new strategies.
ML ONLY adjusts parameters of user-defined rules.

Usage:
------
    python main.py --data ./data/btcusdt_1m.csv --strategy rsi_mean_reversion
    python main.py --data ./data/ --all-strategies --timeframes 1m,5m,15m
    python main.py --config ./config/experiment.json

Author: Research Project
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that degrades unsupported characters instead of failing."""

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                stream = self.stream
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        SafeStreamHandler(sys.stdout),
        logging.FileHandler(
            f'research_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8',
            errors='replace',
        )
    ]
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Ensure all required directories exist."""
    dirs = [
        './data',
        './research_output',
        './research_output/reports',
        './research_output/audit',
        './cache'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def prompt_for_strategy():
    """Prompt user to select a strategy interactively."""
    strategies = ['rsi_mean_reversion', 'ema_crossover', 'bollinger_breakout', 'custom']
    descriptions = {
        'rsi_mean_reversion': 'RSI Mean Reversion - Trades oversold/overbought conditions',
        'ema_crossover': 'EMA Crossover - Trades EMA trend crossovers',
        'bollinger_breakout': 'Bollinger Breakout - Trades price breakouts from bands',
        'custom': 'Custom Algorithm - Enter your own trading rules as a string'
    }
    
    print("\n" + "=" * 70)
    print("AVAILABLE STRATEGIES")
    print("=" * 70)
    for i, strategy in enumerate(strategies, 1):
        desc = descriptions.get(strategy, 'No description')
        print(f"{i}. {strategy:25} - {desc}")
    print("=" * 70 + "\n")
    
    while True:
        try:
            choice = input("Enter strategy number (1-4) or name: ").strip().lower()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(strategies):
                    selected = strategies[idx]
                    print(f"\n✓ Selected: {selected}\n")
                    return selected
                else:
                    print(f"❌ Please enter a number between 1 and {len(strategies)}\n")
            elif choice in strategies:
                print(f"\n✓ Selected: {choice}\n")
                return choice
            else:
                print(f"❌ Strategy '{choice}' not found. Please try again.\n")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error reading input: {e}")
            print("❌ Invalid input. Please try again.\n")


def prompt_for_algorithm():
    """Prompt user to enter custom algorithm string."""
    print("\nEnter your trading algorithm as a string.")
    print("Examples:")
    print("  EMA20 < EMA50 AND price < EMA50")
    print("  RSI crosses down from 45-60 AND volume > volume_ema_20")
    print("  price breaks below lower_bollinger_band AND EMA20 < EMA50")
    print("\nSupported syntax:")
    print("  - Comparisons: <, >, <=, >=, ==, !=")
    print("  - Crossovers: 'RSI crosses up/down from X-Y'")
    print("  - Breaks: 'price breaks above/below [indicator]'")
    print("  - Lookback: Add _1, _5, etc (e.g., RSI_1 for 1 bar ago)")
    print("  - Logic: AND, OR")
    print("  - Sell logic: Use 'SELL: [conditions]' for different exit rules\n")
    
    algorithm = input("Enter algorithm: ").strip()
    if not algorithm:
        print("❌ Algorithm cannot be empty")
        return prompt_for_algorithm()
    return algorithm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ML & Quantitative Trading Research Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python main.py --data ./data/btcusdt_1m.csv --strategy rsi_mean_reversion
  
  # Run all strategies on multiple timeframes
  python main.py --data ./data/ --all-strategies --timeframes 1m,5m,15m
  
  # Quick test with fewer optimization trials
  python main.py --data ./data/btcusdt_1m.csv --quick --trials 20
  
  # Paper trading simulation
  python main.py --paper-trade --symbol BTCUSDT
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to OHLCV data file or directory'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    
    # Strategy arguments
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Strategy to test (rsi_mean_reversion, ema_crossover, bollinger_breakout, or custom)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default=None,
        help='''Custom trading algorithm (use with --strategy custom).
                Example: EMA20 < EMA50 AND price < EMA50 AND RSI crosses down from 45-60
                Use AND/OR to combine conditions, SELL: to specify sell conditions separately'''
    )
    
    parser.add_argument(
        '--all-strategies',
        action='store_true',
        help='Test all available strategies'
    )
    
    # Timeframe arguments
    parser.add_argument(
        '--timeframes',
        type=str,
        default='5m',
        help='Comma-separated timeframes (default: 5m)'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['bayesian', 'random', 'evolutionary', 'all'],
        default='all',
        help='Optimization method (default: all)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with fewer trials'
    )
    
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Enable walk-forward validation (re-optimizes parameters periodically)'
    )
    
    parser.add_argument(
        '--wf-windows',
        type=int,
        default=5,
        help='Number of walk-forward windows (default: 5)'
    )
    
    parser.add_argument(
        '--wf-train-ratio',
        type=float,
        default=0.8,
        help='Train/test split ratio for each window (default: 0.8)'
    )
    
    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Run paper trading simulation with ML-optimized parameters'
    )

    # Hybrid human+live optimization mode
    parser.add_argument(
        '--hybrid-live',
        action='store_true',
        help=(
            'Hybrid mode: start from human params, replace last N historical windows '
            'with newest CoinGecko data, run all optimizers, and compare results'
        )
    )

    parser.add_argument(
        '--hybrid-replace-windows',
        type=int,
        default=2,
        help='How many of the last walk-forward windows to replace with CoinGecko data (default: 2)'
    )

    parser.add_argument(
        '--coingecko-days',
        type=int,
        default=1,
        help='How many days of CoinGecko data to pull for replacement (default: 1)'
    )

    parser.add_argument(
        '--hybrid-stream-seconds',
        type=int,
        default=20,
        help=(
            'How many seconds to poll CoinGecko for current price samples and merge them '
            'into the latest replacement data (default: 20). '
            'Set 0 to disable this live-like sampling.'
        )
    )

    parser.add_argument(
        '--hybrid-stream-interval',
        type=float,
        default=5.0,
        help='Seconds between CoinGecko polls during --hybrid-stream-seconds (default: 5.0)'
    )

    parser.add_argument(
        '--sample-rows',
        type=int,
        default=0,
        help='Use a smaller dataset by keeping only the last N rows (default: 0 = disabled)'
    )

    parser.add_argument(
        '--human-params',
        type=str,
        default=None,
        help='JSON object with human parameter overrides (e.g. {"rsi_buy_threshold":30,"rsi_sell_threshold":70})'
    )

    parser.add_argument(
        '--human-param',
        action='append',
        default=None,
        help='Repeatable human override as key=value (PowerShell-friendly). Example: --human-param rsi_buy_threshold=30'
    )

    parser.add_argument(
        '--human-params-file',
        type=str,
        default=None,
        help='Path to a JSON file with human parameter overrides'
    )
    
    parser.add_argument(
        '--replay-speed',
        type=float,
        default=60.0,
        help='Simulation speed multiplier (default: 60 = 1hr data per minute)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./research_output',
        help='Output directory (default: ./research_output)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def run_research_pipeline(args):
    """Run the main research pipeline."""
    from research_pipeline import ResearchPipeline
    from config.settings import OptimizationConfig, BacktestConfig, DataConfig
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Determine strategies
    if args.all_strategies:
        strategies = ['rsi_mean_reversion', 'ema_crossover', 'bollinger_breakout']
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = ['rsi_mean_reversion']  # Default
    
    # Configure trials
    n_trials = args.trials
    if args.quick:
        n_trials = min(20, n_trials)
    
    # Create configs
    opt_config = OptimizationConfig(
        bayesian_n_calls=n_trials,
        random_search_n_iter=n_trials
    )
    backtest_config = BacktestConfig()
    
    # Create pipeline
    pipeline = ResearchPipeline(
        optimization_config=opt_config,
        backtest_config=backtest_config,
        output_dir=args.output
    )
    
    # Collect data paths
    data_paths = []
    if args.data:
        data_path = Path(args.data)
        if data_path.is_file():
            data_paths = [str(data_path)]
        elif data_path.is_dir():
            data_paths = [str(f) for f in data_path.glob('*.csv')]
        else:
            # Convenience fallback: allow passing only filename and resolve in common data dirs.
            data_config = DataConfig()
            candidate_dirs = [Path(data_config.data_dir), Path('./data'), Path('./data/raw')]
            for candidate_dir in candidate_dirs:
                candidate = candidate_dir / args.data
                if candidate.is_file():
                    data_paths = [str(candidate)]
                    logger.info(f"Resolved data file to: {candidate}")
                    break
    
    if not data_paths:
        logger.error("No data files found. Please provide --data argument.")
        return None
    
    logger.info(f"Starting research with {len(data_paths)} data files")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Optimization trials: {n_trials}")
    if args.walk_forward:
        logger.info(f"Walk-Forward Mode: {args.wf_windows} windows, {args.wf_train_ratio:.0%} train ratio")
    
    # Run research
    session = pipeline.run_full_research(
        data_paths=data_paths,
        timeframes=timeframes,
        strategies=strategies,
        walk_forward=args.walk_forward,
        wf_windows=args.wf_windows,
        wf_train_ratio=args.wf_train_ratio
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RESEARCH SUMMARY")
    print("="*60)
    
    summary = session.get_summary()
    
    if 'session_id' not in summary:
        print(f"Status: {summary.get('status', 'unknown')}")
        print("No experiments were completed successfully.")
        print(f"\nCheck logs for details: {args.output}/")
        print("="*60)
        return session
    
    print(f"Session ID: {summary['session_id']}")
    print(f"Experiments: {summary['n_experiments']}")
    print(f"ML Recommended: {summary['ml_recommended_count']} ({summary['ml_recommended_pct']:.1%})")
    print(f"Avg Sharpe Improvement: {summary['avg_sharpe_improvement']:.2%}")
    print(f"Strategies: {', '.join(summary['strategies_tested'])}")
    
    if session.overall_findings.get('key_insights'):
        print("\nKey Insights:")
        for insight in session.overall_findings['key_insights']:
            print(f"  • {insight}")
    
    print(f"\nDetailed reports: {args.output}/reports/")
    print("="*60)
    
    return session


def run_paper_trading(args):
    """Run paper trading simulation with ML-optimized parameters."""
    logger.info("Paper trading mode - 1 minute timeframe")
    
    from realtime.binance_websocket import SimulatedWebSocket
    from realtime.live_feature_updater import LiveFeatureUpdater
    from realtime.paper_trader import PaperTrader, PaperTradingSession
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    from config.settings import FeatureConfig
    from datetime import datetime
    import pandas as pd
    import json
    from pathlib import Path
    import time
    
    # Load historical data for simulation
    if not args.data:
        logger.error("Please provide --data for paper trading simulation")
        return
    
    data = pd.read_csv(args.data)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    
    # Load ML-optimized parameters from latest session
    ml_params = None
    output_dir = Path(args.output)
    session_files = sorted(output_dir.glob("session_*.json"), reverse=True)
    
    if session_files:
        try:
            with open(session_files[0]) as f:
                session_data = json.load(f)
            
            # Find RSI strategy result
            for result in session_data.get('results', []):
                if result.get('strategy_name') == 'rsi_mean_reversion':
                    ml_params = result.get('ml_params')
                    logger.info(f"Loaded ML parameters from: {session_files[0].name}")
                    break
        except Exception as e:
            logger.warning(f"Could not load ML parameters: {e}")
    
    # Create strategy with ML params
    strategy = RSIMeanReversionStrategy()
    
    if ml_params:
        strategy.set_strategy_specific_params(ml_params)
        print("\n" + "="*60)
        print("USING ML-OPTIMIZED PARAMETERS:")
        print("="*60)
        for param, value in ml_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
        print("="*60 + "\n")
    else:
        print("\n[INFO] Using default human parameters (no ML optimization found)\n")
    
    # Create components for 1-minute data
    websocket = SimulatedWebSocket(
        historical_data=data,
        symbol=args.symbol,
        replay_speed=args.replay_speed  # User-configurable speed
    )
    
    feature_updater = LiveFeatureUpdater(
        symbols=[args.symbol],
        feature_config=FeatureConfig(),
        min_warmup_candles=50  # Reduced for 1m timeframe
    )
    
    # Get stop loss and take profit from ML params or defaults
    stop_loss = ml_params.get('stop_loss_pct', 2.0) / 100 if ml_params else 0.02
    take_profit = ml_params.get('take_profit_pct', 4.0) / 100 if ml_params else 0.04
    position_size = ml_params.get('position_size_pct', 10.0) / 100 if ml_params else 0.1
    
    paper_trader = PaperTrader(
        initial_capital=100000,
        position_size_pct=position_size,
        default_stop_loss_pct=stop_loss,
        default_take_profit_pct=take_profit
    )
    
    # Create session
    session = PaperTradingSession(
        websocket=websocket,
        feature_updater=feature_updater,
        strategy=strategy,
        paper_trader=paper_trader
    )
    
    print(f"\n{'='*60}")
    print("LIVE PAPER TRADING SIMULATION (1-MINUTE TIMEFRAME)")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Data: {args.data}")
    print(f"Initial Capital: $100,000")
    print(f"Stop Loss: {stop_loss*100:.1f}%")
    print(f"Take Profit: {take_profit*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}%")
    print(f"{'='*60}")
    print("Press Ctrl+C to stop\n")
    
    logger.info(f"Starting paper trading simulation for {args.symbol}")
    
    try:
        session.start()
        
        # Wait for completion
        last_print_time = time.time()
        while websocket._running:
            time.sleep(1)
            
            # Print status every 5 seconds
            if time.time() - last_print_time > 5:
                stats = session.get_session_stats()
                trading_stats = stats.get('trading_stats', {})
                
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Trades: {trading_stats.get('total_trades', 0)}, "
                      f"P&L: ${trading_stats.get('total_pnl', 0):.2f}, "
                      f"Win Rate: {trading_stats.get('win_rate', 0)*100:.1f}%", end='')
                
                last_print_time = time.time()
    
    except KeyboardInterrupt:
        logger.info("\nStopping paper trading...")
    
    finally:
        session.stop()
    
    # Print results
    metrics = paper_trader.get_performance_metrics()
    print("\n\n" + "="*60)
    print("PAPER TRADING RESULTS")
    print("="*60)
    print(f"Total Trades: {metrics['n_trades']}")
    print(f"Winning Trades: {metrics.get('n_winning', 0)}")
    print(f"Losing Trades: {metrics.get('n_losing', 0)}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print("="*60)
    
    # Show trade log
    if paper_trader.closed_trades:
        print("\nTRADE LOG:")
        for trade in paper_trader.closed_trades[-10:]:  # Last 10 trades
            print(f"  {trade.entry_time.strftime('%m/%d %H:%M')} -> {trade.exit_time.strftime('%H:%M')}: "
                  f"{trade.side.value.upper()} @ {trade.entry_price:.2f} -> {trade.exit_price:.2f} "
                  f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%) [{trade.exit_reason}]")


def run_hybrid_live(args):
    """Run hybrid human+live optimization workflow (single file, single timeframe, single strategy)."""
    from hybrid_flow import run_hybrid_live_optimization
    import json

    if not args.data:
        logger.error("Hybrid mode requires --data pointing to a CSV file")
        return None

    # Resolve a single data file
    data_path = Path(args.data)
    if data_path.is_dir():
        csvs = sorted(data_path.glob('*.csv'))
        if not csvs:
            logger.error(f"No CSV files found in directory: {data_path}")
            return None
        data_path = csvs[0]

    if not data_path.is_file():
        # Convenience fallback for bare filename
        from config.settings import DataConfig

        data_config = DataConfig()
        candidate_dirs = [Path(data_config.data_dir), Path('./data'), Path('./data/raw')]
        for candidate_dir in candidate_dirs:
            candidate = candidate_dir / args.data
            if candidate.is_file():
                data_path = candidate
                logger.info(f"Resolved data file to: {candidate}")
                break

    if not data_path.is_file():
        logger.error(f"Could not resolve data file: {args.data}")
        return None

    # Choose exactly one timeframe
    timeframes = [tf.strip() for tf in args.timeframes.split(',') if tf.strip()]
    timeframe = timeframes[0] if timeframes else '1m'
    if len(timeframes) > 1:
        logger.warning(f"Hybrid mode runs a single timeframe; using: {timeframe}")

    # Choose exactly one strategy
    if args.all_strategies:
        logger.warning("Hybrid mode runs a single strategy; using: rsi_mean_reversion")
        strategy_name = 'rsi_mean_reversion'
    else:
        strategy_name = args.strategy or 'rsi_mean_reversion'

    # Merge --human-param key=value overrides into the JSON overrides
    def _parse_cli_value(raw: str):
        raw = raw.strip()
        if raw.lower() in {'true', 'false'}:
            return raw.lower() == 'true'
        try:
            if '.' in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    merged_overrides = {}
    if args.human_param:
        for item in args.human_param:
            if '=' not in item:
                logger.warning(f"Ignoring invalid --human-param (expected key=value): {item}")
                continue
            key, value = item.split('=', 1)
            merged_overrides[key.strip()] = _parse_cli_value(value)

    human_params_json = args.human_params
    if merged_overrides:
        if human_params_json:
            try:
                base = json.loads(human_params_json)
                if isinstance(base, dict):
                    base.update(merged_overrides)
                    human_params_json = json.dumps(base)
                else:
                    human_params_json = json.dumps(merged_overrides)
            except Exception:
                human_params_json = json.dumps(merged_overrides)
        else:
            human_params_json = json.dumps(merged_overrides)

    artifacts = run_hybrid_live_optimization(
        data_path=str(data_path),
        symbol=args.symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        output_dir=args.output,
        n_trials=args.trials if not args.quick else min(20, args.trials),
        wf_windows=args.wf_windows,
        wf_train_ratio=args.wf_train_ratio,
        replace_windows=args.hybrid_replace_windows,
        coingecko_days=args.coingecko_days,
        stream_seconds=args.hybrid_stream_seconds,
        stream_interval_seconds=args.hybrid_stream_interval,
        sample_rows=args.sample_rows,
        human_params_json=human_params_json,
        human_params_file=args.human_params_file,
        algorithm=args.algorithm if hasattr(args, 'algorithm') else None,
    )

    print("\n" + "=" * 60)
    print("HYBRID LIVE OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"JSON: {artifacts.result_json_path}")
    print(f"Report: {artifacts.report_md_path}")
    print("(Edit human params and re-run using --human-params-file research_output/human_params_template.json)")
    print("=" * 60)

    return artifacts


def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     ML & Quantitative Trading Research Project               ║
║                                                              ║
║  Studying ML-assisted parameter optimization for             ║
║  human-defined trading strategies                            ║
║                                                              ║
║  ML does NOT invent strategies.                              ║
║  ML ONLY adjusts parameters of YOUR rules.                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    setup_environment()
    args = parse_args()
    
    # Interactive strategy selection if not provided
    if not args.strategy and not args.all_strategies:
        args.strategy = prompt_for_strategy()
    
    # Interactive algorithm input for custom strategy
    if args.strategy == 'custom' and not args.algorithm:
        args.algorithm = prompt_for_algorithm()
    
    # Validate custom strategy
    if args.strategy == 'custom':
        if not args.algorithm:
            print("\n❌ ERROR: --strategy custom requires --algorithm parameter")
            print("\nExample:")
            print("  python main.py --data <file.csv> --strategy custom --algorithm \"EMA20 < EMA50 AND price < EMA50\"")
            sys.exit(1)
        print(f"\n✓ Custom Algorithm: {args.algorithm}\n")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run appropriate mode
    if args.paper_trade:
        run_paper_trading(args)
    elif args.hybrid_live:
        run_hybrid_live(args)
    else:
        run_research_pipeline(args)


if __name__ == '__main__':
    main()
