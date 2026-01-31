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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'research_{datetime.now().strftime("%Y%m%d")}.log')
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
        choices=['rsi_mean_reversion', 'ema_crossover', 'bollinger_breakout'],
        help='Strategy to test'
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
        '--paper-trade',
        action='store_true',
        help='Run paper trading simulation'
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
    from config.settings import OptimizationConfig, BacktestConfig
    
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
    
    if not data_paths:
        logger.error("No data files found. Please provide --data argument.")
        return None
    
    logger.info(f"Starting research with {len(data_paths)} data files")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Optimization trials: {n_trials}")
    
    # Run research
    session = pipeline.run_full_research(
        data_paths=data_paths,
        timeframes=timeframes,
        strategies=strategies
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RESEARCH SUMMARY")
    print("="*60)
    
    summary = session.get_summary()
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
    """Run paper trading simulation."""
    logger.info("Paper trading mode")
    
    from realtime.binance_websocket import SimulatedWebSocket
    from realtime.live_feature_updater import LiveFeatureUpdater
    from realtime.paper_trader import PaperTrader, PaperTradingSession
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    from config.settings import FeatureConfig
    import pandas as pd
    
    # Load historical data for simulation
    if not args.data:
        logger.error("Please provide --data for paper trading simulation")
        return
    
    data = pd.read_csv(args.data)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    
    # Create components
    websocket = SimulatedWebSocket(
        historical_data=data,
        symbol=args.symbol,
        replay_speed=10.0  # 10x speed for testing
    )
    
    feature_updater = LiveFeatureUpdater(
        symbols=[args.symbol],
        feature_config=FeatureConfig(),
        min_warmup_candles=100
    )
    
    strategy = RSIMeanReversionStrategy()
    paper_trader = PaperTrader(initial_capital=100000)
    
    # Create session
    session = PaperTradingSession(
        websocket=websocket,
        feature_updater=feature_updater,
        strategy=strategy,
        paper_trader=paper_trader
    )
    
    logger.info(f"Starting paper trading simulation for {args.symbol}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        session.start()
        
        # Wait for completion
        import time
        while websocket._running:
            time.sleep(1)
            stats = session.get_session_stats()
            if stats['trading_stats']['total_trades'] > 0:
                print(f"Trades: {stats['trading_stats']['total_trades']}, "
                      f"P&L: ${stats['trading_stats']['total_pnl']:.2f}")
    
    except KeyboardInterrupt:
        logger.info("Stopping paper trading...")
    
    finally:
        session.stop()
    
    # Print results
    metrics = paper_trader.get_performance_metrics()
    print("\n" + "="*60)
    print("PAPER TRADING RESULTS")
    print("="*60)
    print(f"Total Trades: {metrics['n_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print("="*60)


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
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run appropriate mode
    if args.paper_trade:
        run_paper_trading(args)
    else:
        run_research_pipeline(args)


if __name__ == '__main__':
    main()
