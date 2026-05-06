"""
Debug script to show available feature columns for custom algorithm development.
"""

import pandas as pd
import sys
from pathlib import Path
from config.settings import DataConfig, FeatureConfig
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine

def show_feature_columns(data_path: str, timeframe: str, sample_rows: int = 100):
    """Load data and show available feature columns."""
    
    print("\n" + "=" * 70)
    print("FEATURE COLUMNS AVAILABLE FOR CUSTOM ALGORITHMS")
    print("=" * 70)
    
    # Load and preprocess data
    data_config = DataConfig()
    loader = DataLoader(data_config.data_dir)
    preprocessor = DataPreprocessor()
    resampler = DataResampler()
    feature_engine = FeatureEngine(FeatureConfig())
    
    print(f"\nLoading: {data_path}")
    raw = loader.load_csv(data_path)
    print(f"Loaded {len(raw)} rows")
    
    # Preprocess
    data = preprocessor.preprocess(raw)
    print(f"After preprocessing: {len(data)} rows")
    
    # Resample
    data = resampler.resample(data, timeframe)
    print(f"After resampling to {timeframe}: {len(data)} rows")
    
    # Sample if requested
    if sample_rows and len(data) > sample_rows:
        data = data.tail(sample_rows)
        print(f"Using last {sample_rows} rows")
    
    # Generate features
    print("\nGenerating features...")
    data_with_features = feature_engine.generate_features(data)
    
    print(f"\nTotal columns: {len(data_with_features.columns)}")
    print("\n" + "-" * 70)
    print("AVAILABLE COLUMNS (for use in --algorithm):")
    print("-" * 70)
    
    # Group columns by type
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    trend = [c for c in data_with_features.columns if 'ema' in c.lower() or 'sma' in c.lower()]
    momentum = [c for c in data_with_features.columns if 'rsi' in c.lower() or 'macd' in c.lower() or 'momentum' in c.lower()]
    volatility = [c for c in data_with_features.columns if 'bb_' in c.lower() or 'atr' in c.lower() or 'volatility' in c.lower()]
    volume = [c for c in data_with_features.columns if 'volume' in c.lower() and c not in ohlcv]
    other = [c for c in data_with_features.columns if c not in ohlcv + trend + momentum + volatility + volume]
    
    print("\nOHLCV (Price & Volume):")
    for col in ohlcv:
        if col in data_with_features.columns:
            print(f"  - {col}")
    
    if trend:
        print("\nTrend Indicators:")
        for col in sorted(trend):
            print(f"  - {col}")
    
    if momentum:
        print("\nMomentum Indicators:")
        for col in sorted(momentum):
            print(f"  - {col}")
    
    if volatility:
        print("\nVolatility Indicators:")
        for col in sorted(volatility):
            print(f"  - {col}")
    
    if volume:
        print("\nVolume Indicators:")
        for col in sorted(volume):
            print(f"  - {col}")
    
    if other:
        print("\nOther Indicators:")
        for col in sorted(other):
            print(f"  - {col}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE ALGORITHMS:")
    print("=" * 70)
    print("""
1. Simple EMA crossover:
   EMA20 < EMA50 AND price < EMA50

2. RSI mean reversion:
   RSI14 < 30 AND price > lower_bollinger_band

3. Breakout:
   price breaks above upper_bollinger_band AND volume > volume_ema_20

4. Complex (combining multiple indicators):
   EMA20 < EMA50 AND RSI14 crosses down from 45-60 AND volume > volume_ema_20

NOTES:
- Column names are case-insensitive (EMA20, ema20, Ema20 all work)
- Use underscores for multi-word indicators (e.g., volume_ema_20)
- Supported operators: <, >, <=, >=, ==, !=
- Supported keywords: "crosses up/down from X-Y", "breaks above/below [indicator]"
- Combine conditions with AND/OR
- Use "SELL: [conditions]" for different exit rules (optional)
""")
    print("=" * 70 + "\n")
    
    return data_with_features

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_features.py <data_file.csv> [timeframe] [sample_rows]")
        print("Example: python debug_features.py data/raw/OHLCV_Binance_BTC-USDT_1min.csv 1m 1000")
        sys.exit(1)
    
    data_path = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1m'
    sample_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    show_feature_columns(data_path, timeframe, sample_rows)
