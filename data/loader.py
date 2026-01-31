"""
OHLCV Data Loader Module.

Handles loading of cryptocurrency OHLCV data from CSV files.
Supports multiple symbols and date range filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and manages OHLCV data from CSV files.
    
    This class handles:
    - Loading individual or multiple symbol CSV files
    - Timestamp parsing and validation
    - Date range filtering
    - Data quality checks
    
    Expected CSV format (Kaggle Binance dataset):
    timestamp, open, high, low, close, volume
    """
    
    # Expected columns in OHLCV data
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing OHLCV CSV files
        """
        self.data_dir = Path(data_dir)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
    
    def list_available_files(self) -> List[str]:
        """List all CSV files in the data directory."""
        if not self.data_dir.exists():
            return []
        return [f.stem for f in self.data_dir.glob("*.csv")]
    
    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        cache_key = symbol
        
        # Check cache first
        if use_cache and cache_key in self._data_cache:
            df = self._data_cache[cache_key].copy()
            return self._filter_date_range(df, start_date, end_date)
        
        # Find the file
        file_path = self._find_file(symbol)
        if file_path is None:
            raise FileNotFoundError(f"No data file found for symbol: {symbol}")
        
        # Load the data
        df = self._load_csv(file_path)
        
        # Cache the full dataset
        if use_cache:
            self._data_cache[cache_key] = df.copy()
        
        # Apply date filter
        return self._filter_date_range(df, start_date, end_date)
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            start_date: Start date string
            end_date: End date string
            parallel: Whether to load files in parallel
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as executor:
                futures = {
                    executor.submit(
                        self.load_symbol, symbol, start_date, end_date
                    ): symbol for symbol in symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        results[symbol] = future.result()
                        logger.info(f"Loaded {symbol}: {len(results[symbol])} rows")
                    except Exception as e:
                        logger.error(f"Failed to load {symbol}: {e}")
        else:
            for symbol in symbols:
                try:
                    results[symbol] = self.load_symbol(symbol, start_date, end_date)
                    logger.info(f"Loaded {symbol}: {len(results[symbol])} rows")
                except Exception as e:
                    logger.error(f"Failed to load {symbol}: {e}")
        
        return results
    
    def _find_file(self, symbol: str) -> Optional[Path]:
        """Find CSV file for a symbol."""
        # Try exact match first
        exact_path = self.data_dir / f"{symbol}.csv"
        if exact_path.exists():
            return exact_path
        
        # Try case-insensitive match
        for f in self.data_dir.glob("*.csv"):
            if f.stem.upper() == symbol.upper():
                return f
        
        # Try partial match (e.g., BTCUSDT_1m.csv)
        for f in self.data_dir.glob(f"*{symbol}*.csv"):
            return f
        
        return None
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load and parse a CSV file.
        
        Handles various timestamp formats and column naming conventions.
        """
        logger.info(f"Loading file: {file_path}")
        
        # Try to infer the format
        df = pd.read_csv(file_path)
        
        # Normalize column names (lowercase, strip whitespace)
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle various column naming conventions
        column_mapping = {
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'unix': 'timestamp',
            'open_time': 'timestamp',
            'close_time': 'close_timestamp',
            'vol': 'volume',
            'quote_volume': 'quote_volume',
            'quote_asset_volume': 'quote_volume',
            'taker_buy_base_vol': 'taker_buy_volume',
            'taker_buy_quote_vol': 'taker_buy_quote_volume',
            'number_of_trades': 'trades_count',
            'ignore': 'ignore'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Parse timestamp
        df = self._parse_timestamp(df)
        
        # Validate required columns
        missing_cols = set(self.OHLCV_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert OHLCV to float
        for col in self.OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic validation
        self._validate_data(df)
        
        return df
    
    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to datetime."""
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found")
        
        ts_col = df['timestamp']
        
        # Check if timestamp is already datetime
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            return df
        
        # Try parsing as Unix timestamp (milliseconds)
        if pd.api.types.is_numeric_dtype(ts_col):
            # Check if milliseconds (13 digits) or seconds (10 digits)
            sample_value = ts_col.iloc[0]
            if sample_value > 1e12:  # Milliseconds
                df['timestamp'] = pd.to_datetime(ts_col, unit='ms', utc=True)
            else:  # Seconds
                df['timestamp'] = pd.to_datetime(ts_col, unit='s', utc=True)
        else:
            # Try parsing as string datetime
            df['timestamp'] = pd.to_datetime(ts_col, utc=True)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Perform basic data validation."""
        # Check for negative values in OHLCV
        for col in self.OHLCV_COLUMNS:
            if (df[col] < 0).any():
                logger.warning(f"Negative values found in {col}")
        
        # Check OHLC relationship (high >= low)
        invalid_candles = df['high'] < df['low']
        if invalid_candles.any():
            logger.warning(f"Found {invalid_candles.sum()} candles where high < low")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            logger.warning("Duplicate timestamps found")
    
    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_dt]
        
        return df
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data."""
        if symbol:
            self._data_cache.pop(symbol, None)
        else:
            self._data_cache.clear()
    
    def get_data_info(self, symbol: str) -> Dict:
        """Get information about loaded data."""
        df = self.load_symbol(symbol)
        
        return {
            'symbol': symbol,
            'rows': len(df),
            'start_date': df.index.min().isoformat(),
            'end_date': df.index.max().isoformat(),
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict()
        }


# Convenience functions for direct use

def load_ohlcv_file(
    file_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load OHLCV data from a single CSV file.
    
    Args:
        file_path: Path to the CSV file
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with OHLCV data
    """
    loader = DataLoader(str(Path(file_path).parent))
    symbol = Path(file_path).stem
    return loader.load_symbol(symbol, start_date, end_date)


def load_multiple_symbols(
    data_dir: str,
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols.
    
    Args:
        data_dir: Directory containing CSV files
        symbols: List of symbols to load
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    loader = DataLoader(data_dir)
    return loader.load_multiple_symbols(symbols, start_date, end_date)
