"""
CoinGecko API Client.

Provides real-time cryptocurrency data from CoinGecko API
for price tracking, market data, and analysis.

NOTE: This is for RESEARCH purposes only.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CoinGeckoConfig:
    """Configuration for CoinGecko API."""
    api_key: str = None
    base_url: str = "https://api.coingecko.com/api/v3"
    pro_base_url: str = "https://pro-api.coingecko.com/api/v3"
    requests_per_minute: int = 30  # Free tier limit
    timeout: int = 30
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("COINGECKO_API_KEY", "")


class CoinGeckoClient:
    """
    CoinGecko API client for fetching cryptocurrency data.
    
    Supports:
    - Real-time price data
    - Historical OHLCV data
    - Market cap and volume data
    - Coin listings and metadata
    """
    
    def __init__(self, config: CoinGeckoConfig = None):
        """Initialize CoinGecko client."""
        self.config = config or CoinGeckoConfig()
        self._last_request_time = 0
        self._min_request_interval = 60 / self.config.requests_per_minute
        
        # CoinGecko API key handling
        # Demo keys (CG-) can be used via query parameter on free API
        if self.config.api_key and self.config.api_key.startswith("CG-"):
            # Use free API with demo key as query parameter
            self.base_url = self.config.base_url
            self._api_key_param = self.config.api_key
            self._headers = {"Content-Type": "application/json"}
            logger.info("Using CoinGecko API with Demo key")
        else:
            self.base_url = self.config.base_url
            self._api_key_param = None
            self._headers = {"Content-Type": "application/json"}
            logger.info("Using CoinGecko Free API")
        
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        
        # Coin ID mapping for common symbols
        self.symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "XRP": "ripple",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            "SOL": "solana",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "AVAX": "avalanche-2",
            "ATOM": "cosmos",
            "LTC": "litecoin",
            "UNI": "uniswap",
            "SHIB": "shiba-inu",
        }
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited request to the API."""
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key to params if available
        if params is None:
            params = {}
        if self._api_key_param:
            params["x_cg_demo_api_key"] = self._api_key_param
        
        try:
            response = self._session.get(
                url, 
                params=params, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {e}")
            raise
    
    def ping(self) -> bool:
        """Check API connectivity."""
        try:
            result = self._request("ping")
            return "gecko_says" in result
        except Exception:
            return False
    
    def get_coin_id(self, symbol: str) -> str:
        """Get CoinGecko coin ID from symbol."""
        symbol_upper = symbol.upper().replace("USDT", "").replace("USD", "")
        return self.symbol_to_id.get(symbol_upper, symbol_upper.lower())
    
    def get_price(
        self, 
        symbols: List[str], 
        vs_currencies: List[str] = ["usd"],
        include_market_cap: bool = True,
        include_24h_vol: bool = True,
        include_24h_change: bool = True
    ) -> Dict[str, Any]:
        """
        Get current price for multiple coins.
        
        Args:
            symbols: List of coin symbols (e.g., ["BTC", "ETH"])
            vs_currencies: Quote currencies (e.g., ["usd", "eur"])
            include_market_cap: Include market cap data
            include_24h_vol: Include 24h volume
            include_24h_change: Include 24h price change
        
        Returns:
            Dictionary with price data for each coin
        """
        coin_ids = [self.get_coin_id(s) for s in symbols]
        
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
            "include_market_cap": str(include_market_cap).lower(),
            "include_24hr_vol": str(include_24h_vol).lower(),
            "include_24hr_change": str(include_24h_change).lower(),
        }
        
        return self._request("simple/price", params)
    
    def get_coin_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed data for a single coin.
        
        Args:
            symbol: Coin symbol (e.g., "BTC")
        
        Returns:
            Detailed coin data including market data, description, etc.
        """
        coin_id = self.get_coin_id(symbol)
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        return self._request(f"coins/{coin_id}", params)
    
    def get_market_data(
        self, 
        symbols: List[str] = None,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: str = "24h,7d,30d"
    ) -> List[Dict[str, Any]]:
        """
        Get market data for coins.
        
        Args:
            symbols: List of symbols to filter (None for top coins)
            vs_currency: Quote currency
            order: Sort order
            per_page: Results per page
            page: Page number
            sparkline: Include sparkline data
            price_change_percentage: Time periods for price change
        
        Returns:
            List of market data for each coin
        """
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": str(sparkline).lower(),
            "price_change_percentage": price_change_percentage
        }
        
        if symbols:
            coin_ids = [self.get_coin_id(s) for s in symbols]
            params["ids"] = ",".join(coin_ids)
        
        return self._request("coins/markets", params)
    
    def get_ohlc(
        self, 
        symbol: str, 
        vs_currency: str = "usd",
        days: int = 1
    ) -> pd.DataFrame:
        """
        Get OHLC data for a coin.
        
        Note: CoinGecko provides limited OHLC data:
        - 1-2 days: 30-minute candles
        - 3-30 days: 4-hour candles
        - 31+ days: 4-day candles
        
        Args:
            symbol: Coin symbol (e.g., "BTC")
            vs_currency: Quote currency
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
        
        Returns:
            DataFrame with OHLC data
        """
        coin_id = self.get_coin_id(symbol)
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        data = self._request(f"coins/{coin_id}/ohlc", params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def get_historical_data(
        self, 
        symbol: str,
        vs_currency: str = "usd",
        days: int = 30,
        interval: str = "daily"
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Coin symbol
            vs_currency: Quote currency
            days: Number of days
            interval: Data interval ("daily" or granular for <90 days)
        
        Returns:
            DataFrame with prices, market caps, and volumes
        """
        coin_id = self.get_coin_id(symbol)
        params = {
            "vs_currency": vs_currency,
            "days": days,
        }
        
        if interval and days <= 90:
            params["interval"] = interval
        
        data = self._request(f"coins/{coin_id}/market_chart", params)
        
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        
        df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def get_trending(self) -> List[Dict[str, Any]]:
        """Get trending coins in the last 24h."""
        data = self._request("search/trending")
        return data.get("coins", [])
    
    def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency market data."""
        return self._request("global")
    
    def search(self, query: str) -> Dict[str, Any]:
        """Search for coins, categories, and exchanges."""
        params = {"query": query}
        return self._request("search", params)


class CoinGeckoDataFetcher:
    """
    High-level data fetcher for integration with trading research.
    
    Formats CoinGecko data to match the expected format of the
    research pipeline.
    """
    
    def __init__(self):
        """Initialize data fetcher."""
        self.client = CoinGeckoClient()
        self._cache = {}
        self._cache_duration = 60  # Cache for 60 seconds
    
    def get_live_price(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get live price data formatted for research pipeline.
        
        Args:
            symbol: Coin symbol
        
        Returns:
            Dictionary with price and market data
        """
        data = self.client.get_price(
            [symbol],
            include_market_cap=True,
            include_24h_vol=True,
            include_24h_change=True
        )
        
        coin_id = self.client.get_coin_id(symbol)
        if coin_id in data:
            coin_data = data[coin_id]
            return {
                "symbol": symbol,
                "price": coin_data.get("usd", 0),
                "market_cap": coin_data.get("usd_market_cap", 0),
                "volume_24h": coin_data.get("usd_24h_vol", 0),
                "change_24h": coin_data.get("usd_24h_change", 0),
                "timestamp": datetime.now()
            }
        return {}
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple coins."""
        data = self.client.get_price(
            symbols,
            include_market_cap=True,
            include_24h_vol=True,
            include_24h_change=True
        )
        
        result = {}
        for symbol in symbols:
            coin_id = self.client.get_coin_id(symbol)
            if coin_id in data:
                coin_data = data[coin_id]
                result[symbol] = {
                    "price": coin_data.get("usd", 0),
                    "market_cap": coin_data.get("usd_market_cap", 0),
                    "volume_24h": coin_data.get("usd_24h_vol", 0),
                    "change_24h": coin_data.get("usd_24h_change", 0),
                }
        return result
    
    def get_ohlcv_dataframe(
        self, 
        symbol: str = "BTC", 
        days: int = 1
    ) -> pd.DataFrame:
        """
        Get OHLCV data as DataFrame.
        
        Args:
            symbol: Coin symbol
            days: Number of days
        
        Returns:
            DataFrame with OHLCV columns
        """
        return self.client.get_ohlc(symbol, days=days)
    
    def get_market_overview(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get market overview for top coins.
        
        Args:
            top_n: Number of top coins to fetch
        
        Returns:
            DataFrame with market data
        """
        data = self.client.get_market_data(per_page=top_n)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        columns = [
            "symbol", "name", "current_price", "market_cap",
            "total_volume", "price_change_percentage_24h",
            "price_change_percentage_7d_in_currency",
            "price_change_percentage_30d_in_currency"
        ]
        available_cols = [c for c in columns if c in df.columns]
        return df[available_cols]


# Convenience function for quick access
def get_crypto_price(symbol: str = "BTC") -> float:
    """Quick function to get current crypto price."""
    client = CoinGeckoClient()
    data = client.get_price([symbol])
    coin_id = client.get_coin_id(symbol)
    return data.get(coin_id, {}).get("usd", 0)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing CoinGecko Client...")
    
    client = CoinGeckoClient()
    
    # Test ping
    if client.ping():
        print("✓ API connection successful")
    
    # Test price fetching
    prices = client.get_price(["BTC", "ETH"])
    print(f"✓ BTC Price: ${prices.get('bitcoin', {}).get('usd', 'N/A'):,.2f}")
    print(f"✓ ETH Price: ${prices.get('ethereum', {}).get('usd', 'N/A'):,.2f}")
    
    # Test OHLC
    ohlc = client.get_ohlc("BTC", days=1)
    print(f"✓ OHLC Data: {len(ohlc)} candles")
    
    # Test data fetcher
    fetcher = CoinGeckoDataFetcher()
    live_data = fetcher.get_live_price("BTC")
    print(f"✓ Live BTC Data: ${live_data.get('price', 0):,.2f}")
    
    print("\nAll tests passed!")
