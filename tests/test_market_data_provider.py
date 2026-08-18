"""Unit tests for the market_data provider layer."""

from pathlib import Path

import pandas as pd
import pytest

from config.settings import MarketDataConfig, load_market_data_config
from market_data.dataset_provider import DatasetProvider
from market_data.factory import create_provider
from market_data.provider import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    normalize_candle_frame,
    validate_candle_schema,
)
from market_data.rest_client import BinanceRESTClient

FIXTURE_CSV = str(Path(__file__).parent / "fixtures" / "sample_ohlcv.csv")


class TestSchema:
    def test_validate_candle_schema_passes_with_required_columns(self):
        df = pd.DataFrame({c: [1] for c in REQUIRED_COLUMNS})
        validate_candle_schema(df)  # should not raise

    def test_validate_candle_schema_raises_when_missing_columns(self):
        df = pd.DataFrame({"open": [1], "close": [1]})
        with pytest.raises(ValueError):
            validate_candle_schema(df)

    def test_normalize_candle_frame_fills_optional_columns(self):
        df = pd.DataFrame({c: [1, 2] for c in REQUIRED_COLUMNS})
        out = normalize_candle_frame(df)
        for col in OPTIONAL_COLUMNS:
            assert col in out.columns
        assert list(out.columns) == REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    def test_normalize_candle_frame_sorts_by_timestamp(self):
        df = pd.DataFrame({
            "timestamp": [2, 1, 3],
            "open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1],
            "close": [1, 1, 1], "volume": [1, 1, 1],
        })
        out = normalize_candle_frame(df)
        assert list(out["timestamp"]) == [1, 2, 3]


class TestDatasetProvider:
    def test_initialize_loads_all_rows(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider.initialize()
        assert not provider.is_finished()

    def test_get_next_candle_returns_rows_in_chronological_order(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider.initialize()

        closes = []
        while not provider.is_finished():
            candle = provider.get_next_candle()
            closes.append(float(candle["close"].iloc[0]))

        assert closes == sorted(closes)
        assert len(closes) == 10  # matches fixture row count

    def test_get_next_candle_returns_none_when_exhausted(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider.initialize()
        while not provider.is_finished():
            provider.get_next_candle()
        assert provider.get_next_candle() is None

    def test_rolling_window_never_exceeds_window_size(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=3)
        provider.initialize()

        for _ in range(10):
            provider.get_next_candle()
            assert len(provider.get_latest_window()) <= 3

    def test_window_has_zero_future_leakage(self):
        """The rolling window must never contain a candle not yet emitted."""
        provider = DatasetProvider(FIXTURE_CSV, window_size=10)
        provider.initialize()

        provider.get_next_candle()
        provider.get_next_candle()
        window = provider.get_latest_window()

        assert len(window) == 2
        full_df = provider._df
        assert window["close"].iloc[-1] == full_df["close"].iloc[1]

    def test_reset_rewinds_cursor_and_clears_window(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider.initialize()
        provider.get_next_candle()
        provider.get_next_candle()

        provider.reset()

        assert not provider.is_finished()
        assert len(provider.get_latest_window()) == 0

    def test_deterministic_replay_across_two_runs(self):
        provider_a = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider_a.initialize()
        closes_a = []
        while not provider_a.is_finished():
            closes_a.append(float(provider_a.get_next_candle()["close"].iloc[0]))

        provider_b = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider_b.initialize()
        closes_b = []
        while not provider_b.is_finished():
            closes_b.append(float(provider_b.get_next_candle()["close"].iloc[0]))

        assert closes_a == closes_b

    def test_output_schema_matches_shared_provider_schema(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        provider.initialize()
        candle = provider.get_next_candle()
        assert list(candle.columns) == REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    def test_get_next_candle_before_initialize_raises(self):
        provider = DatasetProvider(FIXTURE_CSV, window_size=5)
        with pytest.raises(RuntimeError):
            provider.get_next_candle()

    def test_rejects_non_positive_window_size(self):
        with pytest.raises(ValueError):
            DatasetProvider(FIXTURE_CSV, window_size=0)


class TestBinanceRESTClientParsing:
    def test_parse_klines_maps_all_schema_fields(self):
        raw_row = [
            1609459200000, "28900.0", "28950.0", "28880.0", "28920.0", "12.5",
            1609459259999, "361500.0", 42, "6.0", "173700.0", "0",
        ]
        df = BinanceRESTClient._parse_klines([raw_row])

        assert len(df) == 1
        row = df.iloc[0]
        assert row["open"] == 28900.0
        assert row["high"] == 28950.0
        assert row["low"] == 28880.0
        assert row["close"] == 28920.0
        assert row["volume"] == 12.5
        assert row["quote_volume"] == 361500.0
        assert row["number_of_trades"] == 42
        assert row["taker_buy_base"] == 6.0
        assert row["taker_buy_quote"] == 173700.0


class TestMarketDataConfig:
    def test_rejects_invalid_data_source(self):
        with pytest.raises(ValueError):
            MarketDataConfig(DATA_SOURCE="bogus")

    def test_rejects_non_positive_window_size(self):
        with pytest.raises(ValueError):
            MarketDataConfig(WINDOW_SIZE=0)

    def test_from_dict_ignores_unknown_keys(self):
        cfg = MarketDataConfig.from_dict({"SYMBOL": "ETHUSDT", "NOT_A_FIELD": 123})
        assert cfg.SYMBOL == "ETHUSDT"

    def test_load_market_data_config_reads_committed_json(self):
        cfg = load_market_data_config("config/beyondalgo.json")
        assert cfg.DATA_SOURCE in ("dataset", "live")
        assert cfg.WINDOW_SIZE == 500


class TestProviderFactory:
    def test_creates_dataset_provider_for_dataset_source(self):
        cfg = MarketDataConfig(DATA_SOURCE="dataset", CSV_PATH=FIXTURE_CSV, WINDOW_SIZE=5)
        provider = create_provider(cfg)
        assert isinstance(provider, DatasetProvider)

    def test_creates_binance_live_provider_for_live_source(self):
        from market_data.binance_provider import BinanceLiveProvider

        cfg = MarketDataConfig(DATA_SOURCE="live", CSV_PATH=FIXTURE_CSV, WINDOW_SIZE=5)
        provider = create_provider(cfg)
        assert isinstance(provider, BinanceLiveProvider)

    def test_dataset_and_live_providers_expose_identical_interface(self):
        from market_data.binance_provider import BinanceLiveProvider
        from market_data.provider import MarketDataProvider

        required_methods = {
            "initialize", "get_next_candle", "get_latest_window",
            "reset", "is_finished", "shutdown",
        }
        assert required_methods <= set(dir(DatasetProvider))
        assert required_methods <= set(dir(BinanceLiveProvider))
        assert issubclass(DatasetProvider, MarketDataProvider)
        assert issubclass(BinanceLiveProvider, MarketDataProvider)
