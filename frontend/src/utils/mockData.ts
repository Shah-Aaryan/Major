// Generate realistic OHLCV crypto data
export interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export const generateMockOHLCV = (
  startPrice: number = 42000,
  periods: number = 100,
  volatility: number = 0.02
): OHLCVData[] => {
  const data: OHLCVData[] = [];
  let currentPrice = startPrice;
  const now = Date.now();
  const intervalMs = 60 * 1000; // 1 minute

  for (let i = 0; i < periods; i++) {
    const change = (Math.random() - 0.5) * 2 * volatility * currentPrice;
    const open = currentPrice;
    const close = currentPrice + change;
    const high = Math.max(open, close) + Math.random() * volatility * currentPrice * 0.5;
    const low = Math.min(open, close) - Math.random() * volatility * currentPrice * 0.5;
    const volume = Math.floor(Math.random() * 1000 + 100);

    data.push({
      time: now - (periods - i) * intervalMs,
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume,
    });

    currentPrice = close;
  }

  return data;
};

export const generateNextCandle = (lastCandle: OHLCVData, volatility: number = 0.015): OHLCVData => {
  const change = (Math.random() - 0.5) * 2 * volatility * lastCandle.close;
  const open = lastCandle.close;
  const close = open + change;
  const high = Math.max(open, close) + Math.random() * volatility * open * 0.3;
  const low = Math.min(open, close) - Math.random() * volatility * open * 0.3;
  const volume = Math.floor(Math.random() * 1000 + 100);

  return {
    time: lastCandle.time + 60 * 1000,
    open: Number(open.toFixed(2)),
    high: Number(high.toFixed(2)),
    low: Number(low.toFixed(2)),
    close: Number(close.toFixed(2)),
    volume,
  };
};

// Calculate EMA
export const calculateEMA = (data: number[], period: number): number[] => {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  
  // First EMA is just the SMA
  let sum = 0;
  for (let i = 0; i < period && i < data.length; i++) {
    sum += data[i];
  }
  ema[period - 1] = sum / period;
  
  // Calculate EMA for the rest
  for (let i = period; i < data.length; i++) {
    ema[i] = (data[i] - (ema[i - 1] || ema[period - 1])) * multiplier + (ema[i - 1] || ema[period - 1]);
  }
  
  return ema;
};

// Sample trading rules
export const sampleRules = [
  'BUY 0.01 BTC IF PRICE > 42000',
  'SELL 0.01 BTC IF PRICE < 41000',
  'BUY 0.05 BTC IF EMA(20) CROSSES ABOVE EMA(50)',
  'SELL 0.05 BTC IF EMA(20) CROSSES BELOW EMA(50)',
  'BUY 0.1 BTC IF RSI < 30',
  'SELL 0.1 BTC IF RSI > 70',
];

// Common trading pairs
export const tradingPairs = [
  'BTC/USDT',
  'ETH/USDT',
  'SOL/USDT',
  'BNB/USDT',
  'XRP/USDT',
  'ADA/USDT',
  'AVAX/USDT',
  'DOT/USDT',
];
