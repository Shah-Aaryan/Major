import React, { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useTrading } from "@/contexts/TradingContext";
import { Button } from "@/components/ui/button";
import CandlestickChart from "@/components/CandlestickChart";
import PortfolioPanel from "@/components/PortfolioPanel";
import ExplainabilityPanel from "@/components/ExplainabilityPanel";
import AuditLog from "@/components/AuditLog";
import EquityCurve from "@/components/EquityCurve";
import {
  generateMockOHLCV,
  generateNextCandle,
  OHLCVData,
} from "@/utils/mockData";
import {
  Activity,
  ArrowLeft,
  Play,
  Pause,
  RotateCcw,
  Clock,
  AlertCircle,
  ToggleLeft,
  ToggleRight,
  ShoppingCart,
  TrendingDown,
} from "lucide-react";

const TradingSimulation = () => {
  const navigate = useNavigate();
  const {
    activeStrategy,
    simulationRunning,
    setSimulationRunning,
    shortSellingEnabled,
    toggleShortSelling,
    resetSimulation,
    addAuditEntry,
    executeTrade,
    portfolio,
    currentPrice,
    updateCurrentPrice,
    addEquityPoint,
  } = useTrading();

  const [ohlcvData, setOhlcvData] = useState<OHLCVData[]>([]);
  const [timeframe, setTimeframe] = useState("1m");
  const [tradeWarning, setTradeWarning] = useState<string | null>(null);
  const [manualTradeAmount, setManualTradeAmount] = useState("0.01");

  // Initialize chart data
  useEffect(() => {
    const initialData = generateMockOHLCV(42000, 50, 0.015);
    setOhlcvData(initialData);
    if (initialData.length > 0) {
      updateCurrentPrice(initialData[initialData.length - 1].close);
    }
  }, []);

  // Simulation loop
  useEffect(() => {
    if (!simulationRunning || ohlcvData.length === 0) return;

    const interval = setInterval(() => {
      setOhlcvData((prev) => {
        const lastCandle = prev[prev.length - 1];
        const newCandle = generateNextCandle(lastCandle, 0.012);
        updateCurrentPrice(newCandle.close);

        // Update equity history
        const btcHoldings = portfolio.holdings["BTC"] || 0;
        const userEquity = portfolio.cash + btcHoldings * newCandle.close;
        // Simulated ML strategy (slightly better performance)
        const mlEquity =
          10000 + (userEquity - 10000) * 1.15 + Math.random() * 50;
        addEquityPoint(userEquity, mlEquity);

        const newData = [...prev.slice(-99), newCandle];
        return newData;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [
    simulationRunning,
    ohlcvData.length,
    portfolio,
    addEquityPoint,
    updateCurrentPrice,
  ]);

  const handleStart = useCallback(() => {
    setSimulationRunning(true);
    addAuditEntry(
      "SIMULATION_STARTED",
      activeStrategy?.rule || "Manual trading"
    );
  }, [setSimulationRunning, addAuditEntry, activeStrategy]);

  const handlePause = useCallback(() => {
    setSimulationRunning(false);
    addAuditEntry(
      "SIMULATION_PAUSED",
      activeStrategy?.rule || "Manual trading"
    );
  }, [setSimulationRunning, addAuditEntry, activeStrategy]);

  const handleReset = useCallback(() => {
    resetSimulation();
    const initialData = generateMockOHLCV(42000, 50, 0.015);
    setOhlcvData(initialData);
    if (initialData.length > 0) {
      updateCurrentPrice(initialData[initialData.length - 1].close);
    }
  }, [resetSimulation, updateCurrentPrice]);

  const handleManualTrade = useCallback(
    (type: "BUY" | "SELL") => {
      const amount = parseFloat(manualTradeAmount);
      if (isNaN(amount) || amount <= 0) {
        setTradeWarning("Invalid trade amount");
        setTimeout(() => setTradeWarning(null), 3000);
        return;
      }

      const success = executeTrade(type, amount, "BTC");
      if (!success) {
        if (type === "SELL" && !shortSellingEnabled) {
          setTradeWarning("Insufficient crypto balance to execute SELL order");
        } else if (type === "BUY") {
          setTradeWarning("Insufficient cash to execute BUY order");
        }
        setTimeout(() => setTradeWarning(null), 3000);
      }
    },
    [manualTradeAmount, executeTrade, shortSellingEnabled]
  );

  const timeframes = ["1m", "5m", "15m"];

  return (
    <div className="min-h-screen bg-background trading-grid">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/dashboard")}
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div className="p-2 rounded-lg bg-primary/20">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <span className="text-lg font-bold gradient-text">
                Trading Simulation
              </span>
              {activeStrategy && (
                <p className="text-xs text-muted-foreground">
                  {activeStrategy.name}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Simulation Status */}
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                simulationRunning
                  ? "bg-success/20 text-success"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  simulationRunning
                    ? "bg-success animate-pulse"
                    : "bg-muted-foreground"
                }`}
              />
              {simulationRunning ? "Live" : "Paused"}
            </div>

            {/* Short Selling Toggle */}
            <button
              onClick={toggleShortSelling}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary hover:bg-secondary/80 transition-colors"
            >
              {shortSellingEnabled ? (
                <ToggleRight className="w-5 h-5 text-warning" />
              ) : (
                <ToggleLeft className="w-5 h-5 text-muted-foreground" />
              )}
              <span className="text-sm">Short Selling</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-4">
        {/* Trade Warning */}
        {tradeWarning && (
          <div className="mb-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20 flex items-center gap-2 animate-slide-up">
            <AlertCircle className="w-5 h-5 text-destructive" />
            <span className="text-destructive text-sm">{tradeWarning}</span>
          </div>
        )}

        <div className="grid lg:grid-cols-12 gap-4 items-start">
          {/* Left Column - Chart & Controls */}
          <div className="lg:col-span-8 space-y-4">
            {/* Chart Card */}
            <div className="glass-card p-4 animate-fade-in">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4">
                  <h3 className="font-semibold text-foreground">
                    {activeStrategy?.pair || "BTC/USDT"}
                  </h3>
                  <span className="font-mono text-xl font-bold text-primary">
                    ${currentPrice.toLocaleString()}
                  </span>
                </div>

                {/* Timeframe Selector */}
                <div className="flex items-center gap-1 bg-secondary/50 rounded-lg p-1">
                  {timeframes.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setTimeframe(tf)}
                      className={`px-3 py-1 rounded-md text-sm font-medium transition-all ${
                        timeframe === tf
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              <CandlestickChart data={ohlcvData} height={350} />
            </div>

            {/* Controls */}
            <div className="glass-card p-4 animate-slide-up">
              <div className="flex flex-wrap items-center gap-4">
                {/* Simulation Controls */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground mr-2">
                    Simulation:
                  </span>
                  {!simulationRunning ? (
                    <Button onClick={handleStart} variant="success" size="sm">
                      <Play className="w-4 h-4 mr-1" />
                      Start
                    </Button>
                  ) : (
                    <Button onClick={handlePause} variant="outline" size="sm">
                      <Pause className="w-4 h-4 mr-1" />
                      Pause
                    </Button>
                  )}
                  <Button onClick={handleReset} variant="ghost" size="sm">
                    <RotateCcw className="w-4 h-4 mr-1" />
                    Reset
                  </Button>
                </div>

                <div className="h-8 w-px bg-border" />

                {/* Manual Trading */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Trade:</span>
                  <input
                    type="number"
                    value={manualTradeAmount}
                    onChange={(e) => setManualTradeAmount(e.target.value)}
                    className="w-24 h-9 px-3 rounded-lg bg-input border border-border text-sm font-mono"
                    step="0.01"
                    min="0.001"
                  />
                  <span className="text-sm text-muted-foreground">BTC</span>
                  <Button
                    onClick={() => handleManualTrade("BUY")}
                    variant="success"
                    size="sm"
                  >
                    <ShoppingCart className="w-4 h-4 mr-1" />
                    Buy
                  </Button>
                  <Button
                    onClick={() => handleManualTrade("SELL")}
                    variant="destructive"
                    size="sm"
                  >
                    <TrendingDown className="w-4 h-4 mr-1" />
                    Sell
                  </Button>
                </div>
              </div>
            </div>

            {/* Equity Curve */}
            <EquityCurve />

            {/* Audit Log on left column */}
            <AuditLog />
          </div>

          {/* Right Column - Panels */}
          <div className="lg:col-span-4 space-y-4 lg:sticky lg:top-4 self-start">
            <PortfolioPanel />
            <ExplainabilityPanel />
          </div>
        </div>
      </main>
    </div>
  );
};

export default TradingSimulation;
