import React from 'react';
import { useTrading } from '@/contexts/TradingContext';
import { Brain, Lightbulb, Settings, Sparkles, AlertTriangle } from 'lucide-react';

const ExplainabilityPanel: React.FC = () => {
  const { activeStrategy, currentPrice, simulationRunning } = useTrading();

  // Mock ML adjustments (simulated)
  const mlAdjustments = [
    {
      parameter: 'EMA Period',
      original: 20,
      adjusted: 25,
      reason: 'Increased to reduce noise during high volatility period',
    },
    {
      parameter: 'Position Size',
      original: '0.01 BTC',
      adjusted: '0.008 BTC',
      reason: 'Reduced due to elevated market risk score',
    },
    {
      parameter: 'Stop Loss',
      original: '5%',
      adjusted: '7%',
      reason: 'Widened to accommodate current volatility regime',
    },
  ];

  // Mock market conditions
  const marketConditions = {
    volatility: 'Medium-High',
    trend: currentPrice > 42000 ? 'Bullish' : 'Bearish',
    momentum: 'Positive',
    riskLevel: 'Moderate',
  };

  if (!activeStrategy) {
    return (
      <div className="glass-card p-4 h-full">
        <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          Explainability
        </h3>
        <div className="flex flex-col items-center justify-center h-32 text-center">
          <AlertTriangle className="w-8 h-8 text-muted-foreground mb-2" />
          <p className="text-muted-foreground text-sm">No active strategy</p>
          <p className="text-muted-foreground text-xs">Select a strategy to view explanations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-4 h-full overflow-auto">
      <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5 text-primary" />
        Explainability Panel
      </h3>

      <div className="space-y-4">
        {/* Active Trading Rule */}
        <div className="p-3 rounded-lg bg-secondary/30 border border-border/50">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-4 h-4 text-primary" />
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Active Trading Rule</span>
          </div>
          <p className="font-mono text-sm text-foreground bg-background/50 p-2 rounded">
            {activeStrategy.rule}
          </p>
        </div>

        {/* Current Parameters */}
        <div className="p-3 rounded-lg bg-secondary/30 border border-border/50">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="w-4 h-4 text-warning" />
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Current Parameters</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Trading Pair:</span>
              <span className="font-mono text-foreground">{activeStrategy.pair}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Current Price:</span>
              <span className="font-mono text-primary">${currentPrice.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Simulation:</span>
              <span className={`font-mono ${simulationRunning ? 'text-success' : 'text-muted-foreground'}`}>
                {simulationRunning ? 'Running' : 'Paused'}
              </span>
            </div>
          </div>
        </div>

        {/* Market Conditions */}
        <div className="p-3 rounded-lg bg-secondary/30 border border-border/50">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-accent" />
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Market Conditions</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="p-2 rounded bg-background/30">
              <span className="text-muted-foreground text-xs block">Volatility</span>
              <span className="font-medium text-warning">{marketConditions.volatility}</span>
            </div>
            <div className="p-2 rounded bg-background/30">
              <span className="text-muted-foreground text-xs block">Trend</span>
              <span className={`font-medium ${marketConditions.trend === 'Bullish' ? 'text-success' : 'text-destructive'}`}>
                {marketConditions.trend}
              </span>
            </div>
            <div className="p-2 rounded bg-background/30">
              <span className="text-muted-foreground text-xs block">Momentum</span>
              <span className="font-medium text-success">{marketConditions.momentum}</span>
            </div>
            <div className="p-2 rounded bg-background/30">
              <span className="text-muted-foreground text-xs block">Risk</span>
              <span className="font-medium text-warning">{marketConditions.riskLevel}</span>
            </div>
          </div>
        </div>

        {/* ML Adjustments */}
        <div className="p-3 rounded-lg bg-gradient-to-r from-chart-2/10 to-primary/10 border border-chart-2/20">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-chart-2" />
            <span className="text-xs text-muted-foreground uppercase tracking-wide">ML-Adjusted Parameters</span>
          </div>
          <div className="space-y-3">
            {mlAdjustments.map((adj, index) => (
              <div key={index} className="p-2 rounded bg-background/30 text-sm">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-foreground">{adj.parameter}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground line-through">{adj.original}</span>
                    <span className="text-primary font-mono">→ {adj.adjusted}</span>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground italic">"{adj.reason}"</p>
              </div>
            ))}
          </div>
        </div>

        {/* Explanation Summary */}
        <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
          <p className="text-sm text-foreground leading-relaxed">
            <span className="text-primary font-medium">Summary: </span>
            The ML optimizer has adjusted parameters to account for current market conditions. 
            EMA period increased to reduce false signals during elevated volatility, 
            while position size and stop-loss have been modified to manage risk appropriately.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExplainabilityPanel;
