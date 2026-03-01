import React from 'react';
import { useTrading } from '@/contexts/TradingContext';
import { Wallet, TrendingUp, TrendingDown, Minus, DollarSign, Coins } from 'lucide-react';

const PortfolioPanel: React.FC = () => {
  const { portfolio, currentPrice, shortSellingEnabled } = useTrading();

  const btcHoldings = portfolio.holdings['BTC'] || 0;
  const holdingsValue = btcHoldings * currentPrice;
  const totalEquity = portfolio.cash + holdingsValue;
  
  // Calculate PnL
  const unrealizedPnL = holdingsValue;
  const realizedPnL = portfolio.cash - portfolio.initialCash + (btcHoldings < 0 ? Math.abs(btcHoldings) * currentPrice : 0);
  const totalPnL = totalEquity - portfolio.initialCash;
  const percentageReturn = ((totalEquity - portfolio.initialCash) / portfolio.initialCash) * 100;

  // Position status
  const getPositionStatus = () => {
    if (btcHoldings > 0) return { label: 'LONG', color: 'text-success', bgColor: 'bg-success/20' };
    if (btcHoldings < 0) return { label: 'SHORT', color: 'text-destructive', bgColor: 'bg-destructive/20' };
    return { label: 'FLAT', color: 'text-muted-foreground', bgColor: 'bg-muted' };
  };

  const position = getPositionStatus();

  const StatCard = ({ 
    label, 
    value, 
    icon: Icon, 
    valueColor = 'text-foreground',
    subValue 
  }: { 
    label: string; 
    value: string; 
    icon: React.ElementType;
    valueColor?: string;
    subValue?: string;
  }) => (
    <div className="p-3 rounded-lg bg-secondary/30 border border-border/50">
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground uppercase tracking-wide">{label}</span>
      </div>
      <p className={`text-lg font-mono font-semibold ${valueColor}`}>{value}</p>
      {subValue && <p className="text-xs text-muted-foreground mt-1">{subValue}</p>}
    </div>
  );

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
        <Wallet className="w-5 h-5 text-primary" />
        Portfolio & PnL
      </h3>

      <div className="space-y-4">
        {/* Position Status */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-border/50">
          <span className="text-sm text-muted-foreground">Position Status</span>
          <span className={`px-3 py-1 rounded-full text-xs font-bold ${position.bgColor} ${position.color}`}>
            {position.label}
          </span>
        </div>

        {/* Portfolio Holdings */}
        <div className="grid grid-cols-2 gap-3">
          <StatCard
            label="Cash (USDT)"
            value={`$${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
            icon={DollarSign}
          />
          <StatCard
            label="BTC Holdings"
            value={btcHoldings.toFixed(6)}
            icon={Coins}
            valueColor={btcHoldings < 0 ? 'text-destructive' : btcHoldings > 0 ? 'text-success' : 'text-foreground'}
            subValue={`≈ $${Math.abs(holdingsValue).toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
          />
        </div>

        {/* Market Price */}
        <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">BTC Market Price</span>
            <span className="text-lg font-mono font-bold text-primary">
              ${currentPrice.toLocaleString()}
            </span>
          </div>
        </div>

        {/* PnL Metrics */}
        <div className="space-y-3 pt-3 border-t border-border">
          <h4 className="text-sm font-medium text-muted-foreground">PnL Metrics</h4>
          
          <div className="grid grid-cols-2 gap-3">
            <StatCard
              label="Unrealized PnL"
              value={`$${unrealizedPnL.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
              icon={btcHoldings >= 0 ? TrendingUp : TrendingDown}
              valueColor={unrealizedPnL >= 0 ? 'text-success' : 'text-destructive'}
            />
            <StatCard
              label="Realized PnL"
              value={`${realizedPnL >= 0 ? '+' : ''}$${realizedPnL.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
              icon={realizedPnL >= 0 ? TrendingUp : TrendingDown}
              valueColor={realizedPnL >= 0 ? 'text-success' : 'text-destructive'}
            />
          </div>

          <div className="p-4 rounded-lg bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Total Equity</span>
              <span className="text-xl font-mono font-bold text-foreground">
                ${totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Return</span>
              <span className={`text-sm font-mono font-semibold ${percentageReturn >= 0 ? 'text-success' : 'text-destructive'}`}>
                {percentageReturn >= 0 ? '+' : ''}{percentageReturn.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        {/* Short Selling Status */}
        {shortSellingEnabled && (
          <div className="p-2 rounded-lg bg-warning/10 border border-warning/20 text-center">
            <span className="text-xs text-warning font-medium">⚠️ Short Selling Enabled</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PortfolioPanel;
