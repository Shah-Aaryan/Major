import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

export interface Strategy {
  id: string;
  name: string;
  rule: string;
  pair: string;
  createdAt: Date;
}

export interface Trade {
  id: string;
  strategyId: string;
  type: 'BUY' | 'SELL';
  amount: number;
  price: number;
  timestamp: Date;
  pnl?: number;
}

export interface AuditEntry {
  id: string;
  timestamp: Date;
  eventType: 'STRATEGY_CREATED' | 'SIMULATION_STARTED' | 'SIMULATION_PAUSED' | 'SIMULATION_RESET' | 'TRADE_EXECUTED';
  ruleText: string;
  hash: string;
}

export interface Portfolio {
  cash: number;
  holdings: { [symbol: string]: number };
  initialCash: number;
}

interface TradingContextType {
  strategies: Strategy[];
  trades: Trade[];
  auditLog: AuditEntry[];
  portfolio: Portfolio;
  currentPrice: number;
  shortSellingEnabled: boolean;
  simulationRunning: boolean;
  activeStrategy: Strategy | null;
  equityHistory: { timestamp: number; userEquity: number; mlEquity: number }[];
  addStrategy: (strategy: Omit<Strategy, 'id' | 'createdAt'>) => void;
  deleteStrategy: (id: string) => void;
  executeTrade: (type: 'BUY' | 'SELL', amount: number, symbol: string) => boolean;
  addAuditEntry: (eventType: AuditEntry['eventType'], ruleText: string) => void;
  toggleShortSelling: () => void;
  setSimulationRunning: (running: boolean) => void;
  setActiveStrategy: (strategy: Strategy | null) => void;
  resetSimulation: () => void;
  updateCurrentPrice: (price: number) => void;
  addEquityPoint: (userEquity: number, mlEquity: number) => void;
}

const TradingContext = createContext<TradingContextType | undefined>(undefined);

const generateHash = () => {
  return '0x' + Array.from({ length: 16 }, () => 
    Math.floor(Math.random() * 16).toString(16)
  ).join('');
};

const INITIAL_CASH = 10000;

export const TradingProvider = ({ children }: { children: ReactNode }) => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio>({
    cash: INITIAL_CASH,
    holdings: {},
    initialCash: INITIAL_CASH,
  });
  const [currentPrice, setCurrentPrice] = useState(42500);
  const [shortSellingEnabled, setShortSellingEnabled] = useState(false);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [activeStrategy, setActiveStrategy] = useState<Strategy | null>(null);
  const [equityHistory, setEquityHistory] = useState<{ timestamp: number; userEquity: number; mlEquity: number }[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    const savedStrategies = localStorage.getItem('trading_strategies');
    if (savedStrategies) {
      setStrategies(JSON.parse(savedStrategies).map((s: Strategy) => ({
        ...s,
        createdAt: new Date(s.createdAt)
      })));
    }
  }, []);

  // Save strategies to localStorage
  useEffect(() => {
    localStorage.setItem('trading_strategies', JSON.stringify(strategies));
  }, [strategies]);

  const addStrategy = (strategy: Omit<Strategy, 'id' | 'createdAt'>) => {
    const newStrategy: Strategy = {
      ...strategy,
      id: `strategy_${Date.now()}`,
      createdAt: new Date(),
    };
    setStrategies(prev => [...prev, newStrategy]);
    addAuditEntry('STRATEGY_CREATED', strategy.rule);
  };

  const deleteStrategy = (id: string) => {
    setStrategies(prev => prev.filter(s => s.id !== id));
  };

  const executeTrade = (type: 'BUY' | 'SELL', amount: number, symbol: string): boolean => {
    const cost = amount * currentPrice;
    
    if (type === 'BUY') {
      if (cost > portfolio.cash) {
        return false;
      }
      setPortfolio(prev => ({
        ...prev,
        cash: prev.cash - cost,
        holdings: {
          ...prev.holdings,
          [symbol]: (prev.holdings[symbol] || 0) + amount,
        },
      }));
    } else {
      const currentHolding = portfolio.holdings[symbol] || 0;
      if (!shortSellingEnabled && amount > currentHolding) {
        return false;
      }
      setPortfolio(prev => ({
        ...prev,
        cash: prev.cash + cost,
        holdings: {
          ...prev.holdings,
          [symbol]: (prev.holdings[symbol] || 0) - amount,
        },
      }));
    }

    const trade: Trade = {
      id: `trade_${Date.now()}`,
      strategyId: activeStrategy?.id || '',
      type,
      amount,
      price: currentPrice,
      timestamp: new Date(),
    };
    setTrades(prev => [...prev, trade]);
    addAuditEntry('TRADE_EXECUTED', `${type} ${amount} ${symbol} @ ${currentPrice}`);
    return true;
  };

  const addAuditEntry = (eventType: AuditEntry['eventType'], ruleText: string) => {
    const entry: AuditEntry = {
      id: `audit_${Date.now()}`,
      timestamp: new Date(),
      eventType,
      ruleText,
      hash: generateHash(),
    };
    setAuditLog(prev => [entry, ...prev].slice(0, 100));
  };

  const toggleShortSelling = () => {
    setShortSellingEnabled(prev => !prev);
  };

  const resetSimulation = () => {
    setPortfolio({
      cash: INITIAL_CASH,
      holdings: {},
      initialCash: INITIAL_CASH,
    });
    setTrades([]);
    setEquityHistory([]);
    setSimulationRunning(false);
    addAuditEntry('SIMULATION_RESET', activeStrategy?.rule || 'N/A');
  };

  const updateCurrentPrice = (price: number) => {
    setCurrentPrice(price);
  };

  const addEquityPoint = (userEquity: number, mlEquity: number) => {
    setEquityHistory(prev => [...prev, {
      timestamp: Date.now(),
      userEquity,
      mlEquity,
    }]);
  };

  return (
    <TradingContext.Provider value={{
      strategies,
      trades,
      auditLog,
      portfolio,
      currentPrice,
      shortSellingEnabled,
      simulationRunning,
      activeStrategy,
      equityHistory,
      addStrategy,
      deleteStrategy,
      executeTrade,
      addAuditEntry,
      toggleShortSelling,
      setSimulationRunning,
      setActiveStrategy,
      resetSimulation,
      updateCurrentPrice,
      addEquityPoint,
    }}>
      {children}
    </TradingContext.Provider>
  );
};

export const useTrading = () => {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTrading must be used within a TradingProvider');
  }
  return context;
};
