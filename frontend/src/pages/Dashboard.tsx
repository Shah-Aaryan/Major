import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTrading } from '@/contexts/TradingContext';
import { Button } from '@/components/ui/button';
import { 
  Activity, 
  Plus, 
  PlayCircle, 
  LogOut, 
  TrendingUp, 
  Trash2,
  FileText,
  Wallet
} from 'lucide-react';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const { strategies, deleteStrategy, setActiveStrategy, portfolio, currentPrice } = useTrading();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleRunSimulation = (strategy: typeof strategies[0]) => {
    setActiveStrategy(strategy);
    navigate('/simulation');
  };

  const totalEquity = portfolio.cash + 
    Object.entries(portfolio.holdings).reduce((acc, [_, amount]) => acc + amount * currentPrice, 0);

  return (
    <div className="min-h-screen bg-background trading-grid">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/20">
              <Activity className="w-6 h-6 text-primary" />
            </div>
            <span className="text-xl font-bold gradient-text">BeyondAlgo</span>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary/50">
              <Wallet className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">Portfolio:</span>
              <span className="font-mono font-semibold text-foreground">
                ${totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="mb-8 animate-fade-in">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Welcome back, <span className="gradient-text">{user?.email}</span>
          </h1>
          <p className="text-muted-foreground">
            Manage your trading strategies and run simulations
          </p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="stat-card animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-sm">Active Strategies</span>
              <FileText className="w-4 h-4 text-primary" />
            </div>
            <p className="text-2xl font-bold text-foreground">{strategies.length}</p>
          </div>
          
          <div className="stat-card animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-sm">Available Cash</span>
              <Wallet className="w-4 h-4 text-success" />
            </div>
            <p className="text-2xl font-bold text-foreground font-mono">
              ${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </p>
          </div>
          
          <div className="stat-card animate-slide-up" style={{ animationDelay: '0.3s' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-sm">BTC Price</span>
              <TrendingUp className="w-4 h-4 text-warning" />
            </div>
            <p className="text-2xl font-bold text-foreground font-mono">
              ${currentPrice.toLocaleString()}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-4 mb-8">
          <Button onClick={() => navigate('/create-strategy')} size="lg" className="glow-primary">
            <Plus className="w-5 h-5 mr-2" />
            Create New Strategy
          </Button>
        </div>

        {/* Strategies List */}
        <div className="glass-card p-6 animate-slide-up">
          <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            Your Trading Strategies
          </h2>

          {strategies.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-secondary flex items-center justify-center">
                <FileText className="w-8 h-8 text-muted-foreground" />
              </div>
              <p className="text-muted-foreground mb-4">No strategies created yet</p>
              <Button onClick={() => navigate('/create-strategy')} variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Create Your First Strategy
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {strategies.map((strategy, index) => (
                <div 
                  key={strategy.id}
                  className="p-4 rounded-lg bg-secondary/50 border border-border hover:border-primary/30 transition-all duration-200 animate-slide-up"
                  style={{ animationDelay: `${0.1 * index}s` }}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-foreground">{strategy.name}</h3>
                        <span className="px-2 py-1 rounded-md bg-primary/20 text-primary text-xs font-mono">
                          {strategy.pair}
                        </span>
                      </div>
                      <p className="text-sm font-mono text-muted-foreground bg-background/50 p-2 rounded-md">
                        {strategy.rule}
                      </p>
                      <p className="text-xs text-muted-foreground mt-2">
                        Created: {new Date(strategy.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <Button 
                        onClick={() => handleRunSimulation(strategy)}
                        variant="success"
                        size="sm"
                      >
                        <PlayCircle className="w-4 h-4 mr-1" />
                        Run Simulation
                      </Button>
                      <Button 
                        onClick={() => deleteStrategy(strategy.id)}
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
