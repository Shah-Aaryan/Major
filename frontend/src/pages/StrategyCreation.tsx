import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTrading } from '@/contexts/TradingContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Activity, 
  ArrowLeft, 
  Save, 
  Lightbulb,
  FileCode,
  Coins
} from 'lucide-react';
import { sampleRules, tradingPairs } from '@/utils/mockData';

const StrategyCreation = () => {
  const [name, setName] = useState('');
  const [rule, setRule] = useState('');
  const [pair, setPair] = useState('BTC/USDT');
  const [error, setError] = useState('');
  const { addStrategy } = useTrading();
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name.trim()) {
      setError('Please enter a strategy name.');
      return;
    }

    if (!rule.trim()) {
      setError('Please enter a trading rule.');
      return;
    }

    addStrategy({ name, rule, pair });
    navigate('/dashboard');
  };

  const insertSampleRule = (sampleRule: string) => {
    setRule(sampleRule);
  };

  return (
    <div className="min-h-screen bg-background trading-grid">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard')}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div className="p-2 rounded-lg bg-primary/20">
              <Activity className="w-6 h-6 text-primary" />
            </div>
            <span className="text-xl font-bold gradient-text">CryptoXplain</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center gap-3">
            <FileCode className="w-8 h-8 text-primary" />
            Create Trading Strategy
          </h1>
          <p className="text-muted-foreground">
            Define human-readable trading rules for your algorithmic strategy
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Form */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="glass-card p-6 animate-slide-up">
              <div className="space-y-6">
                {/* Strategy Name */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground flex items-center gap-2">
                    Strategy Name
                  </label>
                  <Input
                    type="text"
                    placeholder="e.g., BTC Momentum Strategy"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>

                {/* Trading Pair */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground flex items-center gap-2">
                    <Coins className="w-4 h-4 text-primary" />
                    Trading Pair
                  </label>
                  <select
                    value={pair}
                    onChange={(e) => setPair(e.target.value)}
                    className="flex h-10 w-full rounded-lg border border-border bg-input px-4 py-2 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  >
                    {tradingPairs.map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>

                {/* Trading Rule */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground flex items-center gap-2">
                    <FileCode className="w-4 h-4 text-primary" />
                    Crypto Trading Rule (Human-Readable)
                  </label>
                  <textarea
                    value={rule}
                    onChange={(e) => setRule(e.target.value)}
                    placeholder={`Enter your trading rule in plain text, e.g.:\n\nBUY 0.01 BTC IF PRICE > 30000\nSELL 0.01 BTC IF EMA(50) CROSSES BELOW EMA(200)`}
                    className="flex min-h-[200px] w-full rounded-lg border border-border bg-input px-4 py-3 text-base font-mono text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary resize-none"
                  />
                  <p className="text-xs text-muted-foreground">
                    Write complete textual trading queries. No numeric input fields – express all logic as readable text.
                  </p>
                </div>

                {error && (
                  <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                    {error}
                  </div>
                )}

                <div className="flex gap-4">
                  <Button type="submit" size="lg" className="flex-1">
                    <Save className="w-5 h-5 mr-2" />
                    Save Strategy
                  </Button>
                  <Button 
                    type="button" 
                    variant="outline" 
                    size="lg"
                    onClick={() => navigate('/dashboard')}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            </form>
          </div>

          {/* Sample Rules Sidebar */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-warning" />
                Example Rules
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Click to insert a sample rule as a starting point:
              </p>
              <div className="space-y-3">
                {sampleRules.map((sampleRule, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => insertSampleRule(sampleRule)}
                    className="w-full text-left p-3 rounded-lg bg-secondary/50 border border-border hover:border-primary/50 hover:bg-secondary transition-all duration-200 text-xs font-mono text-muted-foreground hover:text-foreground"
                  >
                    {sampleRule}
                  </button>
                ))}
              </div>
            </div>

            <div className="glass-card p-6 mt-4 animate-slide-up" style={{ animationDelay: '0.2s' }}>
              <h3 className="font-semibold text-foreground mb-3">Rule Syntax Guide</h3>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p><code className="text-primary">BUY/SELL</code> - Trade direction</p>
                <p><code className="text-primary">[amount]</code> - Quantity to trade</p>
                <p><code className="text-primary">IF</code> - Condition trigger</p>
                <p><code className="text-primary">PRICE</code> - Current market price</p>
                <p><code className="text-primary">EMA(n)</code> - Exponential MA</p>
                <p><code className="text-primary">CROSSES</code> - Crossover signal</p>
                <p><code className="text-primary">RSI</code> - Relative Strength Index</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default StrategyCreation;
