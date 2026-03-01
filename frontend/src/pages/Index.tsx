import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Activity, TrendingUp, Brain, Shield, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Index = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const features = [
    {
      icon: TrendingUp,
      title: 'Paper Trading',
      description: 'Practice strategies with simulated trading on real-time market data',
    },
    {
      icon: Brain,
      title: 'Explainable AI',
      description: 'Understand every decision with transparent ML-powered insights',
    },
    {
      icon: Shield,
      title: 'Risk Management',
      description: 'Built-in portfolio tracking and position management tools',
    },
  ];

  return (
    <div className="min-h-screen bg-background trading-grid">
      {/* Ambient Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-accent/5 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-border/50 bg-card/30 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-primary/20 glow-primary">
              <Activity className="w-7 h-7 text-primary" />
            </div>
            <span className="text-2xl font-bold gradient-text">CryptoXplain</span>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate('/login')}>
              Sign In
            </Button>
            <Button onClick={() => navigate('/register')}>
              Get Started
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10">
        <section className="container mx-auto px-4 py-20 md:py-32">
          <div className="max-w-4xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm mb-8">
              <Activity className="w-4 h-4" />
              Real-Time Explainable Algorithmic Trading
            </div>
            
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-foreground mb-6 leading-tight">
              Trade Crypto with
              <span className="gradient-text block mt-2">Transparency & Intelligence</span>
            </h1>
            
            <p className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
              Create human-readable trading strategies, simulate paper trades on decentralized assets, 
              and understand every algorithmic decision with our explainable AI platform.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button size="xl" onClick={() => navigate('/register')} className="glow-primary">
                Start Trading Now
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
              <Button size="xl" variant="outline" onClick={() => navigate('/login')}>
                Access Platform
              </Button>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="container mx-auto px-4 py-20">
          <div className="grid md:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="glass-card p-6 animate-slide-up hover:border-primary/30 transition-all duration-300"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="container mx-auto px-4 py-20">
          <div className="glass-card p-10 md:p-16 text-center glow-primary">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Ready to Build Your Strategy?
            </h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Join our academic trading platform and experience the future of explainable algorithmic trading.
            </p>
            <Button size="xl" onClick={() => navigate('/register')}>
              Create Free Account
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/50 py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>CryptoXplain — Academic MVP for Explainable Algorithmic Trading</p>
          <p className="mt-2">Paper trading only. No real funds involved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
