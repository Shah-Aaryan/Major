import React from 'react';
import { useTrading } from '@/contexts/TradingContext';
import { FileText, Clock, Hash, Tag } from 'lucide-react';

const AuditLog: React.FC = () => {
  const { auditLog } = useTrading();

  const getEventColor = (eventType: string) => {
    switch (eventType) {
      case 'STRATEGY_CREATED':
        return 'text-primary bg-primary/20';
      case 'SIMULATION_STARTED':
        return 'text-success bg-success/20';
      case 'SIMULATION_PAUSED':
        return 'text-warning bg-warning/20';
      case 'SIMULATION_RESET':
        return 'text-muted-foreground bg-muted';
      case 'TRADE_EXECUTED':
        return 'text-chart-2 bg-chart-2/20';
      default:
        return 'text-muted-foreground bg-muted';
    }
  };

  const formatEventType = (eventType: string) => {
    return eventType.replace(/_/g, ' ');
  };

  return (
    <div className="glass-card p-4 h-full flex flex-col">
      <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
        <FileText className="w-5 h-5 text-primary" />
        Audit Log
      </h3>

      {auditLog.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center py-8">
          <FileText className="w-10 h-10 text-muted-foreground mb-3" />
          <p className="text-muted-foreground text-sm">No audit entries yet</p>
          <p className="text-muted-foreground text-xs">Actions will be logged here</p>
        </div>
      ) : (
        <div className="flex-1 overflow-auto space-y-3 pr-2">
          {auditLog.map((entry, index) => (
            <div
              key={entry.id}
              className="p-3 rounded-lg bg-secondary/30 border border-border/50 animate-slide-up"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-2">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${getEventColor(entry.eventType)}`}>
                  {formatEventType(entry.eventType)}
                </span>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </div>
              </div>

              {/* Rule Text */}
              <div className="mb-2">
                <div className="flex items-center gap-1 mb-1">
                  <Tag className="w-3 h-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">Rule/Action:</span>
                </div>
                <p className="font-mono text-xs text-foreground bg-background/50 p-2 rounded truncate">
                  {entry.ruleText}
                </p>
              </div>

              {/* Hash */}
              <div className="flex items-center gap-1">
                <Hash className="w-3 h-3 text-muted-foreground" />
                <span className="font-mono text-xs text-muted-foreground">
                  {entry.hash}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Log Stats */}
      <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
        <span>{auditLog.length} entries logged</span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          Live
        </span>
      </div>
    </div>
  );
};

export default AuditLog;
