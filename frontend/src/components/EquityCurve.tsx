import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { useTrading } from "@/contexts/TradingContext";
import { TrendingUp } from "lucide-react";

const EquityCurve: React.FC = () => {
  const { equityHistory } = useTrading();

  const baseUser = equityHistory[0]?.userEquity || 10000;
  const baseML = equityHistory[0]?.mlEquity || 10000;

  const chartData = equityHistory.map((point, index) => {
    const userReturn = ((point.userEquity - baseUser) / baseUser) * 100;
    const mlReturn = ((point.mlEquity - baseML) / baseML) * 100;

    return {
      time: index,
      timeLabel: new Date(point.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
      userEquity: point.userEquity,
      mlEquity: point.mlEquity,
      userReturn,
      mlReturn,
    };
  });

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length > 0) {
      return (
        <div className="glass-card p-3 border border-border text-sm">
          <p className="text-muted-foreground mb-2">
            {payload[0].payload.timeLabel}
          </p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-primary" />
              <span className="text-muted-foreground">Your Strategy:</span>
              <span className="font-mono text-foreground">
                {payload[0].value?.toFixed(2)}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-chart-2" />
              <span className="text-muted-foreground">ML-Optimized:</span>
              <span className="font-mono text-foreground">
                {payload[1]?.value?.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  if (equityHistory.length === 0) {
    return (
      <div className="glass-card p-4 h-full">
        <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          Performance Comparison
        </h3>
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <TrendingUp className="w-10 h-10 text-muted-foreground mb-3" />
          <p className="text-muted-foreground text-sm">
            No performance data yet
          </p>
          <p className="text-muted-foreground text-xs">
            Start simulation to track equity
          </p>
        </div>
      </div>
    );
  }

  const latestUser = chartData[chartData.length - 1]?.userReturn || 0;
  const latestML = chartData[chartData.length - 1]?.mlReturn || 0;
  const userReturn = latestUser.toFixed(2);
  const mlReturn = latestML.toFixed(2);

  const returns = chartData.flatMap((d) => [d.userReturn, d.mlReturn]);
  const minReturn = Math.min(...returns);
  const maxReturn = Math.max(...returns);
  const padding = Math.max(0.5, (maxReturn - minReturn) * 0.1);
  const domain = [minReturn - padding, maxReturn + padding];

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-primary" />
        Performance Comparison
      </h3>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
          <span className="text-xs text-muted-foreground block mb-1">
            Your Strategy
          </span>
          <span className="font-mono text-lg font-semibold text-primary">
            {Number(userReturn) >= 0 ? "+" : ""}
            {userReturn}%
          </span>
        </div>
        <div className="p-3 rounded-lg bg-chart-2/10 border border-chart-2/20">
          <span className="text-xs text-muted-foreground block mb-1">
            ML-Optimized
          </span>
          <span className="font-mono text-lg font-semibold text-chart-2">
            {Number(mlReturn) >= 0 ? "+" : ""}
            {mlReturn}%
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
        >
          <XAxis
            dataKey="timeLabel"
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--border))" }}
          />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--border))" }}
            domain={domain}
            tickFormatter={(value) => `${value.toFixed(1)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px" }}
            formatter={(value) => (
              <span className="text-muted-foreground">{value}</span>
            )}
          />
          <Line
            type="monotone"
            dataKey="userReturn"
            name="Your Strategy"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "hsl(var(--primary))" }}
          />
          <Line
            type="monotone"
            dataKey="mlReturn"
            name="ML-Optimized"
            stroke="hsl(var(--chart-2))"
            strokeWidth={2}
            dot={false}
            strokeDasharray="5 5"
            activeDot={{ r: 4, fill: "hsl(var(--chart-2))" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EquityCurve;
