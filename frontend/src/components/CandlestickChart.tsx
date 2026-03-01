import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";
import { OHLCVData } from "@/utils/mockData";

interface CandlestickChartProps {
  data: OHLCVData[];
  height?: number;
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  height = 400,
}) => {
  const MIN_WINDOW = 15;

  const [windowRange, setWindowRange] = useState(() => {
    const end = data.length;
    const start = Math.max(
      0,
      end - Math.max(MIN_WINDOW, Math.floor(end * 0.6))
    );
    return { start, end };
  });

  const chartData = useMemo(() => {
    return data.map((candle, index) => {
      const isBullish = candle.close >= candle.open;
      return {
        ...candle,
        index,
        timeLabel: new Date(candle.time).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
        // For candlestick visualization
        wickLow: candle.low,
        wickHigh: candle.high,
        bodyLow: Math.min(candle.open, candle.close),
        bodyHigh: Math.max(candle.open, candle.close),
        bodyHeight: Math.abs(candle.close - candle.open) || 0.5,
        isBullish,
      };
    });
  }, [data]);

  const clampWindow = useCallback(
    (start: number, end: number) => {
      const span = Math.max(MIN_WINDOW, Math.min(end - start, data.length));
      const clampedStart = Math.max(0, Math.min(start, data.length - span));
      return { start: clampedStart, end: clampedStart + span };
    },
    [data.length]
  );

  useEffect(() => {
    setWindowRange((prev) => {
      const desiredSpan = Math.max(
        MIN_WINDOW,
        Math.min(prev.end - prev.start, data.length)
      );
      const end = data.length;
      const start = Math.max(0, end - desiredSpan);
      return { start, end };
    });
  }, [data.length]);

  const handleZoom = useCallback(
    (direction: "in" | "out") => {
      setWindowRange((prev) => {
        const span = prev.end - prev.start;
        const delta = Math.max(5, Math.round(span * 0.2));
        const newSpan =
          direction === "in"
            ? Math.max(MIN_WINDOW, span - delta)
            : Math.min(data.length, span + delta);
        const center = prev.start + span / 2;
        const start = Math.round(center - newSpan / 2);
        return clampWindow(start, start + newSpan);
      });
    },
    [clampWindow, data.length]
  );

  const handlePan = useCallback(
    (direction: "left" | "right") => {
      setWindowRange((prev) => {
        const span = prev.end - prev.start;
        const step = Math.max(1, Math.round(span * 0.1));
        const offset = direction === "left" ? -step : step;
        return clampWindow(prev.start + offset, prev.end + offset);
      });
    },
    [clampWindow]
  );

  const handleWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (event.deltaY < 0) {
        handleZoom("in");
      } else {
        handleZoom("out");
      }
    },
    [handleZoom]
  );

  const visibleData = useMemo(() => {
    return chartData.slice(windowRange.start, windowRange.end);
  }, [chartData, windowRange]);

  const priceRange = useMemo(() => {
    if (visibleData.length === 0) return { min: 0, max: 100 };
    const prices = visibleData.flatMap((d) => [d.high, d.low]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const padding = (max - min) * 0.1;
    return { min: min - padding, max: max + padding };
  }, [visibleData]);

  const currentPrice =
    visibleData.length > 0 ? visibleData[visibleData.length - 1].close : 0;
  const priceLabelPosition =
    priceRange.max === priceRange.min
      ? 50
      : Math.min(
          100,
          Math.max(
            0,
            ((priceRange.max - currentPrice) /
              (priceRange.max - priceRange.min)) *
              100
          )
        );

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length > 0) {
      const d = payload[0].payload;
      return (
        <div className="glass-card p-3 border border-border text-sm">
          <p className="text-muted-foreground mb-1">{d.timeLabel}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono">
            <span className="text-muted-foreground">Open:</span>
            <span className="text-foreground">${d.open.toLocaleString()}</span>
            <span className="text-muted-foreground">High:</span>
            <span className="text-success">${d.high.toLocaleString()}</span>
            <span className="text-muted-foreground">Low:</span>
            <span className="text-destructive">${d.low.toLocaleString()}</span>
            <span className="text-muted-foreground">Close:</span>
            <span className={d.isBullish ? "text-success" : "text-destructive"}>
              ${d.close.toLocaleString()}
            </span>
            <span className="text-muted-foreground">Volume:</span>
            <span className="text-foreground">{d.volume}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom candlestick shape
  const CandlestickShape = (props: any) => {
    const { x, width, payload, viewBox } = props;
    if (!payload) return null;

    const { open, close, high, low, isBullish } = payload;
    const chartHeight = viewBox?.height ?? height;
    const chartTop = viewBox?.y ?? 0;
    const priceToY = (price: number) => {
      const range = priceRange.max - priceRange.min || 1;
      return chartTop + ((priceRange.max - price) / range) * chartHeight;
    };

    const candleColor = isBullish
      ? "hsl(var(--bullish))"
      : "hsl(var(--bearish))";
    const candleWidth = Math.max(width * 0.7, 3);
    const wickWidth = 1;

    const openY = priceToY(open);
    const closeY = priceToY(close);
    const highY = priceToY(high);
    const lowY = priceToY(low);

    const bodyTop = Math.min(openY, closeY);
    const bodyBottom = Math.max(openY, closeY);
    const bodyHeight = Math.max(bodyBottom - bodyTop, 1);

    return (
      <g>
        {/* Wick */}
        <line
          x1={x + width / 2}
          y1={highY}
          x2={x + width / 2}
          y2={lowY}
          stroke={candleColor}
          strokeWidth={wickWidth}
        />
        {/* Body */}
        <rect
          x={x + (width - candleWidth) / 2}
          y={bodyTop}
          width={candleWidth}
          height={bodyHeight}
          fill={isBullish ? candleColor : candleColor}
          stroke={candleColor}
          strokeWidth={1}
          rx={1}
        />
      </g>
    );
  };

  if (data.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        No chart data available
      </div>
    );
  }

  return (
    <div className="relative" onWheel={handleWheel}>
      <div className="flex justify-end gap-2 mb-2 text-xs font-medium text-muted-foreground">
        <button
          type="button"
          className="rounded border border-border px-2 py-1 hover:bg-muted"
          onClick={() => handlePan("left")}
        >
          Pan Left
        </button>
        <button
          type="button"
          className="rounded border border-border px-2 py-1 hover:bg-muted"
          onClick={() => handlePan("right")}
        >
          Pan Right
        </button>
        <button
          type="button"
          className="rounded border border-border px-2 py-1 hover:bg-muted"
          onClick={() => handleZoom("in")}
        >
          Zoom +
        </button>
        <button
          type="button"
          className="rounded border border-border px-2 py-1 hover:bg-muted"
          onClick={() => handleZoom("out")}
        >
          Zoom -
        </button>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={visibleData}
          margin={{ top: 20, right: 70, left: 8, bottom: 10 }}
        >
          <XAxis
            dataKey="timeLabel"
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--border))" }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[priceRange.min, priceRange.max]}
            stroke="hsl(var(--muted-foreground))"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
            orientation="right"
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={currentPrice}
            stroke="hsl(var(--primary))"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <Bar
            dataKey="bodyHeight"
            shape={<CandlestickShape />}
            isAnimationActive={false}
          >
            {visibleData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={
                  entry.isBullish
                    ? "hsl(var(--bullish))"
                    : "hsl(var(--bearish))"
                }
              />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>

      {/* Current Price Label */}
      <div
        className="absolute right-0 transform -translate-y-1/2 px-2 py-1 bg-primary text-primary-foreground text-xs font-mono rounded-l-md"
        style={{ top: `${priceLabelPosition}%` }}
      >
        ${currentPrice.toLocaleString()}
      </div>
    </div>
  );
};

export default CandlestickChart;
