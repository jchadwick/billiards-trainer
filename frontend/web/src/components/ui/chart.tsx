import * as React from "react";
import { ResponsiveContainer } from "recharts";
import { cn } from "../../lib/utils";

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
}

export function ChartContainer({
  label,
  series,
  className,
  children,
}: {
  label: string;
  series: ChartSeries[];
  className?: string;
  children: React.ReactElement;
}) {
  return (
    <div className={cn("chart-container", className)}>
      <div className="chart-legend" aria-label={`${label} legend`}>
        {series.map((item) => (
          <span className="chart-legend-item" key={item.key}>
            <span
              aria-hidden="true"
              className="chart-legend-swatch"
              style={{ backgroundColor: item.color }}
            />
            {item.label}
          </span>
        ))}
      </div>
      <div className="chart-canvas" role="img" aria-label={label}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface TooltipEntry {
  color?: string;
  dataKey?: string | number;
  name?: string | number;
  value?: unknown;
}

export function ChartTooltipContent({
  active,
  payload,
  label,
  series,
  formatValue = (value) => String(value),
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
  series: ChartSeries[];
  formatValue?: (value: unknown) => string;
}) {
  if (!active || !payload?.length) return null;
  const sampledAt = label == null ? null : new Date(label);
  return (
    <div className="chart-tooltip">
      {sampledAt && Number.isFinite(sampledAt.getTime()) && (
        <p className="chart-tooltip-time">{sampledAt.toLocaleTimeString()}</p>
      )}
      {payload.map((entry) => {
        const seriesItem = series.find(
          (item) => item.key === String(entry.dataKey ?? entry.name),
        );
        if (entry.value == null) return null;
        return (
          <p className="chart-tooltip-row" key={String(entry.dataKey ?? entry.name)}>
            <span
              aria-hidden="true"
              className="chart-legend-swatch"
              style={{ backgroundColor: entry.color ?? seriesItem?.color }}
            />
            <span>{seriesItem?.label ?? entry.name}</span>
            <strong>{formatValue(entry.value)}</strong>
          </p>
        );
      })}
    </div>
  );
}
