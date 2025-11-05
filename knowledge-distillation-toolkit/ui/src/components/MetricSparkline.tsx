interface MetricSparklineProps {
  data: number[];
  color: string;
}

export function MetricSparkline({ data, color }: MetricSparklineProps) {
  // Simplified static representation
  return (
    <div className="w-full h-10 flex items-end gap-1">
      {data.map((value, index) => (
        <div
          key={index}
          className="flex-1 rounded-t"
          style={{
            height: `${(value / Math.max(...data)) * 100}%`,
            backgroundColor: color,
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
}
