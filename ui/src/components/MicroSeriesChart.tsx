import React from 'react';

interface MicroPoint {
  batch_idx: number;
  loss?: number;
  accuracy?: number;
}

interface MicroSeriesChartProps {
  epoch: number;
  role: 'teacher' | 'student';
  phase: 'train' | 'eval';
  points: MicroPoint[];
  height?: number;
}

// Simple SVG line renderer (no external deps) for streaming micro-series
export const MicroSeriesChart: React.FC<MicroSeriesChartProps> = ({ epoch, role, phase, points, height = 160 }) => {
  if (!points.length) return <div className="text-xs text-text-muted">No data yet</div>;
  const losses = points.filter(p => typeof p.loss === 'number');
  const accs = points.filter(p => typeof p.accuracy === 'number');
  const maxBatch = Math.max(...points.map(p => p.batch_idx));
  const maxLoss = losses.length ? Math.max(...(losses.map(p => p.loss as number))) : 1;
  const minLoss = losses.length ? Math.min(...(losses.map(p => p.loss as number))) : 0;
  const maxAcc = accs.length ? Math.max(...(accs.map(p => (p.accuracy || 0)))) : 1;

  const lossPath = losses.map(p => {
    const x = (p.batch_idx / maxBatch) * 100;
    const y = 100 - ((p.loss! - minLoss) / (maxLoss - minLoss + 1e-9)) * 100;
    return `${x},${y}`;
  }).join(' ');

  const accPath = accs.map(p => {
    const x = (p.batch_idx / maxBatch) * 100;
    const y = 100 - ((p.accuracy! / maxAcc) * 100);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">{role} {phase} epoch {epoch}</span>
        <span className="text-text-muted">batches: {maxBatch}</span>
      </div>
      <div className="relative w-full" style={{ height }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full rounded-md bg-bg-tertiary">
          <polyline points={lossPath} fill="none" stroke="#5B9BD5" strokeWidth={1.2} strokeLinejoin="round" strokeLinecap="round" />
          {accPath && (
            <polyline points={accPath} fill="none" stroke="#2ca02c" strokeWidth={1.2} strokeLinejoin="round" strokeLinecap="round" />
          )}
        </svg>
        <div className="absolute top-1 left-2 text-[10px] text-text-muted">loss↓</div>
        {accPath && <div className="absolute top-1 right-2 text-[10px] text-text-muted">acc↑</div>}
      </div>
    </div>
  );
};
