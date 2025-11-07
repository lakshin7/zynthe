import { GlassCard } from './GlassCard';
import { KPIChip } from './KPIChip';
import { MetricSparkline } from './MetricSparkline';
import { LogPanel } from './LogPanel';
import { ProcessCanvas } from './ProcessCanvas';
import { TrendingUp, Zap, Target } from 'lucide-react';

const compressionData = [65, 68, 70, 67, 72, 75, 78, 82, 80, 85, 88, 90];
const accuracyData = [94, 94.2, 93.8, 94.5, 94.8, 95, 94.7, 95.2, 95.5, 95.3, 95.8, 96];
const speedData = [120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165];

export function DashboardGrid() {
  return (
    <div className="p-6 space-y-6">
      {/* Top metrics row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Compression Rate */}
        <GlassCard size="small">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-amber-600 dark:text-cyan-400" />
                <span className="text-slate-600 dark:text-slate-300">Compression</span>
              </div>
              <KPIChip status="success">+12%</KPIChip>
            </div>
            <div className="space-y-1">
              <div className="text-slate-900 dark:text-white">90% Reduced</div>
              <MetricSparkline data={compressionData} color="#FB923C" />
            </div>
          </div>
        </GlassCard>

        {/* Accuracy Retention */}
        <GlassCard size="small">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-rose-600 dark:text-fuchsia-400" />
                <span className="text-slate-600 dark:text-slate-300">Accuracy</span>
              </div>
              <KPIChip status="success">96.0%</KPIChip>
            </div>
            <div className="space-y-1">
              <div className="text-slate-900 dark:text-white">Maintained</div>
              <MetricSparkline data={accuracyData} color="#F472B6" />
            </div>
          </div>
        </GlassCard>

        {/* Inference Speed */}
        <GlassCard size="small">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-emerald-600 dark:text-amber-400" />
                <span className="text-slate-600 dark:text-slate-300">Speed</span>
              </div>
              <KPIChip status="info">165ms</KPIChip>
            </div>
            <div className="space-y-1">
              <div className="text-slate-900 dark:text-white">Inference Time</div>
              <MetricSparkline data={speedData} color="#6EE7B7" />
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Main content row */}
      <div className="grid grid-cols-[1fr,400px] gap-4">
        {/* Process Canvas */}
        <GlassCard size="large">
          <ProcessCanvas />
        </GlassCard>

        {/* Metrics & Logs */}
        <GlassCard size="side">
          <LogPanel />
        </GlassCard>
      </div>
    </div>
  );
}
