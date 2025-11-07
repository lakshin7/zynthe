import { useState, useEffect } from 'react';
import { Droplets, TrendingDown, Target, Clock, Zap, Activity } from 'lucide-react';
import { GlassCard } from './GlassCard';

interface DistillationMetrics {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  temperature: number;
  progress: number;
  eta: string;
}

export function DistillationPage() {
  const [metrics, setMetrics] = useState<DistillationMetrics>({
    epoch: 0,
    totalEpochs: 10,
    loss: 0,
    accuracy: 0,
    learningRate: 0.001,
    temperature: 3.0,
    progress: 0,
    eta: 'Calculating...',
  });
  const [isActive, setIsActive] = useState(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);

  useEffect(() => {
    // Listen to WebSocket for distillation updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'training_metrics') {
        setMetrics(data.metrics);
        setIsActive(true);
        
        // Update history
        if (data.metrics.loss !== undefined) {
          setLossHistory(prev => [...prev.slice(-49), data.metrics.loss]);
        }
        if (data.metrics.accuracy !== undefined) {
          setAccuracyHistory(prev => [...prev.slice(-49), data.metrics.accuracy]);
        }
      } else if (data.type === 'training_update' && data.status === 'completed') {
        setIsActive(false);
      }
    };
    
    return () => ws.close();
  }, []);

  const renderSparkline = (data: number[], color: string) => {
    if (data.length < 2) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <svg className="w-full h-16" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    );
  };

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <Droplets className="w-8 h-8 text-amber-600 dark:text-cyan-400" />
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
            Knowledge Distillation
          </h1>
          {isActive && (
            <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/30">
              <Activity className="w-4 h-4 text-emerald-600 dark:text-emerald-400 animate-pulse" />
              <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">Training</span>
            </span>
          )}
        </div>
        <p className="text-slate-600 dark:text-slate-300">
          Training student model to mimic teacher's knowledge
        </p>
      </div>

      {/* Progress Overview */}
      <GlassCard>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                Epoch {metrics.epoch} / {metrics.totalEpochs}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                {metrics.progress.toFixed(1)}% Complete • ETA: {metrics.eta}
              </div>
            </div>
            <Clock className="w-8 h-8 text-amber-600 dark:text-cyan-400" />
          </div>
          
          {/* Progress bar */}
          <div className="relative h-3 bg-white/30 dark:bg-slate-700/30 rounded-full overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-amber-500 to-rose-500 dark:from-cyan-500 dark:to-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${metrics.progress}%` }}
            />
          </div>
        </div>
      </GlassCard>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Loss */}
        <GlassCard>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-rose-500" />
                <h3 className="font-semibold text-slate-900 dark:text-white">Training Loss</h3>
              </div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {metrics.loss.toFixed(4)}
              </div>
            </div>
            {lossHistory.length > 1 && (
              <div className="bg-white/20 dark:bg-slate-800/20 rounded-lg p-2">
                {renderSparkline(lossHistory, '#f43f5e')}
              </div>
            )}
          </div>
        </GlassCard>

        {/* Accuracy */}
        <GlassCard>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-emerald-500" />
                <h3 className="font-semibold text-slate-900 dark:text-white">Accuracy</h3>
              </div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">
                {(metrics.accuracy * 100).toFixed(2)}%
              </div>
            </div>
            {accuracyHistory.length > 1 && (
              <div className="bg-white/20 dark:bg-slate-800/20 rounded-lg p-2">
                {renderSparkline(accuracyHistory, '#10b981')}
              </div>
            )}
          </div>
        </GlassCard>

        {/* Learning Rate */}
        <GlassCard>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-500" />
              <h3 className="font-semibold text-slate-900 dark:text-white">Learning Rate</h3>
            </div>
            <div className="text-xl font-bold text-slate-900 dark:text-white">
              {metrics.learningRate.toFixed(4)}
            </div>
          </div>
        </GlassCard>

        {/* Temperature */}
        <GlassCard>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-500" />
              <h3 className="font-semibold text-slate-900 dark:text-white">Temperature</h3>
            </div>
            <div className="text-xl font-bold text-slate-900 dark:text-white">
              {metrics.temperature.toFixed(1)}
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Status Message */}
      {!isActive && (
        <GlassCard>
          <div className="text-center py-8">
            <Droplets className="w-12 h-12 text-slate-400 dark:text-slate-500 mx-auto mb-3" />
            <p className="text-slate-600 dark:text-slate-300">
              No active distillation training. Start a new training run to see live metrics.
            </p>
          </div>
        </GlassCard>
      )}
    </div>
  );
}
