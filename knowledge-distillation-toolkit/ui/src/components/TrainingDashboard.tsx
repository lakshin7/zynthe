import { useState, useEffect, useRef } from 'react';
import { X, Activity, BarChart3, FileText, Pause, Play, Square, Save, TrendingDown, TrendingUp, Clock, Zap } from 'lucide-react';

interface TrainingDashboardProps {
  experimentId: string;
  onClose: () => void;
}

interface TrainingMetrics {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  temperature: number;
  stage: string;
  progress: number;
  eta: string;
}

interface LogEntry {
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error' | 'debug';
  message: string;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ experimentId, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'logs'>('overview');
  const [metrics, setMetrics] = useState<TrainingMetrics>({
    epoch: 0,
    totalEpochs: 10,
    loss: 0,
    accuracy: 0,
    learningRate: 0.001,
    temperature: 3.0,
    stage: 'Initializing',
    progress: 0,
    eta: 'Calculating...',
  });
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8765/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Training dashboard WebSocket connected');
      // Subscribe to experiment updates
      ws.send(JSON.stringify({ type: 'subscribe', experiment_id: experimentId }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'training_metrics') {
        setMetrics(data.metrics);
        
        // Update history
        if (data.metrics.loss !== undefined) {
          setLossHistory(prev => [...prev.slice(-49), data.metrics.loss]);
        }
        if (data.metrics.accuracy !== undefined) {
          setAccuracyHistory(prev => [...prev.slice(-49), data.metrics.accuracy]);
        }
      }
      
      if (data.type === 'training_log') {
        setLogs(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          level: data.level || 'info',
          message: data.message,
        }]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('Training dashboard WebSocket disconnected');
    };

    return () => {
      ws.close();
    };
  }, [experimentId]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handlePauseResume = async () => {
    try {
      const endpoint = isPaused ? 'resume' : 'pause';
      const response = await fetch(`http://localhost:8765/api/training/${experimentId}/${endpoint}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        setIsPaused(!isPaused);
        console.log(isPaused ? 'Training resumed' : 'Training paused');
      }
    } catch (error) {
      console.error('Failed to pause/resume training:', error);
    }
  };

  const handleStop = async () => {
    if (confirm('Are you sure you want to stop training? This cannot be undone.')) {
      try {
        const response = await fetch(`http://localhost:8765/api/training/${experimentId}/stop`, {
          method: 'POST',
        });
        
        if (response.ok) {
          console.log('Training stopped');
          onClose();
        }
      } catch (error) {
        console.error('Failed to stop training:', error);
      }
    }
  };

  const handleSaveCheckpoint = async () => {
    try {
      const response = await fetch(`http://localhost:8765/api/training/${experimentId}/checkpoint`, {
        method: 'POST',
      });
      
      if (response.ok) {
        console.log('Checkpoint saved');
        // Show success notification
      }
    } catch (error) {
      console.error('Failed to save checkpoint:', error);
    }
  };

  const getLogColor = (level: string) => {
    switch (level) {
      case 'success': return 'text-emerald-600 dark:text-emerald-400';
      case 'warning': return 'text-amber-600 dark:text-amber-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      case 'debug': return 'text-slate-500 dark:text-slate-500';
      default: return 'text-slate-700 dark:text-slate-300';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Dashboard */}
      <div className="relative w-full max-w-6xl h-[90vh] transition-all duration-300">
        <div className="h-full rounded-2xl backdrop-blur-xl bg-white/70 dark:bg-slate-800/70 border border-white/20 dark:border-white/5 shadow-2xl flex flex-col">
          {/* Top refraction highlight */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/40 dark:via-white/10 to-transparent" />

          {/* Header */}
          <div className="p-6 border-b border-white/10 dark:border-white/5 flex items-center justify-between flex-shrink-0">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-white flex items-center gap-3">
                <Activity className="w-7 h-7 text-cyan-500" />
                Live Training Dashboard
              </h2>
              <p className="text-slate-500 dark:text-slate-400 mt-1 text-sm">
                Experiment ID: {experimentId}
              </p>
            </div>
            <div className="flex items-center gap-3">
              {/* Control Buttons */}
              <button
                onClick={handlePauseResume}
                className="px-4 py-2 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-600 dark:text-amber-400 border border-amber-500/30 transition-all duration-200 flex items-center gap-2"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPaused ? 'Resume' : 'Pause'}
              </button>
              <button
                onClick={handleSaveCheckpoint}
                className="px-4 py-2 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 border border-cyan-500/30 transition-all duration-200 flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                Save
              </button>
              <button
                onClick={handleStop}
                className="px-4 py-2 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 border border-red-500/30 transition-all duration-200 flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Stop
              </button>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-white/50 dark:hover:bg-slate-700/50 transition-all duration-200 group"
                aria-label="Close dashboard"
              >
                <X className="w-6 h-6 text-slate-700 dark:text-slate-200 group-hover:text-slate-900 dark:group-hover:text-white transition-colors" />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="px-6 pt-4 border-b border-white/10 dark:border-white/5 flex-shrink-0">
            <div className="flex gap-2">
              {[
                { id: 'overview', label: 'Overview', icon: Activity },
                { id: 'metrics', label: 'Metrics', icon: BarChart3 },
                { id: 'logs', label: 'Logs', icon: FileText },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    px-4 py-2 rounded-t-xl transition-all duration-200 flex items-center gap-2
                    ${activeTab === tab.id
                      ? 'bg-white/50 dark:bg-slate-700/50 text-slate-900 dark:text-white border-t border-x border-white/30 dark:border-white/20'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }
                  `}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto px-6 py-4 min-h-0">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-4">
                {/* Progress Bar */}
                <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Training Progress</h3>
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Epoch {metrics.epoch}/{metrics.totalEpochs}
                    </span>
                  </div>
                  <div className="mb-2">
                    <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300 rounded-full"
                        style={{ width: `${metrics.progress}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">{metrics.progress.toFixed(1)}% Complete</span>
                    <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      ETA: {metrics.eta}
                    </span>
                  </div>
                </div>

                {/* Current Stage */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-400/30">
                  <div className="flex items-center gap-3">
                    <Zap className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
                    <div>
                      <div className="text-sm text-slate-600 dark:text-slate-400">Current Stage</div>
                      <div className="text-lg font-semibold text-slate-900 dark:text-white">{metrics.stage}</div>
                    </div>
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-600 dark:text-slate-400">Loss</span>
                      <TrendingDown className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {metrics.loss.toFixed(4)}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-600 dark:text-slate-400">Accuracy</span>
                      <TrendingUp className="w-4 h-4 text-cyan-500" />
                    </div>
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {(metrics.accuracy * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-600 dark:text-slate-400">Learning Rate</span>
                    </div>
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {metrics.learningRate.toFixed(6)}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-600 dark:text-slate-400">Temperature</span>
                    </div>
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {metrics.temperature.toFixed(1)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Metrics Tab */}
            {activeTab === 'metrics' && (
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Loss Over Time</h3>
                  <div className="h-48 flex items-end gap-1">
                    {lossHistory.length === 0 ? (
                      <div className="w-full h-full flex items-center justify-center text-slate-500 dark:text-slate-400">
                        No data yet
                      </div>
                    ) : (
                      lossHistory.map((loss, i) => {
                        const maxLoss = Math.max(...lossHistory);
                        const height = (loss / maxLoss) * 100;
                        return (
                          <div
                            key={i}
                            className="flex-1 bg-gradient-to-t from-red-500 to-orange-500 rounded-t"
                            style={{ height: `${height}%`, minHeight: '2px' }}
                            title={`Loss: ${loss.toFixed(4)}`}
                          />
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Accuracy Over Time</h3>
                  <div className="h-48 flex items-end gap-1">
                    {accuracyHistory.length === 0 ? (
                      <div className="w-full h-full flex items-center justify-center text-slate-500 dark:text-slate-400">
                        No data yet
                      </div>
                    ) : (
                      accuracyHistory.map((acc, i) => {
                        const height = acc * 100;
                        return (
                          <div
                            key={i}
                            className="flex-1 bg-gradient-to-t from-cyan-500 to-blue-500 rounded-t"
                            style={{ height: `${height}%`, minHeight: '2px' }}
                            title={`Accuracy: ${(acc * 100).toFixed(2)}%`}
                          />
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Logs Tab */}
            {activeTab === 'logs' && (
              <div className="h-full">
                <div className="h-full p-4 rounded-xl bg-slate-900/90 dark:bg-slate-950/90 border border-white/10 overflow-y-auto font-mono text-xs">
                  {logs.length === 0 ? (
                    <div className="text-slate-500 text-center py-8">No logs yet</div>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} className="flex gap-3 py-1 hover:bg-white/5">
                        <span className="text-slate-500 flex-shrink-0">{log.timestamp}</span>
                        <span className={`flex-shrink-0 uppercase font-semibold ${getLogColor(log.level)}`}>
                          [{log.level}]
                        </span>
                        <span className="text-slate-300 flex-1">{log.message}</span>
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
