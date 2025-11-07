import { useState, useEffect } from 'react';
import { X, TrendingDown, CheckCircle2, Activity, Terminal, Download, FileText, Target, Zap } from 'lucide-react';
import { MetricSparkline } from './MetricSparkline';

interface ProjectDetailsModalProps {
  onClose: () => void;
  projectTitle: string;
  projectId: string;
}

interface ExperimentDetails {
  id: string;
  name: string;
  timestamp: string;
  config: any;
  results: any;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    loss?: number;
  };
  stages: {
    [key: string]: {
      status: string;
      progress: number;
      message: string;
      details?: any;
    };
  };
  logs: string[];
  export_files: {
    [key: string]: {
      path: string;
      size: number;
    };
  };
}

const mockLossData = [2.4, 2.1, 1.9, 1.7, 1.5, 1.4, 1.2, 1.1, 1.0, 0.9];
const mockAccuracyData = [65, 68, 72, 75, 78, 81, 84, 86, 88, 90];

const logLevelColors = {
  info: 'text-blue-600 dark:text-blue-400',
  success: 'text-emerald-600 dark:text-emerald-400',
  warning: 'text-amber-600 dark:text-amber-400',
  error: 'text-rose-600 dark:text-rose-400',
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function parseLogLine(line: string) {
  // Try to extract timestamp, level, and message
  const match = line.match(/(\d{2}:\d{2}:\d{2}).*?\[(\w+)\]\s*(.*)/i);
  if (match) {
    return {
      time: match[1],
      level: match[2].toLowerCase(),
      message: match[3]
    };
  }
  // Fallback format
  return {
    time: new Date().toLocaleTimeString('en-US', { hour12: false }),
    level: 'info',
    message: line
  };
}

export function ProjectDetailsModal({ onClose, projectTitle, projectId }: ProjectDetailsModalProps) {
  const [details, setDetails] = useState<ExperimentDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'metrics' | 'logs' | 'config'>('metrics');

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(`http://localhost:8765/api/experiments/${projectId}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch experiment details: ${response.statusText}`);
        }
        
        const data = await response.json();
        setDetails(data);
      } catch (err) {
        console.error('Failed to fetch experiment details:', err);
        setError(err instanceof Error ? err.message : 'Failed to load experiment details');
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [projectId]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-8">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Modal - larger size for details */}
      <div className="relative w-full max-w-5xl h-[80vh] transition-all duration-300">
        {/* Glass card container */}
        <div className="h-full rounded-2xl backdrop-blur-xl bg-white/70 dark:bg-slate-800/70 border border-white/20 dark:border-white/5 shadow-2xl flex flex-col">
          {/* Top refraction highlight */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/40 dark:via-white/10 to-transparent" />

          {/* Header */}
          <div className="p-6 border-b border-white/10 dark:border-white/5 flex items-center justify-between flex-shrink-0">
            <div>
              <h2 className="text-slate-900 dark:text-white">{projectTitle}</h2>
              <p className="text-slate-500 dark:text-slate-400 mt-1">
                Detailed metrics and logs
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/50 dark:hover:bg-slate-700/50 transition-all duration-200 group"
              aria-label="Close modal"
            >
              <X className="w-6 h-6 text-slate-700 dark:text-slate-200 group-hover:text-slate-900 dark:group-hover:text-white transition-colors" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-6">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-3">
                  <div className="w-12 h-12 mx-auto border-4 border-cyan-400/20 border-t-cyan-400 rounded-full animate-spin"></div>
                  <p className="text-slate-500 dark:text-slate-400">Loading experiment details...</p>
                </div>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-3">
                  <p className="text-red-500 dark:text-red-400">{error}</p>
                  <button
                    onClick={onClose}
                    className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
                  >
                    Close
                  </button>
                </div>
              </div>
            ) : details ? (
              <div className="space-y-6">
                {/* Tab Navigation */}
                <div className="flex gap-2 border-b border-white/10 dark:border-white/5">
                  {[
                    { id: 'metrics', label: 'Metrics', icon: Activity },
                    { id: 'logs', label: 'Logs', icon: Terminal },
                    { id: 'config', label: 'Config', icon: FileText },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`
                        flex items-center gap-2 px-4 py-2 transition-all duration-200
                        ${activeTab === tab.id
                          ? 'text-cyan-600 dark:text-cyan-400 border-b-2 border-cyan-500'
                          : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                        }
                      `}
                    >
                      <tab.icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Metrics Tab */}
                {activeTab === 'metrics' && (
                  <div className="space-y-6">
                    {/* Pipeline Stages */}
                    <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                      <h3 className="text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                        <Zap className="w-5 h-5 text-cyan-500" />
                        Pipeline Stages
                      </h3>
                      <div className="grid grid-cols-5 gap-3">
                        {Object.entries(details.stages).map(([name, stage]) => (
                          <div
                            key={name}
                            className={`
                              p-3 rounded-lg border transition-all duration-200
                              ${stage.status === 'completed'
                                ? 'bg-emerald-500/10 dark:bg-emerald-500/5 border-emerald-300/30 dark:border-emerald-400/20'
                                : stage.status === 'running'
                                ? 'bg-cyan-500/10 dark:bg-cyan-500/5 border-cyan-300/30 dark:border-cyan-400/20'
                                : 'bg-white/20 dark:bg-slate-700/20 border-white/20 dark:border-white/10 opacity-50'
                              }
                            `}
                          >
                            <div className="text-slate-900 dark:text-white mb-1 capitalize text-sm">
                              {name}
                            </div>
                            <div className="text-slate-600 dark:text-slate-400 text-xs mb-2">
                              {stage.message || stage.status}
                            </div>
                            {stage.progress > 0 && (
                              <div className="w-full bg-white/20 dark:bg-slate-800/20 rounded-full h-1.5">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    stage.status === 'completed' ? 'bg-emerald-500' : 'bg-cyan-500'
                                  }`}
                                  style={{ width: `${stage.progress}%` }}
                                />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    {details.metrics && Object.keys(details.metrics).length > 0 && (
                      <div className="grid grid-cols-3 gap-4">
                        {details.metrics.accuracy !== undefined && (
                          <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <Target className="w-5 h-5 text-emerald-500 dark:text-cyan-400" />
                              <span className="text-slate-700 dark:text-slate-300">Accuracy</span>
                            </div>
                            <div className="mb-2">
                              <span className="text-2xl text-slate-900 dark:text-white">
                                {(details.metrics.accuracy * 100).toFixed(2)}%
                              </span>
                            </div>
                            <MetricSparkline data={mockAccuracyData} color="#10b981" />
                          </div>
                        )}

                        {details.metrics.loss !== undefined && (
                          <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <TrendingDown className="w-5 h-5 text-rose-500 dark:text-rose-400" />
                              <span className="text-slate-700 dark:text-slate-300">Loss</span>
                            </div>
                            <div className="mb-2">
                              <span className="text-2xl text-slate-900 dark:text-white">
                                {details.metrics.loss.toFixed(3)}
                              </span>
                            </div>
                            <MetricSparkline data={mockLossData} color="#f43f5e" />
                          </div>
                        )}

                        {details.metrics.f1_score !== undefined && (
                          <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <CheckCircle2 className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                              <span className="text-slate-700 dark:text-slate-300">F1 Score</span>
                            </div>
                            <div className="mb-2">
                              <span className="text-2xl text-slate-900 dark:text-white">
                                {(details.metrics.f1_score * 100).toFixed(2)}%
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Export Files */}
                    {Object.keys(details.export_files).length > 0 && (
                      <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                        <h3 className="text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                          <Download className="w-5 h-5 text-violet-500" />
                          Export Models
                        </h3>
                        <div className="space-y-2">
                          {Object.entries(details.export_files).map(([filename, fileInfo]) => (
                            <div
                              key={filename}
                              className="flex items-center justify-between p-3 rounded-lg bg-white/30 dark:bg-slate-800/30 hover:bg-white/50 dark:hover:bg-slate-800/50 transition-colors"
                            >
                              <div className="flex items-center gap-3">
                                <FileText className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                                <div>
                                  <div className="text-slate-900 dark:text-white text-sm">{filename}</div>
                                  <div className="text-slate-500 dark:text-slate-400 text-xs">
                                    {formatBytes(fileInfo.size)}
                                  </div>
                                </div>
                              </div>
                              <button
                                onClick={() => {
                                  window.open(`http://localhost:8765/api/download/${details.id}/${filename}`, '_blank');
                                }}
                                className="px-3 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-700 dark:text-cyan-300 border border-cyan-300/30 transition-all"
                              >
                                <Download className="w-4 h-4" />
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Logs Tab */}
                {activeTab === 'logs' && (
                  <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Terminal className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                      <h3 className="text-slate-900 dark:text-white">Training Logs</h3>
                    </div>
                    <div className="space-y-1 max-h-96 overflow-y-auto font-mono text-sm">
                      {details.logs.length > 0 ? (
                        details.logs.map((logLine, index) => {
                          const parsed = parseLogLine(logLine);
                          return (
                            <div
                              key={index}
                              className="flex gap-3 p-2 rounded bg-white/30 dark:bg-slate-800/30 hover:bg-white/50 dark:hover:bg-slate-800/50 transition-colors duration-200"
                            >
                              <span className="text-slate-500 dark:text-slate-400 flex-shrink-0 text-xs">
                                {parsed.time}
                              </span>
                              <span className={`flex-shrink-0 text-xs ${logLevelColors[parsed.level as keyof typeof logLevelColors] || 'text-slate-500'}`}>
                                [{parsed.level.toUpperCase()}]
                              </span>
                              <span className="text-slate-700 dark:text-slate-300 flex-1 text-xs break-all">
                                {parsed.message}
                              </span>
                            </div>
                          );
                        })
                      ) : (
                        <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                          No logs available
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Config Tab */}
                {activeTab === 'config' && (
                  <div className="rounded-xl backdrop-blur-md bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10 p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <FileText className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                      <h3 className="text-slate-900 dark:text-white">Configuration</h3>
                    </div>
                    <pre className="text-sm text-slate-700 dark:text-slate-300 font-mono overflow-auto max-h-96 bg-white/30 dark:bg-slate-800/30 p-4 rounded-lg">
                      {JSON.stringify(details.config, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ) : null}
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-white/10 dark:border-white/5 flex justify-end gap-3 flex-shrink-0">
            <button
              onClick={onClose}
              className="px-6 py-2.5 rounded-xl bg-cyan-500/20 dark:bg-cyan-500/20 hover:bg-cyan-500/30 dark:hover:bg-cyan-500/30 text-cyan-700 dark:text-cyan-300 border border-cyan-300/30 dark:border-cyan-400/30 transition-all duration-200"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
