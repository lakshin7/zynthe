import { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Download, TrendingUp, TrendingDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, Button, StatusBadge, ProgressBar } from '../components/base';

interface TrainingMetrics {
  step: number;
  loss: number;
  accuracy: number;
  learning_rate: number;
  timestamp: string;
}

interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix?: number[][];
}

export function TrainingMonitor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [experiment, setExperiment] = useState<any>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationMetrics | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch experiment details
  useEffect(() => {
    if (!id) return;

    const fetchExperiment = async () => {
      try {
        const response = await fetch(`http://localhost:8765/api/experiments/${id}`);
        const data = await response.json();
        setExperiment(data);
        
        // Set initial logs
        if (data.logs) {
          setLogs(data.logs);
        }
      } catch (error) {
        console.error('Failed to fetch experiment:', error);
      }
    };

    fetchExperiment();

    // Setup WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected for training monitor');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.experiment_id !== id) return;

      if (data.type === 'training_metrics') {
        // Add new metric point
        setMetrics(prev => [...prev, {
          step: data.step,
          loss: data.loss,
          accuracy: data.accuracy,
          learning_rate: data.learning_rate,
          timestamp: new Date().toISOString()
        }]);

        // Update evaluation metrics if provided
        if (data.evaluation) {
          setEvaluation(data.evaluation);
        }
      } else if (data.type === 'training_log') {
        setLogs(prev => [...prev, data.message]);
      } else if (data.type === 'training_update') {
        // Refresh experiment data
        fetchExperiment();
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [id]);

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  if (!experiment) {
    return (
      <div className="h-full flex items-center justify-center bg-bg-primary">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-text-secondary">Loading experiment...</p>
        </div>
      </div>
    );
  }

  const currentStep = metrics.length > 0 ? metrics[metrics.length - 1].step : 0;
  const totalSteps = 1000; // This should come from config
  const progress = (currentStep / totalSteps) * 100;

  const currentMetrics = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const bestLoss = metrics.length > 0 ? Math.min(...metrics.map(m => m.loss)) : 0;
  const bestAccuracy = metrics.length > 0 ? Math.max(...metrics.map(m => m.accuracy)) : 0;

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="bg-white border-b border-border-light px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/projects')}
              className="p-2 hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-text-secondary" />
            </button>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-text-primary">{experiment.name}</h1>
                <StatusBadge status={experiment.is_active ? 'running' : 'completed'} pulse={experiment.is_active}>
                  {experiment.is_active ? 'Training' : 'Completed'}
                </StatusBadge>
              </div>
              <p className="text-text-secondary text-sm mt-1">
                Step {currentStep} / {totalSteps} • {Math.round(progress)}%
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" icon={<Download className="w-4 h-4" />}>
              Export Model
            </Button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-4">
          <ProgressBar progress={progress} status="running" animated />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto custom-scrollbar">
        <div className="p-8 space-y-6">
          {/* Metrics Overview */}
          <div className="grid grid-cols-4 gap-4">
            <Card padding="md">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-text-secondary">Current Loss</p>
                <TrendingDown className="w-4 h-4 text-accent" />
              </div>
              <p className="text-3xl font-bold text-text-primary">
                {currentMetrics?.loss.toFixed(4) || '0.0000'}
              </p>
              <p className="text-xs text-text-muted mt-1">
                Best: {bestLoss.toFixed(4)}
              </p>
            </Card>

            <Card padding="md">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-text-secondary">Current Accuracy</p>
                <TrendingUp className="w-4 h-4 text-accent" />
              </div>
              <p className="text-3xl font-bold text-accent">
                {currentMetrics ? (currentMetrics.accuracy * 100).toFixed(1) : '0.0'}%
              </p>
              <p className="text-xs text-text-muted mt-1">
                Best: {(bestAccuracy * 100).toFixed(1)}%
              </p>
            </Card>

            <Card padding="md">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-text-secondary">Learning Rate</p>
              </div>
              <p className="text-2xl font-bold text-text-primary">
                {currentMetrics?.learning_rate.toExponential(2) || '0'}
              </p>
              <p className="text-xs text-text-muted mt-1">
                Adaptive
              </p>
            </Card>

            <Card padding="md">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-text-secondary">ETA</p>
              </div>
              <p className="text-2xl font-bold text-primary">
                {progress > 0 ? `${Math.round((100 - progress) * 0.5)}` : '--'} min
              </p>
              <p className="text-xs text-text-muted mt-1">
                Estimated
              </p>
            </Card>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            {/* Loss Chart */}
            <Card padding="lg">
              <h3 className="text-lg font-bold text-text-primary mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E7EF" />
                  <XAxis
                    dataKey="step"
                    stroke="#90A4AE"
                    tick={{ fill: '#546E7A', fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#90A4AE"
                    tick={{ fill: '#546E7A', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      border: '1px solid #E0E7EF',
                      borderRadius: '8px',
                      padding: '12px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#5B9BD5"
                    strokeWidth={2}
                    dot={false}
                    animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Accuracy Chart */}
            <Card padding="lg">
              <h3 className="text-lg font-bold text-text-primary mb-4">Training Accuracy</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E7EF" />
                  <XAxis
                    dataKey="step"
                    stroke="#90A4AE"
                    tick={{ fill: '#546E7A', fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#90A4AE"
                    tick={{ fill: '#546E7A', fontSize: 12 }}
                    domain={[0, 1]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      border: '1px solid #E0E7EF',
                      borderRadius: '8px',
                      padding: '12px'
                    }}
                    formatter={(value: any) => `${(value * 100).toFixed(1)}%`}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#7FD4A8"
                    strokeWidth={2}
                    dot={false}
                    animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Evaluation Metrics */}
          {evaluation && (
            <Card padding="lg">
              <h3 className="text-lg font-bold text-text-primary mb-4">Evaluation Metrics</h3>
              <div className="grid grid-cols-4 gap-6">
                <div className="text-center p-4 bg-bg-tertiary rounded-lg">
                  <p className="text-sm text-text-secondary mb-2">Accuracy</p>
                  <p className="text-3xl font-bold text-accent">
                    {(evaluation.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-bg-tertiary rounded-lg">
                  <p className="text-sm text-text-secondary mb-2">Precision</p>
                  <p className="text-3xl font-bold text-primary">
                    {(evaluation.precision * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-bg-tertiary rounded-lg">
                  <p className="text-sm text-text-secondary mb-2">Recall</p>
                  <p className="text-3xl font-bold text-warning">
                    {(evaluation.recall * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-bg-tertiary rounded-lg">
                  <p className="text-sm text-text-secondary mb-2">F1 Score</p>
                  <p className="text-3xl font-bold text-text-primary">
                    {(evaluation.f1 * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Confusion Matrix (if available) */}
              {evaluation.confusion_matrix && (
                <div className="mt-6">
                  <h4 className="text-sm font-semibold text-text-primary mb-3">Confusion Matrix</h4>
                  <div className="grid grid-cols-2 gap-2 max-w-md">
                    {evaluation.confusion_matrix.map((row, i) =>
                      row.map((value, j) => (
                        <div
                          key={`${i}-${j}`}
                          className="p-4 text-center rounded-lg border border-border-light"
                          style={{
                            backgroundColor: `rgba(91, 155, 213, ${value / Math.max(...evaluation.confusion_matrix!.flat())})`,
                          }}
                        >
                          <p className="text-2xl font-bold text-white drop-shadow-md">{value}</p>
                          <p className="text-xs text-white/80 mt-1">
                            {i === j ? 'Correct' : 'Incorrect'}
                          </p>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* Live Logs */}
          <Card padding="lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-text-primary">Live Logs</h3>
              <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="rounded border-border-medium focus:ring-2 focus:ring-primary/20"
                />
                Auto-scroll
              </label>
            </div>
            
            <div className="bg-bg-tertiary rounded-lg p-4 h-64 overflow-auto custom-scrollbar font-mono text-sm">
              {logs.length === 0 ? (
                <p className="text-text-muted">Waiting for training logs...</p>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} className="text-text-secondary mb-1">
                    <span className="text-text-muted mr-2">[{new Date().toLocaleTimeString()}]</span>
                    {log}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
