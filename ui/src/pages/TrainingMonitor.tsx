import { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Download, TrendingUp, TrendingDown, Activity, BarChart3, CheckCircle2, Loader2, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, Button, StatusBadge, ProgressBar } from '../components/base';
import { ConfusionMatrixCard } from '../components/ConfusionMatrixCard';
import { fetchArtifacts, fetchConfusion } from '../api/zynthe-api';
import { EvaluationMonitor } from '../components/EvaluationMonitor';
import useNotifications from '../hooks/useNotifications';
import NotificationSettings from '../components/NotificationSettings';

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

interface PipelineStage {
  name: string;
  displayName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime?: string;
  endTime?: string;
  message?: string;
}

export function TrainingMonitor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotifications();
  const [activeTab, setActiveTab] = useState<'training' | 'evaluation'>('training');
  const [experiment, setExperiment] = useState<any>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationMetrics | null>(null);
  const [teacherCM, setTeacherCM] = useState<{ image_path?: string; metrics?: Record<string, number> } | null>(null);
  const [studentCM, setStudentCM] = useState<{ image_path?: string; metrics?: Record<string, number> } | null>(null);
  const [artifactImages, setArtifactImages] = useState<string[]>([]);
  // const [batchRows, setBatchRows] = useState<Record<string, string>[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [currentStage, setCurrentStage] = useState<string>('Initializing');
  const [wsConnected, setWsConnected] = useState(false);
  const [previousStatus, setPreviousStatus] = useState<string>('');
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>([
    { name: 'downloading_teacher', displayName: 'Downloading Teacher Model', status: 'pending', progress: 0 },
    { name: 'downloading_student', displayName: 'Downloading Student Model', status: 'pending', progress: 0 },
    { name: 'loading_data', displayName: 'Loading Dataset', status: 'pending', progress: 0 },
    { name: 'initializing', displayName: 'Initializing', status: 'pending', progress: 0 },
    { name: 'preflight', displayName: 'Preflight Check', status: 'pending', progress: 0 },
    { name: 'training', displayName: 'Training', status: 'pending', progress: 0 },
    { name: 'evaluation', displayName: 'Evaluation', status: 'pending', progress: 0 },
  ]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch experiment details
  useEffect(() => {
    if (!id) return;

    const fetchExperiment = async () => {
      try {
        console.log(`Fetching experiment: ${id}`);
        const response = await fetch(`http://localhost:8765/api/experiments/${id}`);
        
        if (!response.ok) {
          console.error(`Failed to fetch experiment: ${response.status} ${response.statusText}`);
          return;
        }
        
        const data = await response.json();
        console.log('Experiment data:', data);
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

    // Try to prefetch artifacts periodically during run
    const poll = setInterval(async () => {
      try {
        if (!id) return;
        const art = await fetchArtifacts(id);
        setArtifactImages(art.images || []);
      } catch {}
    }, 5000);

    // Setup WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected for training monitor');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message received:', data);
      
      // Check if message is for this experiment (handle both experiment_id and experimentId)
      const msgExpId = data.experiment_id || data.experimentId;
      if (msgExpId && msgExpId !== id) {
        console.log(`Ignoring message for experiment ${msgExpId}, current is ${id}`);
        return;
      }

      if (data.type === 'training_metrics') {
        console.log('Updating metrics:', data.metrics);
        
        // Handle the metrics object structure from backend
        const metricsData = data.metrics || data;
        setMetrics(prev => [...prev, {
          step: metricsData.epoch || data.step || prev.length,
          loss: metricsData.loss || 0,
          accuracy: metricsData.accuracy || 0,
          learning_rate: metricsData.learningRate || 0.001,
          timestamp: new Date().toISOString()
        }]);

        // Update current stage
        if (metricsData.stage) {
          setCurrentStage(metricsData.stage);
        }

        // Update evaluation metrics if provided
        if (data.evaluation) {
          setEvaluation(data.evaluation);
        }
      } else if (data.type === 'training_log') {
        console.log('Adding log:', data.message);
        setLogs(prev => [...prev, data.message]);
      } else if (data.type === 'training_update') {
        console.log('Training update:', data);
        
        // Check if training completed
        if (data.status === 'completed' && previousStatus !== 'completed') {
          setPreviousStatus('completed');
          showNotification({
            title: '🎉 Training Complete!',
            body: `Experiment "${experiment?.name || id}" has finished training successfully.`,
            playSound: true,
          });
        } else if (data.status === 'failed' && previousStatus !== 'failed') {
          setPreviousStatus('failed');
          showNotification({
            title: '❌ Training Failed',
            body: `Experiment "${experiment?.name || id}" encountered an error.`,
            playSound: true,
          });
        }
        
        // Update current stage
        if (data.stage) {
          setCurrentStage(data.stage);
          
          // Update pipeline stages
          setPipelineStages(prev => prev.map(stage => {
            if (stage.name === data.stage.toLowerCase()) {
              return { ...stage, status: 'running', progress: data.progress || 0, message: data.message };
            } else if (data.completed_stages?.includes(stage.name)) {
              return { ...stage, status: 'completed', progress: 100 };
            }
            return stage;
          }));
        }
        
        // Refresh experiment data
        fetchExperiment();
      } else if (data.type === 'stage_complete') {
        console.log('Stage complete:', data.stage);
        // Mark stage as complete
        setPipelineStages(prev => prev.map(stage => 
          stage.name === data.stage?.toLowerCase()
            ? { ...stage, status: 'completed', progress: 100, endTime: new Date().toISOString() }
            : stage
        ));
      } else if (data.type === 'stage_started') {
        console.log('Stage started:', data.stage);
        // Mark stage as running
        setPipelineStages(prev => prev.map(stage => 
          stage.name === data.stage?.toLowerCase()
            ? { ...stage, status: 'running', progress: 0, startTime: new Date().toISOString() }
            : stage
        ));
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };

    return () => {
      ws.close();
      setWsConnected(false);
      clearInterval(poll);
    };
  }, [id]);

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Set a basic experiment object if not loaded yet but we have an ID
  useEffect(() => {
    if (id && !experiment) {
      // Set a minimal experiment object to prevent "Loading..." state blocking the UI
      const timer = setTimeout(() => {
        if (!experiment) {
          console.log('Setting fallback experiment object');
          setExperiment({
            experiment_id: id,
            experiment_name: `Experiment ${id}`,
            status: 'running',
            created_at: new Date().toISOString()
          });
        }
      }, 2000); // Wait 2 seconds before showing fallback
      
      return () => clearTimeout(timer);
    }
  }, [id, experiment]);

  if (!experiment) {
    return (
      <div className="h-full flex items-center justify-center bg-bg-primary">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-text-secondary">Loading experiment...</p>
          <p className="text-text-muted text-sm mt-2">ID: {id}</p>
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
  // Extract micro-series from batch log (optional future)

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
                {currentStage && experiment.is_active && (
                  <span className="px-3 py-1 bg-primary/10 text-primary text-sm font-semibold rounded-full">
                    {currentStage}
                  </span>
                )}
              </div>
              <p className="text-text-secondary text-sm mt-1">
                Step {currentStep} / {totalSteps} • {Math.round(progress)}% • {
                  progress > 0 && experiment.is_active
                    ? `ETA: ${Math.round((100 - progress) * 0.5)} min`
                    : 'Calculating...'
                }
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* WebSocket Status Indicator */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium ${
              wsConnected 
                ? 'bg-accent/10 text-accent' 
                : 'bg-gray-100 text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-accent animate-pulse' : 'bg-gray-400'
              }`} />
              {wsConnected ? 'Live' : 'Connecting...'}
            </div>
            
            {/* Notification Settings (in a dropdown or modal) */}
            <div className="relative group">
              <button className="p-2 hover:bg-bg-tertiary rounded-lg transition-colors text-text-secondary hover:text-text-primary">
                🔔
              </button>
              <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-lg shadow-lg border border-border-light p-4 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                <NotificationSettings />
              </div>
            </div>
            
            <Button variant="secondary" size="sm" icon={<Download className="w-4 h-4" />}>
              Export Model
            </Button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-4">
          <ProgressBar progress={progress} status="running" animated />
        </div>

        {/* Pipeline Stages */}
        <div className="mt-6 flex items-center justify-between gap-4">
          {pipelineStages.map((stage, index) => (
            <div key={stage.name} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                {/* Stage Icon */}
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                    stage.status === 'completed'
                      ? 'bg-accent border-accent text-white'
                      : stage.status === 'running'
                      ? 'bg-primary border-primary text-white animate-pulse'
                      : stage.status === 'failed'
                      ? 'bg-error border-error text-white'
                      : 'bg-bg-tertiary border-border-medium text-text-muted'
                  }`}
                >
                  {stage.status === 'completed' ? (
                    <CheckCircle2 className="w-6 h-6" />
                  ) : stage.status === 'running' ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <Clock className="w-6 h-6" />
                  )}
                </div>
                {/* Stage Name */}
                <p
                  className={`text-xs font-medium mt-2 text-center ${
                    stage.status === 'running'
                      ? 'text-primary font-semibold'
                      : stage.status === 'completed'
                      ? 'text-accent'
                      : 'text-text-muted'
                  }`}
                >
                  {stage.displayName}
                </p>
                {/* Progress Percentage */}
                {stage.status === 'running' && (
                  <p className="text-xs text-primary font-bold mt-1">
                    {Math.round(stage.progress)}%
                  </p>
                )}
                {stage.message && stage.status === 'running' && (
                  <p className="text-xs text-text-secondary mt-1">{stage.message}</p>
                )}
              </div>
              {/* Connector Line */}
              {index < pipelineStages.length - 1 && (
                <div
                  className={`h-1 flex-1 mx-2 rounded transition-all ${
                    stage.status === 'completed'
                      ? 'bg-accent'
                      : stage.status === 'running'
                      ? 'bg-gradient-to-r from-primary to-bg-tertiary'
                      : 'bg-bg-tertiary'
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-4 border-b border-border-light">
          <button
            onClick={() => setActiveTab('training')}
            className={`px-6 py-3 font-medium text-sm transition-all relative ${
              activeTab === 'training'
                ? 'text-primary border-b-2 border-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Training Metrics
            </div>
          </button>
          <button
            onClick={() => setActiveTab('evaluation')}
            className={`px-6 py-3 font-medium text-sm transition-all relative ${
              activeTab === 'evaluation'
                ? 'text-primary border-b-2 border-primary'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Evaluation
            </div>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto custom-scrollbar">
        {activeTab === 'training' ? (
          <div className="p-8 space-y-6">{/* Training content */}
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

          {/* Confusion matrices (fetch when available) */}
          {id && (
            <div className="grid grid-cols-2 gap-6">
              <Card padding="lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-bold text-text-primary">Confusion Matrices</h3>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={async () => {
                      try {
                        const t = await fetchConfusion(id, 'teacher');
                        const s = await fetchConfusion(id, 'student');
                        setTeacherCM({ image_path: t.image_path, metrics: t.metrics });
                        setStudentCM({ image_path: s.image_path, metrics: s.metrics });
                      } catch (e) {
                        console.warn('Confusion not ready yet');
                      }
                    }}
                  >Refresh</Button>
                </div>
                <div className="grid grid-cols-2 gap-6">
                  <ConfusionMatrixCard role="teacher" imagePath={teacherCM?.image_path} metrics={teacherCM?.metrics} />
                  <ConfusionMatrixCard role="student" imagePath={studentCM?.image_path} metrics={studentCM?.metrics} />
                </div>
              </Card>
            </div>
          )}

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

          {/* Artifacts Gallery (thumbnails) */}
          {artifactImages.length > 0 && (
            <Card padding="lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-bold text-text-primary">Artifacts</h3>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={async () => {
                    if (!id) return;
                    try {
                      const art = await fetchArtifacts(id);
                      setArtifactImages(art.images || []);
                    } catch {}
                  }}
                >Refresh</Button>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {artifactImages.slice(0, 12).map((img) => (
                  <img key={img} src={`/experiments/${img}`} className="w-full rounded border border-border-light" />
                ))}
              </div>
            </Card>
          )}
          </div>
        ) : (
          /* Evaluation Tab */
          <div className="p-8">
            <EvaluationMonitor
              experimentId={id || ''}
              onComplete={(result) => {
                console.log('Evaluation completed:', result);
                setEvaluation(result);
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
