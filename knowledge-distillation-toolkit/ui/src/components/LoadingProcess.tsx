import { useState, useEffect } from 'react';
import { 
  Database, FileSearch, Cpu, HardDrive, CheckCircle2, Loader2,
  Zap, Activity, PlayCircle, PauseCircle, AlertCircle, TrendingUp,
  Download, RefreshCw, Settings
} from 'lucide-react';

interface ProcessStep {
  id: string;
  label: string;
  icon: React.ReactNode;
  status: 'pending' | 'loading' | 'completed' | 'error';
  duration: number; // ms
  details?: string;
  timestamp?: Date;
}

interface BackgroundActivity {
  id: string;
  type: 'training' | 'evaluation' | 'quantization' | 'system' | 'data';
  label: string;
  status: 'active' | 'completed' | 'error' | 'paused';
  progress?: number;
  icon: React.ReactNode;
  details?: string;
  timestamp: Date;
  experimentId?: string;
}

export function LoadingProcess() {
  const [steps, setSteps] = useState<ProcessStep[]>([
    {
      id: 'connect',
      label: 'Connecting to backend',
      icon: <Database className="w-5 h-5" />,
      status: 'loading',
      duration: 800,
      timestamp: new Date(),
    },
    {
      id: 'scan',
      label: 'Scanning experiment directory',
      icon: <FileSearch className="w-5 h-5" />,
      status: 'pending',
      duration: 1000,
      timestamp: new Date(),
    },
    {
      id: 'parse',
      label: 'Parsing experiment metadata',
      icon: <HardDrive className="w-5 h-5" />,
      status: 'pending',
      duration: 600,
      timestamp: new Date(),
    },
    {
      id: 'compute',
      label: 'Computing metrics & status',
      icon: <Cpu className="w-5 h-5" />,
      status: 'pending',
      duration: 700,
      timestamp: new Date(),
    },
  ]);

  const [backgroundActivities, setBackgroundActivities] = useState<BackgroundActivity[]>([]);
  const [showActivities, setShowActivities] = useState(false);

  // Connect to WebSocket for real-time activity updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onopen = () => {
      console.log('[LoadingProcess] WebSocket connected');
      addActivity({
        id: `ws-${Date.now()}`,
        type: 'system',
        label: 'WebSocket connected',
        status: 'completed',
        icon: <Zap className="w-4 h-4" />,
        details: 'Real-time updates enabled',
        timestamp: new Date(),
      });
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('[LoadingProcess] WebSocket error:', error);
      addActivity({
        id: `ws-error-${Date.now()}`,
        type: 'system',
        label: 'WebSocket connection error',
        status: 'error',
        icon: <AlertCircle className="w-4 h-4" />,
        details: 'Failed to establish real-time connection',
        timestamp: new Date(),
      });
    };
    
    ws.onclose = () => {
      console.log('[LoadingProcess] WebSocket disconnected');
    };
    
    return () => {
      ws.close();
    };
  }, []);

  const handleWebSocketMessage = (data: any) => {
    const timestamp = new Date();
    
    switch (data.type) {
      case 'training_started':
        addActivity({
          id: `train-start-${data.experiment_id}`,
          type: 'training',
          label: 'Training started',
          status: 'active',
          icon: <PlayCircle className="w-4 h-4" />,
          details: data.experiment_name || data.experiment_id,
          timestamp,
          experimentId: data.experiment_id,
        });
        break;
      
      case 'training_update':
        const status = data.status === 'completed' ? 'completed' : 
                      data.status === 'failed' ? 'error' :
                      data.status === 'paused' ? 'paused' : 'active';
        
        addActivity({
          id: `train-update-${data.experiment_id}-${timestamp.getTime()}`,
          type: 'training',
          label: `Training ${data.status}`,
          status,
          icon: status === 'completed' ? <CheckCircle2 className="w-4 h-4" /> :
                status === 'error' ? <AlertCircle className="w-4 h-4" /> :
                status === 'paused' ? <PauseCircle className="w-4 h-4" /> :
                <Activity className="w-4 h-4" />,
          details: data.message || data.stage || data.experiment_id,
          timestamp,
          experimentId: data.experiment_id,
        });
        break;
      
      case 'training_metrics':
        addActivity({
          id: `metrics-${data.experiment_id}-${timestamp.getTime()}`,
          type: 'training',
          label: 'Training progress',
          status: 'active',
          progress: data.metrics?.progress,
          icon: <TrendingUp className="w-4 h-4" />,
          details: `Epoch ${data.metrics?.epoch}/${data.metrics?.totalEpochs} - Loss: ${data.metrics?.loss?.toFixed(4)} - Acc: ${(data.metrics?.accuracy * 100).toFixed(1)}%`,
          timestamp,
          experimentId: data.experiment_id,
        });
        break;
      
      case 'training_log':
        if (data.level === 'error' || data.level === 'success') {
          addActivity({
            id: `log-${data.experiment_id}-${timestamp.getTime()}`,
            type: 'system',
            label: data.level === 'error' ? 'Training error' : 'Training success',
            status: data.level === 'error' ? 'error' : 'completed',
            icon: data.level === 'error' ? <AlertCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />,
            details: data.message,
            timestamp,
            experimentId: data.experiment_id,
          });
        }
        break;
      
      case 'experiment_update':
        addActivity({
          id: `exp-update-${timestamp.getTime()}`,
          type: 'system',
          label: 'Experiment updated',
          status: 'completed',
          icon: <RefreshCw className="w-4 h-4" />,
          details: 'Experiment list refreshed',
          timestamp,
        });
        break;
      
      case 'checkpoint_saved':
        addActivity({
          id: `checkpoint-${data.experiment_id}-${timestamp.getTime()}`,
          type: 'system',
          label: 'Checkpoint saved',
          status: 'completed',
          icon: <Download className="w-4 h-4" />,
          details: data.experiment_id,
          timestamp,
          experimentId: data.experiment_id,
        });
        break;
      
      default:
        // Generic activity for unknown message types
        addActivity({
          id: `activity-${timestamp.getTime()}`,
          type: 'system',
          label: data.type || 'System activity',
          status: 'completed',
          icon: <Activity className="w-4 h-4" />,
          details: JSON.stringify(data).slice(0, 100),
          timestamp,
        });
    }
  };

  const addActivity = (activity: BackgroundActivity) => {
    setBackgroundActivities(prev => {
      // Keep only last 50 activities
      const updated = [activity, ...prev].slice(0, 50);
      return updated;
    });
    
    // Auto-show activities panel when new activity arrives
    if (!showActivities) {
      setShowActivities(true);
    }
  };

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    const processNextStep = (currentSteps: ProcessStep[]) => {
      // Find first non-completed step
      const nextStepIndex = currentSteps.findIndex(
        s => s.status !== 'completed'
      );

      if (nextStepIndex === -1) {
        // All completed - show background activities
        setTimeout(() => setShowActivities(true), 500);
        return;
      }

      const nextStep = currentSteps[nextStepIndex];

      if (nextStep.status === 'pending') {
        // Start loading this step
        setSteps(prev =>
          prev.map((s, i) =>
            i === nextStepIndex ? { ...s, status: 'loading', timestamp: new Date() } : s
          )
        );

        // Complete it after duration
        timeoutId = setTimeout(() => {
          setSteps(prev =>
            prev.map((s, i) =>
              i === nextStepIndex ? { ...s, status: 'completed', timestamp: new Date() } : s
            )
          );
        }, nextStep.duration);
      } else if (nextStep.status === 'completed') {
        // Move to next step immediately
        processNextStep(
          currentSteps.map((s, i) =>
            i === nextStepIndex ? { ...s, status: 'completed' } : s
          )
        );
      }
    };

    // Start processing after mount
    timeoutId = setTimeout(() => {
      processNextStep(steps);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [steps]);

  const getActivityIcon = (activity: BackgroundActivity) => {
    return activity.icon;
  };

  const getActivityColor = (activity: BackgroundActivity) => {
    switch (activity.status) {
      case 'completed':
        return 'pastel-gradient-green border-green-200 dark:border-green-400 text-green-800 dark:text-green-300';
      case 'error':
        return 'pastel-gradient-pink border-red-200 dark:border-red-400 text-red-800 dark:text-red-300';
      case 'paused':
        return 'pastel-gradient-orange border-orange-200 dark:border-orange-400 text-orange-800 dark:text-orange-300';
      case 'active':
      default:
        return 'pastel-gradient-blue border-blue-200 dark:border-blue-400 text-blue-800 dark:text-blue-300 glow-pastel';
    }
  };

  const getActivityIconBg = (activity: BackgroundActivity) => {
    switch (activity.status) {
      case 'completed':
        return 'bg-gradient-to-br from-green-300 to-teal-300';
      case 'error':
        return 'bg-gradient-to-br from-red-300 to-pink-300';
      case 'paused':
        return 'bg-gradient-to-br from-orange-300 to-yellow-300';
      case 'active':
      default:
        return 'bg-gradient-to-br from-purple-300 to-blue-300 pulse-pastel';
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const allStepsComplete = steps.every(s => s.status === 'completed');

  return (
    <div className="flex items-center justify-center min-h-[400px] p-6 bg-animated-pastel">
      <div className="w-full max-w-4xl space-y-6">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="w-20 h-20 mx-auto mb-4 relative float-animation">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-purple-200 to-blue-200 opacity-30 blur-xl animate-pulse"></div>
            <div className="relative w-20 h-20 rounded-full glass-pastel border-2 border-gradient-pastel flex items-center justify-center shadow-pastel-lg">
              {allStepsComplete ? (
                <CheckCircle2 className="w-10 h-10 text-green-500" />
              ) : (
                <Loader2 className="w-10 h-10 text-purple-500 animate-pastel-spin" />
              )}
            </div>
          </div>
          <h3 className="text-2xl font-bold text-gradient-pastel mb-2">
            {allStepsComplete ? 'System Ready' : 'Initializing System'}
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {allStepsComplete 
              ? 'Monitoring background activities...' 
              : 'Please wait while we set up your workspace'}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Initial Setup Steps */}
          <div className="space-y-3 card-pastel p-6 hover-lift">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-bold flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-200 to-blue-200 flex items-center justify-center">
                  <Settings className="w-4 h-4 text-purple-700" />
                </div>
                <span className="text-gradient-pastel">Initialization</span>
              </h4>
              <div className="badge-pastel-blue">
                {steps.filter(s => s.status === 'completed').length}/{steps.length}
              </div>
            </div>
            
            {steps.map((step) => (
              <div
                key={step.id}
                className={`
                  flex items-center gap-4 p-4 rounded-xl transition-smooth
                  ${
                    step.status === 'completed'
                      ? 'pastel-gradient-green border-2 border-green-200'
                      : step.status === 'loading'
                      ? 'pastel-gradient-blue border-2 border-blue-200 glow-pastel'
                      : step.status === 'error'
                      ? 'pastel-gradient-pink border-2 border-red-200'
                      : 'bg-white/60 dark:bg-slate-800/60 border-2 border-slate-200 dark:border-slate-700'
                  }
                `}
              >
                {/* Icon */}
                <div
                  className={`
                    flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-smooth shadow-pastel-md
                    ${
                      step.status === 'completed'
                        ? 'bg-gradient-to-br from-green-300 to-teal-300 text-white'
                        : step.status === 'loading'
                        ? 'bg-gradient-to-br from-purple-300 to-blue-300 text-white pulse-pastel'
                        : step.status === 'error'
                        ? 'bg-gradient-to-br from-red-300 to-pink-300 text-white'
                        : 'bg-slate-200 dark:bg-slate-700 text-slate-400 dark:text-slate-500'
                    }
                  `}
                >
                  {step.status === 'completed' ? (
                    <CheckCircle2 className="w-6 h-6" />
                  ) : step.status === 'error' ? (
                    <AlertCircle className="w-6 h-6" />
                  ) : (
                    step.icon
                  )}
                </div>

                {/* Label */}
                <div className="flex-1 min-w-0">
                  <p
                    className={`
                      text-sm font-bold transition-colors duration-300 truncate
                      ${
                        step.status === 'completed'
                          ? 'text-green-700 dark:text-green-400'
                          : step.status === 'loading'
                          ? 'text-purple-700 dark:text-purple-400'
                          : step.status === 'error'
                          ? 'text-red-700 dark:text-red-400'
                          : 'text-slate-500 dark:text-slate-400'
                      }
                    `}
                  >
                    {step.label}
                  </p>
                  {step.details && (
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 truncate">
                      {step.details}
                    </p>
                  )}
                </div>

                {/* Status indicator */}
                {step.status === 'loading' && (
                  <div className="flex-shrink-0">
                    <div className="flex gap-1.5">
                      <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce [animation-delay:0ms]"></span>
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]"></span>
                      <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce [animation-delay:300ms]"></span>
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            {/* Progress bar */}
            <div className="mt-6 pt-4 border-t-2 border-purple-100 dark:border-slate-700">
              <div className="progress-pastel">
                <div
                  className="progress-pastel-fill"
                  style={{
                    width: `${
                      (steps.filter(s => s.status === 'completed').length /
                        steps.length) *
                      100
                    }%`,
                  }}
                />
              </div>
              <div className="mt-3 flex justify-between text-xs font-semibold">
                <span className="text-purple-600 dark:text-purple-400">
                  {steps.filter(s => s.status === 'completed').length} of{' '}
                  {steps.length} complete
                </span>
                <span className="badge-pastel-blue">
                  {Math.round(
                    (steps.filter(s => s.status === 'completed').length /
                      steps.length) *
                      100
                  )}
                  %
                </span>
              </div>
            </div>
          </div>

          {/* Right: Background Activities */}
          <div className="space-y-3 card-pastel p-6 hover-lift">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-bold flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-200 to-purple-200 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-pink-700" />
                </div>
                <span className="text-gradient-pastel">Activities</span>
              </h4>
              {backgroundActivities.length > 0 && (
                <div className="badge-pastel-pink">
                  {backgroundActivities.filter(a => a.status === 'active').length} active
                </div>
              )}
            </div>
            
            <div className="max-h-[400px] overflow-y-auto space-y-3 pr-2 custom-scrollbar">
              {backgroundActivities.length === 0 ? (
                <div className="flex items-center justify-center h-48 rounded-2xl glass-pastel border-2 border-dashed border-purple-200 dark:border-slate-700">
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-gradient-to-br from-purple-100 to-blue-100 flex items-center justify-center">
                      <Activity className="w-8 h-8 text-purple-400 animate-pulse" />
                    </div>
                    <p className="text-sm font-semibold text-slate-600 dark:text-slate-400">
                      No activities yet
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                      Activities will appear here
                    </p>
                  </div>
                </div>
              ) : (
                backgroundActivities.map((activity) => (
                  <div
                    key={activity.id}
                    className={`
                      flex items-start gap-3 p-4 rounded-xl border-2 transition-smooth hover-lift
                      ${getActivityColor(activity)}
                    `}
                  >
                    {/* Icon */}
                    <div
                      className={`
                        flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center text-white shadow-pastel-md
                        ${getActivityIconBg(activity)}
                      `}
                    >
                      {getActivityIcon(activity)}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm font-bold truncate">
                          {activity.label}
                        </p>
                        <span className="text-xs font-semibold whitespace-nowrap opacity-70">
                          {formatTime(activity.timestamp)}
                        </span>
                      </div>
                      {activity.details && (
                        <p className="text-xs opacity-80 mt-1 line-clamp-2">
                          {activity.details}
                        </p>
                      )}
                      {activity.progress !== undefined && (
                        <div className="mt-3">
                          <div className="progress-pastel">
                            <div
                              className="progress-pastel-fill"
                              style={{ width: `${activity.progress}%` }}
                            />
                          </div>
                          <p className="text-xs font-semibold mt-1 opacity-80">
                            {activity.progress.toFixed(0)}% complete
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
