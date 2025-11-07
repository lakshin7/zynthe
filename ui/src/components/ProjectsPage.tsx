import { useState, useEffect } from 'react';
import { Filter, PlayCircle, CheckCircle2, Clock, Plus, BarChart3 } from 'lucide-react';
import { ProjectCard, stageIcons, stageColors } from './ProjectCard';
import { ProjectDetailsModal } from './ProjectDetailsModal';
// import { NewTrainingModal } from './NewTrainingModal';
import { TrainingDashboard } from './TrainingDashboard';
import { ModelComparisonModal } from './ModelComparisonModal';
import { LoadingProcess } from './LoadingProcess';
import { useToast } from './Toast';

type FilterType = 'all' | 'running' | 'completed' | 'queued';

type StageStatus = 'completed' | 'running' | 'upcoming';

interface Stage {
  id: string;
  label: string;
  icon: React.ReactNode;
  status: StageStatus;
  color: string;
  tooltipText: string;
}

interface Project {
  id: string;
  title: string;
  modelCount: number;
  lastUpdated: string;
  cpuUsage: number;
  gpuUsage: number;
  autoProgress: boolean;
  stages: Stage[];
}

// Mock project data (fallback)
const mockProjects: Project[] = [
  {
    id: '1',
    title: 'LLaMA 3.1 Quantization',
    modelCount: 3,
    lastUpdated: '2 min ago',
    cpuUsage: 45,
    gpuUsage: 87,
    autoProgress: true,
    stages: [
      {
        id: 'preflight-1',
        label: 'Preflight',
        icon: stageIcons.preflight,
        status: 'completed' as const,
        color: stageColors.preflight,
        tooltipText: 'Preflight checks: passed ✅',
      },
      {
        id: 'distillation-1',
        label: 'Distillation',
        icon: stageIcons.distillation,
        status: 'completed' as const,
        color: stageColors.distillation,
        tooltipText: 'Distillation loss: 0.87 (converged)',
      },
      {
        id: 'quantization-1',
        label: 'Quantization',
        icon: stageIcons.quantization,
        status: 'running' as const,
        color: stageColors.quantization,
        tooltipText: 'Quantization: INT8 conversion 67% complete',
      },
      {
        id: 'evaluation-1',
        label: 'Evaluation',
        icon: stageIcons.evaluation,
        status: 'upcoming' as const,
        color: stageColors.evaluation,
        tooltipText: 'Evaluation: pending',
      },
      {
        id: 'deployment-1',
        label: 'Deployment',
        icon: stageIcons.deployment,
        status: 'upcoming' as const,
        color: stageColors.deployment,
        tooltipText: 'Deployment: pending',
      },
    ],
  },
  {
    id: '2',
    title: 'BERT-Base Distillation',
    modelCount: 2,
    lastUpdated: '15 min ago',
    cpuUsage: 0,
    gpuUsage: 0,
    autoProgress: false,
    stages: [
      {
        id: 'preflight-2',
        label: 'Preflight',
        icon: stageIcons.preflight,
        status: 'completed' as const,
        color: stageColors.preflight,
        tooltipText: 'Preflight checks: passed ✅',
      },
      {
        id: 'distillation-2',
        label: 'Distillation',
        icon: stageIcons.distillation,
        status: 'completed' as const,
        color: stageColors.distillation,
        tooltipText: 'Distillation: completed successfully',
      },
      {
        id: 'quantization-2',
        label: 'Quantization',
        icon: stageIcons.quantization,
        status: 'completed' as const,
        color: stageColors.quantization,
        tooltipText: 'Quantization: INT8 applied',
      },
      {
        id: 'evaluation-2',
        label: 'Evaluation',
        icon: stageIcons.evaluation,
        status: 'completed' as const,
        color: stageColors.evaluation,
        tooltipText: 'Evaluation: 96.2% accuracy maintained',
      },
      {
        id: 'deployment-2',
        label: 'Deployment',
        icon: stageIcons.deployment,
        status: 'completed' as const,
        color: stageColors.deployment,
        tooltipText: 'Deployed to production',
      },
    ],
  },
  {
    id: '3',
    title: 'ResNet-50 Optimization',
    modelCount: 1,
    lastUpdated: '1 hour ago',
    cpuUsage: 0,
    gpuUsage: 0,
    autoProgress: false,
    stages: [
      {
        id: 'preflight-3',
        label: 'Preflight',
        icon: stageIcons.preflight,
        status: 'upcoming' as const,
        color: stageColors.preflight,
        tooltipText: 'Queued for execution',
      },
      {
        id: 'distillation-3',
        label: 'Distillation',
        icon: stageIcons.distillation,
        status: 'upcoming' as const,
        color: stageColors.distillation,
        tooltipText: 'Queued',
      },
      {
        id: 'quantization-3',
        label: 'Quantization',
        icon: stageIcons.quantization,
        status: 'upcoming' as const,
        color: stageColors.quantization,
        tooltipText: 'Queued',
      },
      {
        id: 'evaluation-3',
        label: 'Evaluation',
        icon: stageIcons.evaluation,
        status: 'upcoming' as const,
        color: stageColors.evaluation,
        tooltipText: 'Queued',
      },
      {
        id: 'deployment-3',
        label: 'Deployment',
        icon: stageIcons.deployment,
        status: 'upcoming' as const,
        color: stageColors.deployment,
        tooltipText: 'Queued',
      },
    ],
  },
  {
    id: '4',
    title: 'GPT-2 Small Compression',
    modelCount: 4,
    lastUpdated: '3 hours ago',
    cpuUsage: 28,
    gpuUsage: 62,
    autoProgress: true,
    stages: [
      {
        id: 'preflight-4',
        label: 'Preflight',
        icon: stageIcons.preflight,
        status: 'completed' as const,
        color: stageColors.preflight,
        tooltipText: 'Preflight checks: passed ✅',
      },
      {
        id: 'distillation-4',
        label: 'Distillation',
        icon: stageIcons.distillation,
        status: 'running' as const,
        color: stageColors.distillation,
        tooltipText: 'Distillation: epoch 5/10, loss decreasing',
      },
      {
        id: 'quantization-4',
        label: 'Quantization',
        icon: stageIcons.quantization,
        status: 'upcoming' as const,
        color: stageColors.quantization,
        tooltipText: 'Queued',
      },
      {
        id: 'evaluation-4',
        label: 'Evaluation',
        icon: stageIcons.evaluation,
        status: 'upcoming' as const,
        color: stageColors.evaluation,
        tooltipText: 'Queued',
      },
      {
        id: 'deployment-4',
        label: 'Deployment',
        icon: stageIcons.deployment,
        status: 'upcoming' as const,
        color: stageColors.deployment,
        tooltipText: 'Queued',
      },
    ],
  },
];

// Helper to format timestamp to "X min ago" format
function getRelativeTime(timestamp: string): string {
  const now = new Date();
  const then = new Date(timestamp);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} min ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
}

// Transform backend experiment data to UI format
function transformExperiment(exp: any): Project {
  const stages: Stage[] = [
    {
      id: `preflight-${exp.id}`,
      label: 'Preflight',
      icon: stageIcons.preflight,
      status: exp.stages.preflight as StageStatus,
      color: stageColors.preflight,
      tooltipText: exp.stages.preflight === 'completed' ? 'Preflight checks: passed ✅' : 
                   exp.stages.preflight === 'running' ? 'Running preflight checks...' : 
                   'Preflight: pending',
    },
    {
      id: `distillation-${exp.id}`,
      label: 'Distillation',
      icon: stageIcons.distillation,
      status: exp.stages.distillation as StageStatus,
      color: stageColors.distillation,
      tooltipText: exp.stages.distillation === 'completed' ? 'Distillation: completed ✅' : 
                   exp.stages.distillation === 'running' ? 'Training student model...' : 
                   'Distillation: pending',
    },
    {
      id: `quantization-${exp.id}`,
      label: 'Quantization',
      icon: stageIcons.quantization,
      status: exp.stages.quantization as StageStatus,
      color: stageColors.quantization,
      tooltipText: exp.stages.quantization === 'completed' ? 'Quantization: completed ✅' : 
                   exp.stages.quantization === 'running' ? 'Quantizing model...' : 
                   'Quantization: pending',
    },
    {
      id: `evaluation-${exp.id}`,
      label: 'Evaluation',
      icon: stageIcons.evaluation,
      status: exp.stages.evaluation as StageStatus,
      color: stageColors.evaluation,
      tooltipText: exp.stages.evaluation === 'completed' ? 'Evaluation: completed ✅' : 
                   exp.stages.evaluation === 'running' ? 'Evaluating model...' : 
                   'Evaluation: pending',
    },
    {
      id: `deployment-${exp.id}`,
      label: 'Deployment',
      icon: stageIcons.deployment,
      status: exp.stages.deployment as StageStatus,
      color: stageColors.deployment,
      tooltipText: exp.stages.deployment === 'completed' ? 'Deployment: ready ✅' : 
                   exp.stages.deployment === 'running' ? 'Deploying model...' : 
                   'Deployment: pending',
    },
  ];

  const hasRunningStage = stages.some(s => s.status === 'running');

  return {
    id: exp.id,
    title: exp.name || exp.id,
    modelCount: exp.model_count || 1,
    lastUpdated: getRelativeTime(exp.timestamp),
    cpuUsage: hasRunningStage ? Math.floor(Math.random() * 60) + 20 : 0,
    gpuUsage: hasRunningStage ? Math.floor(Math.random() * 40) + 50 : 0,
    autoProgress: hasRunningStage,
    stages,
  };
}

export function ProjectsPage() {
  const { showToast } = useToast();
  const [filter, setFilter] = useState<FilterType>('all');
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [showNewTraining, setShowNewTraining] = useState(false);
  const [showModelComparison, setShowModelComparison] = useState(false);
  const [liveTrainingId, setLiveTrainingId] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<Record<string, any>>({});

  // Fetch experiments from backend
  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://localhost:8765/api/experiments');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch experiments: ${response.statusText}`);
        }
        
        const data = await response.json();
        const transformedProjects = data.map((exp: any) => {
          // Apply live metrics if available
          const metrics = liveMetrics[exp.id];
          const project = transformExperiment(exp);
          
          if (metrics) {
            // Update project with live metrics
            project.cpuUsage = Math.floor(metrics.metrics?.progress || 0) % 100;
            project.gpuUsage = 70 + Math.floor((metrics.metrics?.progress || 0) % 30);
            project.autoProgress = true;
            
            // Update tooltip with live info
            const distStage = project.stages.find(s => s.label === 'Distillation');
            if (distStage && metrics.metrics) {
              distStage.tooltipText = `Epoch ${metrics.metrics.epoch}/${metrics.metrics.totalEpochs} - Loss: ${metrics.metrics.loss?.toFixed(4)} - Acc: ${(metrics.metrics.accuracy * 100).toFixed(1)}%`;
            }
          }
          
          return project;
        });
        setProjects(transformedProjects);
        
        // Show success toast
        if (transformedProjects.length > 0 && showToast) {
          showToast('success', `Loaded ${transformedProjects.length} experiment${transformedProjects.length > 1 ? 's' : ''}`);
        }
      } catch (err) {
        console.error('Failed to fetch experiments:', err);
        setError(err instanceof Error ? err.message : 'Failed to load experiments');
        if (showToast) {
          showToast('error', 'Failed to load experiments from backend');
        }
        // Fallback to mock data on error
        setProjects(mockProjects);
      } finally {
        setLoading(false);
      }
    };

    fetchExperiments();
    
    // Use WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket update:', data);
      
      // Handle different message types
      if (data.type === 'training_started') {
        // Set the live training ID and open dashboard
        setLiveTrainingId(data.experiment_id);
        fetchExperiments();
      } else if (data.type === 'training_metrics') {
        // Update live metrics for the experiment
        setLiveMetrics(prev => ({
          ...prev,
          [data.experiment_id]: data
        }));
        // Refresh to show updated metrics
        fetchExperiments();
      } else if (data.type === 'training_update') {
        // Refresh experiments when training status changes
        fetchExperiments();
        
        // If training completed, clear live training
        if (data.status === 'completed' || data.status === 'failed') {
          if (liveTrainingId === data.experiment_id) {
            // Keep dashboard open for a moment to show completion
            setTimeout(() => {
              setLiveTrainingId(null);
            }, 3000);
          }
        }
      } else if (data.type === 'experiment_update') {
        fetchExperiments();
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    return () => {
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Calculate stats
  const runningCount = projects.filter(p => 
    p.stages.some(s => s.status === 'running')
  ).length;
  const completedCount = projects.filter(p => 
    p.stages.every(s => s.status === 'completed')
  ).length;
  const queuedCount = projects.filter(p => 
    p.stages.every(s => s.status === 'upcoming')
  ).length;

  // Filter projects
  const filteredProjects = projects.filter(project => {
    if (filter === 'all') return true;
    if (filter === 'running') return project.stages.some(s => s.status === 'running');
    if (filter === 'completed') return project.stages.every(s => s.status === 'completed');
    if (filter === 'queued') return project.stages.every(s => s.status === 'upcoming');
    return true;
  });

  const selectedProjectData = projects.find(p => p.id === selectedProject);

  const handleStartTraining = async (config: any) => {
    try {
      const response = await fetch('http://localhost:8765/api/training/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      const result = await response.json();
      console.log('Training started:', result);
      
      // Automatically open the training dashboard
      setLiveTrainingId(result.experiment_id);
      setShowNewTraining(false);
      
      // Refresh experiments list after a short delay
      setTimeout(() => {
        window.location.reload();
      }, 500);
    } catch (err) {
      console.error('Failed to start training:', err);
      alert('Failed to start training. Please try again.');
    }
  };

  return (
    <div className="h-full flex flex-col bg-animated-pastel">
      {/* Header with status bar */}
      <div className="flex-shrink-0 p-6 space-y-4">
        {/* Status bar */}
        <div className="flex items-center justify-between p-5 rounded-2xl glass-pastel border-2 border-gradient-pastel shadow-pastel-lg hover-lift">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-200 to-teal-200 flex items-center justify-center">
                <PlayCircle className="w-5 h-5 text-blue-700" />
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-700 dark:text-blue-400">{runningCount}</div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">Running</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-200 to-teal-200 flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-green-700" />
              </div>
              <div>
                <div className="text-2xl font-bold text-green-700 dark:text-green-400">{completedCount}</div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">Completed</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-200 to-yellow-200 flex items-center justify-center">
                <Clock className="w-5 h-5 text-orange-700" />
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-700 dark:text-orange-400">{queuedCount}</div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">Queued</span>
              </div>
            </div>
          </div>

          {/* Filter buttons */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-200 to-pink-200 flex items-center justify-center">
              <Filter className="w-4 h-4 text-purple-700" />
            </div>
            <div className="flex gap-2 glass-pastel p-1.5 rounded-xl border border-purple-200 dark:border-slate-700">
              {[
                { value: 'all', label: 'All' },
                { value: 'running', label: 'Running' },
                { value: 'completed', label: 'Completed' },
                { value: 'queued', label: 'Queued' },
              ].map((f) => (
                <button
                  key={f.value}
                  onClick={() => setFilter(f.value as FilterType)}
                  className={`px-4 py-2 rounded-lg font-semibold transition-smooth text-sm ${
                    filter === f.value
                      ? 'bg-gradient-to-br from-purple-300 to-pink-300 text-white shadow-pastel-md'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-white/50 dark:hover:bg-slate-700/50'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Page title and buttons */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-gradient-pastel">Active Projects</h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 font-medium">
              Monitor and manage your model compression pipelines
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => setShowModelComparison(true)}
              className="btn-pastel-secondary hover-lift shadow-pastel-md"
            >
              <BarChart3 className="w-5 h-5" />
              Compare Models
            </button>
            <button
              onClick={() => setShowNewTraining(true)}
              className="btn-pastel-primary hover-lift shadow-pastel-lg glow-pastel"
            >
              <Plus className="w-5 h-5" />
              New Training
            </button>
          </div>
        </div>
      </div>

      {/* Project cards grid */}
      <div className="flex-1 overflow-auto px-6 pb-6 custom-scrollbar">
        {loading ? (
          <LoadingProcess />
        ) : error ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center space-y-4 card-pastel p-8 max-w-md hover-lift">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-red-200 to-pink-200 flex items-center justify-center shadow-pastel-md">
                <span className="text-3xl">⚠️</span>
              </div>
              <h3 className="text-xl font-bold text-gradient-pastel">Failed to load experiments</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">{error}</p>
              <button 
                onClick={() => window.location.reload()}
                className="btn-pastel-primary shadow-pastel-md hover-lift"
              >
                Retry
              </button>
            </div>
          </div>
        ) : filteredProjects.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center space-y-4 card-pastel p-8 hover-lift">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-purple-200 to-blue-200 flex items-center justify-center shadow-pastel-md">
                <span className="text-3xl">📂</span>
              </div>
              <p className="text-lg font-bold text-gradient-pastel">No experiments found</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {filter !== 'all' ? 'Try changing the filter' : 'Start a new training run to get started'}
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {filteredProjects.map((project) => (
              <ProjectCard
                key={project.id}
                title={project.title}
                modelCount={project.modelCount}
                lastUpdated={project.lastUpdated}
                stages={project.stages}
                cpuUsage={project.cpuUsage}
                gpuUsage={project.gpuUsage}
                autoProgress={project.autoProgress}
                onViewDetails={() => setSelectedProject(project.id)}
                onViewLive={() => setLiveTrainingId(project.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Training Dashboard */}
      {liveTrainingId && (
        <TrainingDashboard
          experimentId={liveTrainingId}
          onClose={() => setLiveTrainingId(null)}
        />
      )}

      {/* Details modal */}
      {selectedProject && selectedProjectData && (
        <ProjectDetailsModal
          onClose={() => setSelectedProject(null)}
          projectTitle={selectedProjectData.title}
          projectId={selectedProjectData.id}
        />
      )}

      {/* New Training modal */}
      {/* Temporarily disabled until NewTrainingModal is restored */}
      {showNewTraining && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-8">
          <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm" onClick={() => setShowNewTraining(false)} />
          <div className="relative bg-white dark:bg-slate-800 rounded-2xl p-8 max-w-md shadow-2xl">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
              New Training
            </h3>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              NewTrainingModal component needs to be restored. Please check the file.
            </p>
            <button
              onClick={() => setShowNewTraining(false)}
              className="w-full px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-600 text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Model Comparison modal */}
      {showModelComparison && (
        <ModelComparisonModal
          onClose={() => setShowModelComparison(false)}
        />
      )}
    </div>
  );
}
