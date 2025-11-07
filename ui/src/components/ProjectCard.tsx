import { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, ChevronRight, CheckSquare, Zap, Minimize2, BarChart3, Rocket } from 'lucide-react';
import { StageNode, StageStatus } from './StageNode';

interface Stage {
  id: string;
  label: string;
  icon: React.ReactNode;
  status: StageStatus;
  color: string;
  tooltipText?: string;
}

interface ProjectCardProps {
  title: string;
  modelCount: number;
  lastUpdated: string;
  stages: Stage[];
  cpuUsage?: number;
  gpuUsage?: number;
  onViewDetails: () => void;
  onViewLive?: () => void;
  autoProgress?: boolean;
}

export function ProjectCard({
  title,
  modelCount,
  lastUpdated,
  stages: initialStages,
  cpuUsage = 0,
  gpuUsage = 0,
  onViewDetails,
  onViewLive,
  autoProgress = false,
}: ProjectCardProps) {
  const [stages] = useState(initialStages);
  const [progress, setProgress] = useState(() => {
    const completed = initialStages.filter(s => s.status === 'completed').length;
    return (completed / initialStages.length) * 100;
  });

  // Calculate current stage and overall progress
  const runningStageIndex = stages.findIndex(s => s.status === 'running');
  const currentStageLabel = runningStageIndex >= 0 ? stages[runningStageIndex].label : 'Queued';

  // Auto-increment progress for running projects
  useEffect(() => {
    if (!autoProgress || runningStageIndex < 0) return;

    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 2;
        if (newProgress >= 100) return 100;
        return Math.min(newProgress, 100);
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [autoProgress, runningStageIndex]);

  return (
    <div className="group relative rounded-2xl card-pastel shadow-pastel-md hover:shadow-pastel-lg transition-smooth overflow-hidden">
      {/* Top refraction highlight */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-purple-300/40 to-transparent" />

      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gradient-pastel group-hover:scale-105 transition-transform duration-300">
              {title}
            </h3>
            <div className="flex items-center gap-3 mt-2 text-sm">
              <span className="badge-pastel-blue">{modelCount} models</span>
              <span className="text-purple-300 dark:text-purple-400">•</span>
              <span className="text-slate-600 dark:text-slate-400">Updated {lastUpdated}</span>
            </div>
          </div>

          {/* Resource usage badges */}
          {(cpuUsage > 0 || gpuUsage > 0) && (
            <div className="flex gap-2">
              {cpuUsage > 0 && (
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl pastel-gradient-blue border border-blue-200 dark:border-blue-400/50 shadow-pastel-sm">
                  <Cpu className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-bold text-blue-700 dark:text-blue-300">{cpuUsage}%</span>
                </div>
              )}
              {gpuUsage > 0 && (
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl pastel-gradient-pink border border-purple-200 dark:border-purple-400/50 shadow-pastel-sm">
                  <HardDrive className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-bold text-purple-700 dark:text-purple-300">{gpuUsage}%</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Progress section */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">Stage: {currentStageLabel}</span>
            <span className="text-sm font-bold badge-pastel-blue">{Math.round(progress)}%</span>
          </div>
          
          {/* Progress bar */}
          <div className="progress-pastel">
            <div
              className="progress-pastel-fill"
              style={{ width: `${progress}%` }}
            >
              {/* Shimmer effect for running projects */}
              {autoProgress && runningStageIndex >= 0 && (
                <div className="absolute inset-0 shimmer-pastel" />
              )}
            </div>
          </div>
        </div>

        {/* Stage pipeline */}
        <div className="flex items-center justify-between px-6 py-4 glass-pastel rounded-2xl border-2 border-gradient-pastel shadow-pastel-sm">
          {stages.map((stage, index) => (
            <StageNode
              key={stage.id}
              label={stage.label}
              icon={stage.icon}
              status={stage.status}
              color={stage.color}
              tooltipText={stage.tooltipText}
              isLast={index === stages.length - 1}
            />
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg pastel-gradient-green">
            <Activity className="w-4 h-4 text-green-600 dark:text-green-400 pulse-pastel" />
            <span className="text-sm font-semibold text-green-700 dark:text-green-300">Live monitoring</span>
          </div>

          <div className="flex items-center gap-2">
            {/* View Live button - only show if training is running */}
            {runningStageIndex >= 0 && onViewLive && (
              <button
                onClick={onViewLive}
                className="btn-pastel-primary shadow-pastel-md glow-pastel"
              >
                <Activity className="w-4 h-4 pulse-pastel" />
                <span>View Live</span>
              </button>
            )}
            
            <button
              onClick={onViewDetails}
              className="btn-pastel-secondary shadow-pastel-sm hover:shadow-pastel-md"
            >
              <span>View Details</span>
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Stage icon components
export const stageIcons = {
  preflight: <CheckSquare className="w-7 h-7" strokeWidth={1.5} />,
  distillation: <Zap className="w-7 h-7" strokeWidth={1.5} />,
  quantization: <Minimize2 className="w-7 h-7" strokeWidth={1.5} />,
  evaluation: <BarChart3 className="w-7 h-7" strokeWidth={1.5} />,
  deployment: <Rocket className="w-7 h-7" strokeWidth={1.5} />,
};

// Stage color palette
export const stageColors = {
  preflight: '#C7E6FF',
  distillation: '#E0D7FF',
  quantization: '#B9F4E0',
  evaluation: '#FFD5D5',
  deployment: '#FFF0B3',
};
