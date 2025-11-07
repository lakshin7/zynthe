import { CheckCircle2, AlertCircle, Loader2, Clock } from 'lucide-react';
import { useState } from 'react';

export type StageStatus = 'upcoming' | 'running' | 'completed' | 'failed';

interface StageNodeProps {
  label: string;
  icon: React.ReactNode;
  status: StageStatus;
  color: string;
  tooltipText?: string;
  isLast?: boolean;
}

const statusConfig = {
  upcoming: {
    opacity: 'opacity-40',
    glow: '',
    icon: Clock,
    iconColor: 'text-slate-400 dark:text-slate-500',
  },
  running: {
    opacity: 'opacity-100',
    glow: 'shadow-lg',
    icon: Loader2,
    iconColor: 'text-current',
  },
  completed: {
    opacity: 'opacity-100',
    glow: '',
    icon: CheckCircle2,
    iconColor: 'text-emerald-500 dark:text-emerald-400',
  },
  failed: {
    opacity: 'opacity-100',
    glow: 'shadow-lg shadow-rose-500/30',
    icon: AlertCircle,
    iconColor: 'text-rose-500 dark:text-rose-400',
  },
};

export function StageNode({ label, icon, status, color, tooltipText, isLast = false }: StageNodeProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <div className="flex items-center gap-3">
      <div className="flex flex-col items-center gap-2 relative">
        {/* Stage node */}
        <div
          className={`
            relative w-16 h-16 rounded-xl flex items-center justify-center
            backdrop-blur-md border transition-all duration-500
            ${config.opacity} ${config.glow}
            ${status === 'completed' ? 'bg-emerald-50/50 dark:bg-emerald-900/20 border-emerald-200/50 dark:border-emerald-500/30' :
              status === 'failed' ? 'bg-rose-50/50 dark:bg-rose-900/20 border-rose-200/50 dark:border-rose-500/30' :
              status === 'running' ? `bg-white/60 dark:bg-slate-700/60 border-white/30 dark:border-white/20` :
              'bg-white/30 dark:bg-slate-700/30 border-white/20 dark:border-white/10'
            }
          `}
          style={status === 'running' ? { backgroundColor: `${color}30` } : {}}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          {/* Main icon */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div style={{ color: status === 'running' ? color : undefined }}>
              {icon}
            </div>
          </div>

          {/* Status indicator */}
          <div className={`absolute -bottom-1 -right-1 ${config.iconColor}`}>
            <StatusIcon 
              className={`w-5 h-5 ${status === 'running' ? 'animate-spin' : ''}`}
              strokeWidth={2.5}
            />
          </div>

          {/* Pulsing glow for running state */}
          {status === 'running' && (
            <div
              className="absolute inset-0 rounded-xl animate-pulse"
              style={{
                background: `radial-gradient(circle at center, ${color}40, transparent 70%)`,
              }}
            />
          )}
        </div>

        {/* Label */}
        <span className={`text-center text-slate-700 dark:text-slate-300 transition-opacity duration-500 ${config.opacity}`}>
          {label}
        </span>

        {/* Tooltip */}
        {showTooltip && tooltipText && (
          <div className="absolute top-full mt-2 px-3 py-2 rounded-lg bg-slate-900/90 dark:bg-slate-800/90 text-white backdrop-blur-sm border border-white/10 shadow-xl z-10 whitespace-nowrap animate-in fade-in duration-200">
            <div className="text-xs">{tooltipText}</div>
          </div>
        )}
      </div>

      {/* Connector line to next stage */}
      {!isLast && (
        <div className="relative w-16 h-1 mb-8">
          {/* Background line */}
          <div className="absolute inset-0 bg-white/30 dark:bg-slate-700/30 rounded-full" />
          
          {/* Progress line */}
          {(status === 'completed' || status === 'running') && (
            <div
              className="absolute inset-0 rounded-full transition-all duration-500"
              style={{
                background: status === 'completed' 
                  ? 'linear-gradient(90deg, #10b981, #10b981)' 
                  : `linear-gradient(90deg, ${color}, transparent)`,
                width: status === 'completed' ? '100%' : '50%',
              }}
            />
          )}

          {/* Animated shimmer for running */}
          {status === 'running' && (
            <div
              className="absolute inset-0 rounded-full animate-pulse"
              style={{
                background: `linear-gradient(90deg, transparent, ${color}60, transparent)`,
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}
