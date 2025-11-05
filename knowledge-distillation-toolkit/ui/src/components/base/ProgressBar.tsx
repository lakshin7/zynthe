interface ProgressBarProps {
  progress: number; // 0-100
  status?: 'running' | 'queued' | 'completed' | 'failed';
  showLabel?: boolean;
  height?: 'sm' | 'md' | 'lg';
  animated?: boolean;
}

export function ProgressBar({
  progress,
  status = 'running',
  showLabel = false,
  height = 'md',
  animated = true,
}: ProgressBarProps) {
  const heightStyles = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };
  
  const fillStyles = {
    running: 'progress-fill-running',
    queued: 'progress-fill-queued',
    completed: 'bg-status-completed',
    failed: 'bg-status-failed',
  };
  
  const clampedProgress = Math.min(100, Math.max(0, progress));
  
  return (
    <div className="w-full">
      <div className={`progress-bar ${heightStyles[height]} ${animated ? 'transition-all duration-500' : ''}`}>
        <div
          className={`${fillStyles[status]} ${heightStyles[height]} ${animated ? 'transition-all duration-500' : ''}`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
      {showLabel && (
        <div className="mt-1 text-xs text-text-secondary text-right">
          {clampedProgress}%
        </div>
      )}
    </div>
  );
}
