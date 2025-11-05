import { ReactNode } from 'react';

interface BadgeProps {
  status: 'running' | 'queued' | 'completed' | 'failed';
  children: ReactNode;
  size?: 'sm' | 'md';
  pulse?: boolean;
}

export function StatusBadge({ status, children, size = 'md', pulse = false }: BadgeProps) {
  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
  };
  
  const statusStyles = {
    running: 'badge-running',
    queued: 'badge-queued',
    completed: 'badge-completed',
    failed: 'badge-failed',
  };
  
  const pulseClass = pulse && status === 'running' ? 'animate-pulse-slow' : '';
  
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${statusStyles[status]} ${sizeStyles[size]} ${pulseClass}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full bg-current ${pulse ? 'animate-pulse' : ''}`} />
      {children}
    </span>
  );
}
