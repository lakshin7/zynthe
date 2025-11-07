import { useState } from 'react';

const sizeClasses = {
  small: 'w-full h-[140px] p-6',
  medium: 'w-full h-[420px] p-8',
  large: 'w-full h-[420px] p-8',
  side: 'w-full h-[140px] p-6',
};

interface GlassCardProps {
  children: React.ReactNode;
  size?: 'small' | 'medium' | 'large' | 'side';
  className?: string;
}

export function GlassCard({ children, size = 'small', className = '' }: GlassCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={`
        relative rounded-xl backdrop-blur-sm
        bg-white/90 dark:bg-slate-800/95
        border border-slate-200/50 dark:border-slate-700/50
        shadow-md dark:shadow-xl
        transition-all duration-200 ease-out
        ${isHovered ? 'shadow-lg dark:shadow-2xl scale-[1.01]' : ''}
        ${sizeClasses[size]}
        ${className}
      `}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
    </div>
  );
}
