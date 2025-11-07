interface LogRowProps {
  level: 'success' | 'warning' | 'info' | 'error';
  timestamp: string;
  message: string;
}

const levelColors = {
  success: 'bg-emerald-500 dark:bg-cyan-400',
  warning: 'bg-amber-500 dark:bg-amber-400',
  info: 'bg-blue-500 dark:bg-blue-400',
  error: 'bg-rose-500 dark:bg-fuchsia-400',
};

export function LogRow({ level, timestamp, message }: LogRowProps) {
  return (
    <div className="flex gap-3 p-3 rounded-lg bg-white/30 dark:bg-slate-700/30 border border-white/10 dark:border-white/5 hover:bg-white/50 dark:hover:bg-slate-700/50 transition-all duration-200">
      {/* Level indicator dot */}
      <div className={`w-2 h-2 rounded-full mt-1.5 ${levelColors[level]} flex-shrink-0`} />
      
      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="text-slate-400 dark:text-slate-500">{timestamp}</div>
        <div className="text-slate-700 dark:text-slate-200 mt-0.5">{message}</div>
      </div>
    </div>
  );
}
