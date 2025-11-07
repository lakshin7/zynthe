interface KPIChipProps {
  children: React.ReactNode;
  status?: 'success' | 'warning' | 'info' | 'error';
}

const statusStyles = {
  success: 'bg-emerald-500/10 dark:bg-cyan-500/15 border-emerald-400/30 dark:border-cyan-400/30 text-emerald-700 dark:text-cyan-300',
  warning: 'bg-amber-500/10 dark:bg-amber-500/15 border-amber-400/30 dark:border-amber-400/30 text-amber-700 dark:text-amber-300',
  info: 'bg-blue-500/10 dark:bg-blue-500/15 border-blue-400/30 dark:border-blue-400/30 text-blue-700 dark:text-blue-300',
  error: 'bg-rose-500/10 dark:bg-fuchsia-500/15 border-rose-400/30 dark:border-fuchsia-400/30 text-rose-700 dark:text-fuchsia-300',
};

export function KPIChip({ children, status = 'info' }: KPIChipProps) {
  return (
    <span
      className={`
        inline-flex items-center px-3 py-1 rounded-full 
        border backdrop-blur-sm transition-all duration-200
        ${statusStyles[status]}
      `}
    >
      {children}
    </span>
  );
}
