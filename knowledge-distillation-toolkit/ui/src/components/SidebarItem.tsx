import { LucideIcon } from 'lucide-react';

interface SidebarItemProps {
  label: string;
  icon: LucideIcon;
  isActive: boolean;
  onClick: () => void;
  badge?: string;
}

export function SidebarItem({ label, icon: Icon, isActive, onClick, badge }: SidebarItemProps) {
  return (
    <button
      onClick={onClick}
      className={`
        w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-smooth relative
        ${isActive
          ? 'pastel-gradient-blue shadow-pastel-md border-2 border-purple-200 dark:border-purple-400/50 hover-lift'
          : 'glass-pastel hover:bg-gradient-to-r hover:from-purple-50 hover:to-blue-50 dark:hover:from-purple-900/20 dark:hover:to-blue-900/20 border border-transparent'
        }
      `}
    >
      <div className={`p-1.5 rounded-lg ${isActive ? 'bg-white/50 dark:bg-slate-700/50' : ''}`}>
        <Icon
          className={`w-5 h-5 transition-smooth ${
            isActive
              ? 'text-purple-600 dark:text-purple-400'
              : 'text-slate-600 dark:text-slate-300'
          }`}
          strokeWidth={isActive ? 2.5 : 2}
        />
      </div>
      <span
        className={`transition-smooth font-semibold text-sm ${
          isActive
            ? 'text-purple-800 dark:text-purple-200'
            : 'text-slate-700 dark:text-slate-300'
        }`}
      >
        {label}
      </span>
      {badge === 'live' && (
        <span className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 px-2 py-1 rounded-full pastel-gradient-green shadow-pastel-sm">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 pulse-pastel" />
          <span className="text-[10px] font-bold text-green-800 dark:text-green-200">LIVE</span>
        </span>
      )}
    </button>
  );
}
