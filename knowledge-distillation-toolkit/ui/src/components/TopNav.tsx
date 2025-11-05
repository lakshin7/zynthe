import { Search, User, Settings } from 'lucide-react';
import { Logo } from './Logo';

interface TopNavProps {
  onSettingsClick: () => void;
  activePage?: string;
}

export function TopNav({ onSettingsClick, activePage = 'projects' }: TopNavProps) {
  const pageNames: Record<string, string> = {
    projects: 'Projects',
    preflight: 'Preflight',
    distillation: 'Distillation',
    quantization: 'Quantization',
    evaluation: 'Evaluation',
    deployment: 'Deployment',
  };

  return (
    <nav className="h-16 px-6 flex items-center justify-between glass-pastel border-b-2 border-gradient-pastel shadow-pastel-sm">
      {/* Left: Logo */}
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-200 to-blue-200 flex items-center justify-center shadow-pastel-md">
          <Logo size={24} />
        </div>
      </div>

      {/* Center: Breadcrumbs */}
      <div className="flex items-center gap-3 text-slate-700 dark:text-slate-200">
        <span className="font-bold text-lg text-gradient-pastel">Zynthe</span>
        <span className="text-purple-300 dark:text-purple-400">/</span>
        <span className="font-semibold text-sm bg-gradient-to-r from-purple-600 to-blue-600 dark:from-purple-400 dark:to-blue-400 bg-clip-text text-transparent">
          {pageNames[activePage] || 'Overview'}
        </span>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        {/* Search */}
        <button className="p-2.5 rounded-xl glass-pastel hover:bg-gradient-to-br hover:from-purple-100 hover:to-blue-100 dark:hover:from-purple-900/30 dark:hover:to-blue-900/30 transition-smooth group shadow-pastel-sm hover:shadow-pastel-md">
          <Search className="w-5 h-5 text-purple-600 dark:text-purple-300 group-hover:text-purple-700 dark:group-hover:text-purple-200 transition-colors" />
        </button>

        {/* Profile */}
        <button className="p-2.5 rounded-xl glass-pastel hover:bg-gradient-to-br hover:from-purple-100 hover:to-blue-100 dark:hover:from-purple-900/30 dark:hover:to-blue-900/30 transition-smooth group shadow-pastel-sm hover:shadow-pastel-md">
          <User className="w-5 h-5 text-purple-600 dark:text-purple-300 group-hover:text-purple-700 dark:group-hover:text-purple-200 transition-colors" />
        </button>

        {/* Settings */}
        <button
          onClick={onSettingsClick}
          className="p-2.5 rounded-xl glass-pastel hover:bg-gradient-to-br hover:from-purple-100 hover:to-blue-100 dark:hover:from-purple-900/30 dark:hover:to-blue-900/30 transition-smooth group shadow-pastel-sm hover:shadow-pastel-md"
        >
          <Settings className="w-5 h-5 text-purple-600 dark:text-purple-300 group-hover:text-purple-700 dark:group-hover:text-purple-200 transition-colors" />
        </button>
      </div>
    </nav>
  );
}
