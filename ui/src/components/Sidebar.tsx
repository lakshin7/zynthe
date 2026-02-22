import { SidebarItem } from './SidebarItem';
import {
  LayoutDashboard,
  FileCheck,
  Droplets,
  Minimize2,
  BarChart3,
  Rocket
} from 'lucide-react';

const menuItems = [
  { id: 'projects', label: 'Projects', icon: LayoutDashboard },
  { id: 'preflight', label: 'Preflight', icon: FileCheck },
  { id: 'distillation', label: 'Distillation', icon: Droplets },
  { id: 'quantization', label: 'Quantization', icon: Minimize2 },
  { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
  { id: 'deployment', label: 'Deployment', icon: Rocket },
];

interface SidebarProps {
  activeItem: string;
  onNavigate: (itemId: string) => void;
  currentTrainingStage?: string | null;
}

export function Sidebar({ activeItem, onNavigate, currentTrainingStage }: SidebarProps) {
  return (
    <aside className="w-64 border-r-2 border-gradient-pastel glass-pastel p-6 shadow-pastel-sm">
      <nav className="space-y-2">
        {menuItems.map((item) => {
          const isTrainingStage = currentTrainingStage === item.id;
          return (
            <SidebarItem
              key={item.id}
              {...item}
              isActive={activeItem === item.id}
              onClick={() => onNavigate(item.id)}
              badge={isTrainingStage ? 'live' : undefined}
            />
          );
        })}
      </nav>

      {/* Cloud Launch Section */}
      <div className="mt-8 pt-6 border-t border-border-light/50">
        <div className="px-3 mb-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
          Cloud Runtime
        </div>
        <a
          href="https://colab.research.google.com/github/lakshin7/zynthe/blob/main/notebooks/Zynthe_Cloud_Backend.ipynb"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 text-text-secondary hover:bg-white hover:text-primary hover:shadow-pastel group"
        >
          <div className="p-1.5 rounded-lg bg-orange-100 text-orange-600 group-hover:bg-primary group-hover:text-white transition-colors">
            <Rocket className="w-4 h-4" />
          </div>
          <span>Launch Cloud (free GPU)</span>
        </a>
      </div>
    </aside >
  );
}
