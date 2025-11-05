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
    </aside>
  );
}
