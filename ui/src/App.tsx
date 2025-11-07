import { useState, useEffect } from 'react';
import { ThemeProvider } from './components/ThemeProvider';
import { ToastProvider } from './components/Toast';
import { TopNav } from './components/TopNav';
import { Sidebar } from './components/Sidebar';
import { DashboardGrid } from './components/DashboardGrid';
import { ProjectsPage } from './components/ProjectsPage';
import { PreflightPage } from './components/PreflightPage';
import { DistillationPage } from './components/DistillationPage';
import { SettingsModal } from './components/SettingsModal';
import { StatusBar } from './components/StatusBar';

export default function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [activePage, setActivePage] = useState('projects');
  const [currentTrainingStage, setCurrentTrainingStage] = useState<string | null>(null);
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null);

  // Listen to WebSocket for live training updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onopen = () => {
      console.log('App WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'training_started') {
        setActiveExperimentId(data.experiment_id);
        setCurrentTrainingStage('preflight');
        setActivePage('preflight');
      } else if (data.type === 'training_metrics' && data.experiment_id === activeExperimentId) {
        const stage = data.metrics?.stage?.toLowerCase();
        if (stage && stage !== currentTrainingStage) {
          setCurrentTrainingStage(stage);
          // Auto-navigate to the current stage
          if (['preflight', 'distillation', 'quantization', 'evaluation', 'deployment'].includes(stage)) {
            setActivePage(stage);
          }
        }
      } else if (data.type === 'training_update' && data.experiment_id === activeExperimentId) {
        if (data.status === 'completed' || data.status === 'failed') {
          setActiveExperimentId(null);
          setCurrentTrainingStage(null);
          setActivePage('projects');
        }
      }
    };
    
    ws.onerror = (error) => {
      console.error('App WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, [activeExperimentId, currentTrainingStage]);

  return (
    <ThemeProvider>
      <ToastProvider>
        <div className="min-h-screen w-full flex items-center justify-center p-8 bg-animated-pastel">
          {/* Subtle ambient glow */}
          <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-30">
            <div className="absolute -top-1/4 -left-1/4 w-1/2 h-1/2 bg-gradient-to-br from-purple-300/20 to-transparent rounded-full blur-3xl animate-pulse" />
            <div className="absolute -bottom-1/4 -right-1/4 w-1/2 h-1/2 bg-gradient-to-tl from-blue-300/20 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-1/3 bg-gradient-to-r from-pink-300/10 to-transparent rounded-full blur-2xl animate-pulse" style={{ animationDelay: '2s' }} />
          </div>

          {/* Main app window */}
          <div className="relative w-full max-w-[1600px] h-[920px] rounded-3xl overflow-hidden shadow-pastel-xl glass-pastel border-2 border-gradient-pastel transition-smooth">
            {/* Top navbar */}
            <TopNav onSettingsClick={() => setShowSettings(true)} activePage={activePage} />

            {/* Main layout */}
            <div className="flex h-[calc(100%-96px)]">
              {/* Left sidebar */}
              <Sidebar 
                activeItem={activePage} 
                onNavigate={setActivePage}
                currentTrainingStage={currentTrainingStage}
              />

              {/* Main content area */}
              <main className="flex-1 overflow-hidden bg-gradient-to-br from-white/50 to-blue-50/30 dark:from-slate-800/50 dark:to-slate-900/50">
                {activePage === 'projects' ? (
                  <ProjectsPage />
                ) : activePage === 'preflight' ? (
                  <PreflightPage />
                ) : activePage === 'distillation' ? (
                  <DistillationPage />
                ) : (
                  <DashboardGrid />
                )}
              </main>
            </div>

            {/* Status bar */}
            <StatusBar />
          </div>

          {/* Settings modal */}
          {showSettings && (
            <SettingsModal onClose={() => setShowSettings(false)} />
          )}
        </div>
      </ToastProvider>
    </ThemeProvider>
  );
}
