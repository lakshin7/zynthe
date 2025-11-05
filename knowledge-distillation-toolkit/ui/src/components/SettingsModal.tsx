import { useState, useEffect } from 'react';
import { X, Cpu, Zap, Bell, Globe, Save, RotateCcw } from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { useToast } from './Toast';

interface SettingsModalProps {
  onClose: () => void;
}

interface TrainingSettings {
  defaultDevice: 'auto' | 'cpu' | 'gpu' | 'mps';
  defaultBatchSize: number;
  defaultEpochs: number;
  defaultLearningRate: number;
  checkpointFrequency: number;
  apiEndpoint: string;
  apiTimeout: number;
  wsReconnectAttempts: number;
  wsReconnectDelay: number;
  notifyTrainingComplete: boolean;
  notifyTrainingError: boolean;
  notifyCheckpoint: boolean;
  soundEnabled: boolean;
}

const DEFAULT_TRAINING_SETTINGS: TrainingSettings = {
  defaultDevice: 'auto',
  defaultBatchSize: 16,
  defaultEpochs: 10,
  defaultLearningRate: 2e-5,
  checkpointFrequency: 1,
  apiEndpoint: 'http://localhost:8765',
  apiTimeout: 30000,
  wsReconnectAttempts: 5,
  wsReconnectDelay: 3000,
  notifyTrainingComplete: true,
  notifyTrainingError: true,
  notifyCheckpoint: false,
  soundEnabled: true,
};

export function SettingsModal({ onClose }: SettingsModalProps) {
  const { theme, setTheme } = useTheme();
  const { showToast } = useToast();

  const [activeTab, setActiveTab] = useState<'appearance' | 'training' | 'notifications'>('appearance');
  const [trainingSettings, setTrainingSettings] = useState<TrainingSettings>(DEFAULT_TRAINING_SETTINGS);
  const [hasChanges, setHasChanges] = useState(false);

  // Load training settings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('zynthe_training_settings');
    if (saved) {
      try {
        setTrainingSettings({ ...DEFAULT_TRAINING_SETTINGS, ...JSON.parse(saved) });
      } catch (e) {
        console.error('Failed to parse training settings:', e);
      }
    }
  }, []);

  const updateTrainingSetting = <K extends keyof TrainingSettings>(key: K, value: TrainingSettings[K]) => {
    setTrainingSettings(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    localStorage.setItem('zynthe_training_settings', JSON.stringify(trainingSettings));
    setHasChanges(false);
    window.dispatchEvent(new CustomEvent('settingsChanged', { detail: trainingSettings }));
    showToast('success', 'Settings saved successfully!');
  };

  const handleReset = () => {
    if (confirm('Reset training settings to defaults?')) {
      setTrainingSettings(DEFAULT_TRAINING_SETTINGS);
      setHasChanges(true);
      showToast('info', 'Settings reset to defaults');
    }
  };

  const tabs = [
    { id: 'appearance', label: 'Appearance', icon: Globe },
    { id: 'training', label: 'Training', icon: Zap },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-8">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Modal - Wider for tabs */}
      <div className="relative w-[800px] max-w-full max-h-[85vh] transition-all duration-300">
        {/* Glass card container */}
        <div className="h-full rounded-2xl backdrop-blur-sm bg-white/90 dark:bg-slate-800/95 border border-slate-200/50 dark:border-slate-700/50 shadow-2xl flex flex-col overflow-hidden">
          
          {/* Header */}
          <div className="p-6 border-b border-slate-200/50 dark:border-slate-700/50 flex items-center justify-between flex-shrink-0">
            <div>
              <h2 className="text-slate-900 dark:text-white">Settings</h2>
              <p className="text-slate-600 dark:text-slate-300 mt-1">
                Configure appearance, training, and notifications
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-all duration-200 group"
              aria-label="Close settings"
            >
              <X className="w-6 h-6 text-slate-700 dark:text-slate-200 group-hover:text-slate-900 dark:group-hover:text-white transition-colors" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 px-6 pt-4 border-b border-slate-200/50 dark:border-slate-700/50 flex-shrink-0">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-t-lg transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-white/60 dark:bg-slate-700/60 text-slate-900 dark:text-white border-b-2 border-blue-600 dark:border-blue-400'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 min-h-0">{activeTab === 'appearance' && (
            <>
            {/* Theme Mode */}
            <div className="space-y-3">
              <label className="block text-slate-700 dark:text-slate-300 font-medium">
                Theme Mode
              </label>
              <div className="flex gap-3">
                {['light', 'dark', 'auto'].map((t) => (
                  <button
                    key={t}
                    onClick={() => setTheme(t as any)}
                    className={`
                      flex-1 px-4 py-3 rounded-xl border capitalize transition-all duration-200
                      ${theme === t
                        ? 'bg-blue-600 dark:bg-blue-500 border-blue-600 dark:border-blue-500 text-white shadow-md'
                        : 'bg-white/60 dark:bg-slate-700/60 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-white dark:hover:bg-slate-700'
                      }
                    `}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            </>
          )}

          {activeTab === 'training' && (
            <>
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Default Training Parameters</h3>
                
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Device
                  </label>
                  <select
                    value={trainingSettings.defaultDevice}
                    onChange={(e) => updateTrainingSetting('defaultDevice', e.target.value as any)}
                    className="w-full px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
                  >
                    <option value="auto">Auto (Let system decide)</option>
                    <option value="cpu">CPU</option>
                    <option value="gpu">GPU (CUDA)</option>
                    <option value="mps">Apple Silicon (MPS)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={trainingSettings.defaultBatchSize}
                    onChange={(e) => updateTrainingSetting('defaultBatchSize', parseInt(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
                    min="1"
                    max="512"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={trainingSettings.defaultEpochs}
                    onChange={(e) => updateTrainingSetting('defaultEpochs', parseInt(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
                    min="1"
                    max="100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    value={trainingSettings.defaultLearningRate}
                    onChange={(e) => updateTrainingSetting('defaultLearningRate', parseFloat(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
                    min="0.000001"
                    max="0.1"
                    step="0.000001"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    API Endpoint
                  </label>
                  <input
                    type="text"
                    value={trainingSettings.apiEndpoint}
                    onChange={(e) => updateTrainingSetting('apiEndpoint', e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
                  />
                </div>
              </div>
            </>
          )}

          {activeTab === 'notifications' && (
            <>
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Training Notifications</h3>
                
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainingSettings.notifyTrainingComplete}
                    onChange={(e) => updateTrainingSetting('notifyTrainingComplete', e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 accent-blue-600 dark:accent-blue-400"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Notify when training completes
                  </span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainingSettings.notifyTrainingError}
                    onChange={(e) => updateTrainingSetting('notifyTrainingError', e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 accent-blue-600 dark:accent-blue-400"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Notify on training errors
                  </span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainingSettings.notifyCheckpoint}
                    onChange={(e) => updateTrainingSetting('notifyCheckpoint', e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 accent-blue-600 dark:accent-blue-400"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Notify on checkpoint saves
                  </span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainingSettings.soundEnabled}
                    onChange={(e) => updateTrainingSetting('soundEnabled', e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 accent-blue-600 dark:accent-blue-400"
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Enable notification sounds
                  </span>
                </label>
              </div>
            </>
          )}
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-slate-200/50 dark:border-slate-700/50 flex justify-between gap-3 flex-shrink-0">
            {activeTab !== 'appearance' && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/60 dark:bg-slate-700/60 hover:bg-white dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 transition-all duration-200"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            )}
            <div className="flex gap-3 ml-auto">
              <button
                onClick={onClose}
                className="px-6 py-2.5 rounded-xl bg-white/60 dark:bg-slate-700/60 hover:bg-white dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition-all duration-200"
              >
                {activeTab === 'appearance' ? 'Done' : 'Cancel'}
              </button>
              {activeTab !== 'appearance' && (
                <button
                  onClick={handleSave}
                  disabled={!hasChanges}
                  className={`
                    flex items-center gap-2 px-6 py-2.5 rounded-xl transition-all duration-200
                    ${hasChanges
                      ? 'bg-blue-600 dark:bg-blue-500 hover:bg-blue-700 dark:hover:bg-blue-600 text-white shadow-md'
                      : 'bg-slate-300 dark:bg-slate-600 text-slate-500 dark:text-slate-400 cursor-not-allowed'
                    }
                  `}
                >
                  <Save className="w-4 h-4" />
                  Save
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
