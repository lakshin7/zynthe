import { LogRow } from './LogRow';

const mockLogs = [
  {
    id: 1,
    level: 'info',
    timestamp: '14:23:45',
    message: 'Model compression initiated for ResNet-50',
  },
  {
    id: 2,
    level: 'success',
    timestamp: '14:24:12',
    message: 'Distillation phase completed successfully',
  },
  {
    id: 3,
    level: 'info',
    timestamp: '14:24:45',
    message: 'Quantization pass 1/3 in progress',
  },
  {
    id: 4,
    level: 'warning',
    timestamp: '14:25:08',
    message: 'Accuracy dip detected, adjusting parameters',
  },
  {
    id: 5,
    level: 'success',
    timestamp: '14:26:30',
    message: 'Quantization complete - 87% size reduction',
  },
  {
    id: 6,
    level: 'info',
    timestamp: '14:27:15',
    message: 'Running validation suite',
  },
  {
    id: 7,
    level: 'success',
    timestamp: '14:28:42',
    message: 'Validation passed - 95.8% accuracy retained',
  },
];

export function LogPanel() {
  return (
    <div className="space-y-4 h-full flex flex-col">
      <div>
        <h3 className="text-slate-900 dark:text-white mb-1">Activity Log</h3>
        <p className="text-slate-500 dark:text-slate-400">Real-time process updates</p>
      </div>

      <div className="flex-1 space-y-2 overflow-auto pr-2">
        {mockLogs.map((log) => (
          <LogRow key={log.id} {...log} />
        ))}
      </div>
    </div>
  );
}
