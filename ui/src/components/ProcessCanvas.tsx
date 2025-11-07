import { ArrowRight, Database, Cpu, Minimize2, CheckCircle2 } from 'lucide-react';

const steps = [
  { 
    id: 1, 
    label: 'Source Model', 
    icon: Database, 
    status: 'complete',
    color: 'text-amber-600 dark:text-amber-400'
  },
  { 
    id: 2, 
    label: 'Distillation', 
    icon: Cpu, 
    status: 'complete',
    color: 'text-rose-600 dark:text-fuchsia-400'
  },
  { 
    id: 3, 
    label: 'Quantization', 
    icon: Minimize2, 
    status: 'active',
    color: 'text-emerald-600 dark:text-cyan-400'
  },
  { 
    id: 4, 
    label: 'Optimized', 
    icon: CheckCircle2, 
    status: 'pending',
    color: 'text-slate-400 dark:text-slate-500'
  },
];

export function ProcessCanvas() {
  return (
    <div className="space-y-4 h-full flex flex-col">
      <div>
        <h3 className="text-slate-900 dark:text-white mb-1">Compression Pipeline</h3>
        <p className="text-slate-500 dark:text-slate-400">ResNet-50 optimization in progress</p>
      </div>

      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-8">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center gap-8">
              {/* Step node */}
              <div className="flex flex-col items-center gap-3">
                <div
                  className={`
                    w-20 h-20 rounded-2xl flex items-center justify-center
                    backdrop-blur-xl border transition-all duration-300
                    ${step.status === 'active' 
                      ? 'bg-white/70 dark:bg-slate-700/70 border-emerald-400/50 dark:border-cyan-400/50 shadow-lg shadow-emerald-500/20 dark:shadow-cyan-500/30' 
                      : step.status === 'complete'
                      ? 'bg-white/50 dark:bg-slate-700/50 border-white/30 dark:border-white/10'
                      : 'bg-white/30 dark:bg-slate-700/30 border-white/20 dark:border-white/5'
                    }
                  `}
                >
                  <step.icon className={`w-10 h-10 ${step.color}`} strokeWidth={1.5} />
                </div>
                <div className={`text-center ${step.status === 'pending' ? 'text-slate-400 dark:text-slate-500' : 'text-slate-700 dark:text-slate-300'}`}>
                  {step.label}
                </div>
                {step.status === 'active' && (
                  <div className="flex items-center gap-2 text-emerald-600 dark:text-cyan-400">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-cyan-400" />
                    <span>Processing</span>
                  </div>
                )}
              </div>

              {/* Arrow connector */}
              {index < steps.length - 1 && (
                <ArrowRight className="w-8 h-8 text-slate-300 dark:text-slate-600" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-slate-600 dark:text-slate-400">
          <span>Overall Progress</span>
          <span>67%</span>
        </div>
        <div className="h-2 bg-white/30 dark:bg-slate-700/30 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-amber-400 via-rose-400 to-emerald-400 dark:from-cyan-400 dark:via-fuchsia-400 dark:to-amber-400 rounded-full transition-all duration-500"
            style={{ width: '67%' }}
          />
        </div>
      </div>
    </div>
  );
}
