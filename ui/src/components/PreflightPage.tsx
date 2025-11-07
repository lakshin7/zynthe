import { useState, useEffect } from 'react';
import { FileCheck, CheckCircle2, AlertCircle, Loader2, Database, Cpu, HardDrive, Zap } from 'lucide-react';
import { GlassCard } from './GlassCard';

interface PreflightCheck {
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  message: string;
  details?: string;
}

export function PreflightPage() {
  const [checks, setChecks] = useState<PreflightCheck[]>([
    { name: 'Dataset Validation', status: 'pending', message: 'Waiting to start...' },
    { name: 'Model Compatibility', status: 'pending', message: 'Waiting to start...' },
    { name: 'Resource Availability', status: 'pending', message: 'Waiting to start...' },
    { name: 'Configuration Validation', status: 'pending', message: 'Waiting to start...' },
  ]);
  const [overallStatus, setOverallStatus] = useState<'pending' | 'running' | 'completed' | 'failed'>('pending');

  useEffect(() => {
    // Listen to WebSocket for preflight updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'training_log' && data.message) {
        const message = data.message.toLowerCase();
        
        // Update checks based on log messages
        if (message.includes('preflight') || message.includes('checking')) {
          setOverallStatus('running');
          
          if (message.includes('dataset')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Dataset Validation' 
                ? { ...c, status: 'running', message: 'Validating dataset...' }
                : c
            ));
          } else if (message.includes('model')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Model Compatibility' 
                ? { ...c, status: 'running', message: 'Checking model compatibility...' }
                : c
            ));
          } else if (message.includes('resource') || message.includes('memory') || message.includes('gpu')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Resource Availability' 
                ? { ...c, status: 'running', message: 'Checking available resources...' }
                : c
            ));
          } else if (message.includes('config')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Configuration Validation' 
                ? { ...c, status: 'running', message: 'Validating configuration...' }
                : c
            ));
          }
        }
        
        // Check for completion
        if (message.includes('passed') || message.includes('success') || message.includes('✓')) {
          if (message.includes('dataset')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Dataset Validation' 
                ? { ...c, status: 'passed', message: 'Dataset validated successfully' }
                : c
            ));
          } else if (message.includes('model')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Model Compatibility' 
                ? { ...c, status: 'passed', message: 'Model compatible' }
                : c
            ));
          } else if (message.includes('resource')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Resource Availability' 
                ? { ...c, status: 'passed', message: 'Resources available' }
                : c
            ));
          } else if (message.includes('config')) {
            setChecks(prev => prev.map(c => 
              c.name === 'Configuration Validation' 
                ? { ...c, status: 'passed', message: 'Configuration valid' }
                : c
            ));
          }
        }
        
        // Check if preflight completed
        if (message.includes('preflight complete') || message.includes('preflight passed')) {
          setOverallStatus('completed');
          setChecks(prev => prev.map(c => ({ ...c, status: c.status === 'pending' ? 'passed' : c.status })));
        }
      }
    };
    
    return () => ws.close();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-cyan-500 animate-spin" />;
      case 'passed':
        return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-rose-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-slate-300 dark:border-slate-600" />;
    }
  };

  const getCheckIcon = (name: string) => {
    if (name.includes('Dataset')) return Database;
    if (name.includes('Model')) return Cpu;
    if (name.includes('Resource')) return HardDrive;
    if (name.includes('Configuration')) return Zap;
    return FileCheck;
  };

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <FileCheck className="w-8 h-8 text-amber-600 dark:text-cyan-400" />
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
            Preflight Checks
          </h1>
        </div>
        <p className="text-slate-600 dark:text-slate-300">
          Validating environment and configuration before training begins
        </p>
      </div>

      {/* Overall Status */}
      <GlassCard>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {overallStatus === 'running' && <Loader2 className="w-6 h-6 text-cyan-500 animate-spin" />}
            {overallStatus === 'completed' && <CheckCircle2 className="w-6 h-6 text-emerald-500" />}
            {overallStatus === 'pending' && <FileCheck className="w-6 h-6 text-slate-400 dark:text-slate-500" />}
            <div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {overallStatus === 'running' && 'Running Preflight Checks...'}
                {overallStatus === 'completed' && 'All Checks Passed!'}
                {overallStatus === 'pending' && 'Waiting to Start'}
                {overallStatus === 'failed' && 'Preflight Failed'}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-300">
                {checks.filter(c => c.status === 'passed').length} / {checks.length} checks completed
              </div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Individual Checks */}
      <div className="grid grid-cols-1 gap-4">
        {checks.map((check, index) => {
          const Icon = getCheckIcon(check.name);
          return (
            <GlassCard key={index}>
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-white/50 dark:bg-slate-700/50">
                  <Icon className="w-6 h-6 text-amber-600 dark:text-cyan-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                      {check.name}
                    </h3>
                    {getStatusIcon(check.status)}
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    {check.message}
                  </p>
                  {check.details && (
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                      {check.details}
                    </p>
                  )}
                </div>
              </div>
            </GlassCard>
          );
        })}
      </div>
    </div>
  );
}
