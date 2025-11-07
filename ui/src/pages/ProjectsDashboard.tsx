import { useState, useEffect } from 'react';
import { Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, StatusBadge, ProgressBar } from '../components/base';

interface Experiment {
  id: string;
  name: string;
  timestamp: string;
  status: 'running' | 'queued' | 'completed' | 'failed';
  stages: {
    preflight: string;
    distillation: string;
    quantization: string;
    evaluation: string;
    deployment: string;
  };
  is_active?: boolean;
  progress?: number;
  teacher?: string;
  student?: string;
  dataset?: string;
  metrics?: {
    loss?: number;
    accuracy?: number;
  };
}

export function ProjectsDashboard() {
  const navigate = useNavigate();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'running' | 'queued' | 'completed' | 'failed'>('all');

  // Fetch experiments from API
  useEffect(() => {
    fetchExperiments();
    
    // Setup WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8765/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_started' || data.type === 'training_update') {
        fetchExperiments();
      }
    };
    
    return () => ws.close();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await fetch('http://localhost:8765/api/experiments');
      const data = await response.json();
      setExperiments(data);
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredExperiments = experiments.filter(exp => {
    if (filter === 'all') return true;
    return exp.status === filter;
  });

  const stats = {
    running: experiments.filter(e => e.status === 'running').length,
    queued: experiments.filter(e => e.status === 'queued').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    failed: experiments.filter(e => e.status === 'failed').length,
  };

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="bg-white border-b border-border-light px-8 py-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-text-primary">Projects</h1>
            <p className="text-text-secondary mt-1">Manage your knowledge distillation experiments</p>
          </div>
          <Button
            variant="primary"
            size="lg"
            icon={<Plus className="w-5 h-5" />}
            onClick={() => navigate('/new-experiment')}
          >
            New Experiment
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-4 gap-4">
          <Card
            padding="md"
            className="cursor-pointer"
            onClick={() => setFilter(filter === 'running' ? 'all' : 'running')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Running</p>
                <p className="text-3xl font-bold text-status-running">{stats.running}</p>
              </div>
              <div className="w-12 h-12 rounded-full bg-status-running/20 flex items-center justify-center">
                <div className="w-6 h-6 rounded-full bg-status-running animate-pulse" />
              </div>
            </div>
          </Card>

          <Card
            padding="md"
            className="cursor-pointer"
            onClick={() => setFilter(filter === 'queued' ? 'all' : 'queued')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Queued</p>
                <p className="text-3xl font-bold text-status-queued">{stats.queued}</p>
              </div>
              <div className="w-12 h-12 rounded-full bg-status-queued/20 flex items-center justify-center">
                <div className="w-6 h-6 rounded-full bg-status-queued" />
              </div>
            </div>
          </Card>

          <Card
            padding="md"
            className="cursor-pointer"
            onClick={() => setFilter(filter === 'completed' ? 'all' : 'completed')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Completed</p>
                <p className="text-3xl font-bold text-status-completed">{stats.completed}</p>
              </div>
              <div className="w-12 h-12 rounded-full bg-status-completed/20 flex items-center justify-center">
                <svg className="w-6 h-6 text-status-completed" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
          </Card>

          <Card
            padding="md"
            className="cursor-pointer"
            onClick={() => setFilter(filter === 'failed' ? 'all' : 'failed')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-secondary text-sm">Failed</p>
                <p className="text-3xl font-bold text-status-failed">{stats.failed}</p>
              </div>
              <div className="w-12 h-12 rounded-full bg-status-failed/20 flex items-center justify-center">
                <svg className="w-6 h-6 text-status-failed" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Experiment List */}
      <div className="flex-1 overflow-auto custom-scrollbar px-8 py-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="spinner-pastel w-12 h-12" />
          </div>
        ) : filteredExperiments.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-text-muted text-lg">No {filter !== 'all' && filter} experiments found</p>
            <Button
              variant="primary"
              size="lg"
              icon={<Plus className="w-5 h-5" />}
              onClick={() => navigate('/new-experiment')}
              className="mt-4"
            >
              Create Your First Experiment
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredExperiments.map((exp) => (
              <Card
                key={exp.id}
                hover
                padding="lg"
                onClick={() => exp.is_active && navigate(`/training/${exp.id}`)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-xl font-semibold text-text-primary">{exp.name}</h3>
                      <StatusBadge status={exp.status} pulse={exp.status === 'running'}>
                        {exp.status.charAt(0).toUpperCase() + exp.status.slice(1)}
                      </StatusBadge>
                    </div>
                    <p className="text-text-secondary text-sm">
                      {new Date(exp.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    {exp.is_active && (
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/training/${exp.id}`);
                        }}
                      >
                        View Live
                      </Button>
                    )}
                    {exp.status === 'completed' && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          // Export functionality
                        }}
                      >
                        Export
                      </Button>
                    )}
                  </div>
                </div>

                {/* Progress Bar for Running/Queued */}
                {(exp.status === 'running' || exp.status === 'queued') && (
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-text-secondary">Progress</span>
                      <span className="text-sm font-semibold text-text-primary">
                        {exp.progress || 0}%
                      </span>
                    </div>
                    <ProgressBar
                      progress={exp.progress || 0}
                      status={exp.status}
                      animated
                    />
                  </div>
                )}

                {/* Metrics for Completed */}
                {exp.status === 'completed' && exp.metrics && (
                  <div className="grid grid-cols-2 gap-4 mb-4 p-3 bg-bg-tertiary rounded-lg">
                    <div>
                      <p className="text-xs text-text-secondary">Accuracy</p>
                      <p className="text-lg font-bold text-accent">
                        {exp.metrics.accuracy ? (exp.metrics.accuracy * 100).toFixed(1) : '0.0'}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">Final Loss</p>
                      <p className="text-lg font-bold text-text-primary">
                        {exp.metrics.loss?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                  </div>
                )}

                {/* Model Info */}
                <div className="flex items-center gap-4 text-sm text-text-secondary">
                  <span>Teacher: {exp.teacher || 'bert-base'}</span>
                  <span>→</span>
                  <span>Student: {exp.student || 'distilbert'}</span>
                  <span>•</span>
                  <span>Dataset: {exp.dataset || 'IMDB'}</span>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
