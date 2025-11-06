import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Play, Square, RefreshCw, Trash2, Clock } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface EvaluationTask {
  task_id: string;
  experiment_id: string;
  eval_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_stage: string;
  result?: {
    accuracy: number;
    f1: number;
    precision: number;
    recall: number;
  };
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress_data?: {
    batch?: number;
    total_batches?: number;
    samples_processed?: number;
    current_accuracy?: number;
    current_loss?: number;
    teacher_accuracy?: number;
    student_accuracy?: number;
    prediction_agreement?: number;
    avg_kl_divergence?: number;
  };
}

interface EvaluationMonitorProps {
  experimentId: string;
  onComplete?: (result: any) => void;
}

export const EvaluationMonitor: React.FC<EvaluationMonitorProps> = ({
  experimentId,
  onComplete,
}) => {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const { lastMessage, isConnected } = useWebSocket('ws://localhost:8765/ws');

  // Fetch all tasks
  const fetchTasks = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8765/api/evaluation/tasks');
      if (response.ok) {
        const data = await response.json();
        setTasks(data.tasks || []);
      }
    } catch (error) {
      console.error('Error fetching tasks:', error);
    }
  }, []);

  // Start evaluation
  const startEvaluation = async (evalType: string = 'standard') => {
    setIsStarting(true);
    try {
      const response = await fetch('http://localhost:8765/api/evaluation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_id: experimentId,
          eval_type: evalType,
        }),
      });

      if (response.ok) {
        await fetchTasks();
      } else {
        const error = await response.json();
        console.error('Failed to start evaluation:', error);
      }
    } catch (error) {
      console.error('Error starting evaluation:', error);
    } finally {
      setIsStarting(false);
    }
  };

  // Cancel evaluation
  const cancelEvaluation = async (taskId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8765/api/evaluation/task/${taskId}/cancel`,
        { method: 'POST' }
      );

      if (response.ok) {
        await fetchTasks();
      }
    } catch (error) {
      console.error('Error cancelling evaluation:', error);
    }
  };

  // Cleanup old tasks
  const cleanupTasks = async () => {
    try {
      const response = await fetch(
        'http://localhost:8765/api/evaluation/tasks/cleanup?max_age_hours=24&keep_last_n=10',
        { method: 'DELETE' }
      );

      if (response.ok) {
        await fetchTasks();
      }
    } catch (error) {
      console.error('Error cleaning up tasks:', error);
    }
  };

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);

      if (data.type === 'evaluation_progress' && data.task_id) {
        // Update task progress in real-time
        setTasks((prev) =>
          prev.map((task) =>
            task.task_id === data.task_id
              ? {
                  ...task,
                  progress: data.progress || task.progress,
                  current_stage: data.stage || task.current_stage,
                  progress_data: { ...task.progress_data, ...data },
                }
              : task
          )
        );
      } else if (data.type === 'evaluation_completed' && data.task_id) {
        // Evaluation completed
        fetchTasks();
        if (onComplete) {
          onComplete(data.result);
        }
      }
    }
  }, [lastMessage, fetchTasks, onComplete]);

  // Poll for updates (backup to WebSocket)
  useEffect(() => {
    fetchTasks();
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, [fetchTasks]);

  // Get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      case 'cancelled':
        return 'bg-gray-500';
      default:
        return 'bg-yellow-500';
    }
  };

  // Format elapsed time
  const formatElapsed = (startedAt?: string, completedAt?: string) => {
    if (!startedAt) return '-';
    const start = new Date(startedAt).getTime();
    const end = completedAt ? new Date(completedAt).getTime() : Date.now();
    const seconds = Math.floor((end - start) / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  // Render task card
  const renderTask = (task: EvaluationTask) => {
    const isRunning = task.status === 'running';
    const progressData = task.progress_data;

    return (
      <Card key={task.task_id} className="mb-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-mono">
                {task.task_id.slice(0, 8)}...
              </CardTitle>
              <Badge className={getStatusColor(task.status)}>
                {task.status}
              </Badge>
              <Badge variant="outline">{task.eval_type}</Badge>
            </div>
            <div className="flex items-center gap-2">
              {isRunning && (
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => cancelEvaluation(task.task_id)}
                >
                  <Square className="h-4 w-4" />
                </Button>
              )}
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                {formatElapsed(task.started_at, task.completed_at)}
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {/* Progress bar */}
          {isRunning && (
            <div className="space-y-2">
              <Progress value={task.progress} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{task.current_stage}</span>
                <span>{task.progress.toFixed(1)}%</span>
              </div>
            </div>
          )}

          {/* Live metrics */}
          {isRunning && progressData && (
            <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
              {progressData.batch !== undefined && (
                <div>
                  <span className="text-muted-foreground">Batch:</span>{' '}
                  <span className="font-mono">
                    {progressData.batch}/{progressData.total_batches}
                  </span>
                </div>
              )}
              {progressData.samples_processed !== undefined && (
                <div>
                  <span className="text-muted-foreground">Samples:</span>{' '}
                  <span className="font-mono">{progressData.samples_processed}</span>
                </div>
              )}
              {progressData.current_accuracy !== undefined && (
                <div>
                  <span className="text-muted-foreground">Accuracy:</span>{' '}
                  <span className="font-mono">
                    {(progressData.current_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {progressData.current_loss !== undefined && (
                <div>
                  <span className="text-muted-foreground">Loss:</span>{' '}
                  <span className="font-mono">
                    {progressData.current_loss.toFixed(4)}
                  </span>
                </div>
              )}
              {progressData.teacher_accuracy !== undefined && (
                <div>
                  <span className="text-muted-foreground">Teacher:</span>{' '}
                  <span className="font-mono">
                    {(progressData.teacher_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {progressData.student_accuracy !== undefined && (
                <div>
                  <span className="text-muted-foreground">Student:</span>{' '}
                  <span className="font-mono">
                    {(progressData.student_accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {progressData.prediction_agreement !== undefined && (
                <div>
                  <span className="text-muted-foreground">Agreement:</span>{' '}
                  <span className="font-mono">
                    {(progressData.prediction_agreement * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {progressData.avg_kl_divergence !== undefined && (
                <div>
                  <span className="text-muted-foreground">KL Div:</span>{' '}
                  <span className="font-mono">
                    {progressData.avg_kl_divergence.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Final results */}
          {task.status === 'completed' && task.result && (
            <div className="mt-3 grid grid-cols-2 gap-3 rounded-md bg-green-50 p-3 text-sm">
              <div>
                <span className="text-muted-foreground">Accuracy:</span>{' '}
                <span className="font-semibold">
                  {(task.result.accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">F1:</span>{' '}
                <span className="font-semibold">
                  {(task.result.f1 * 100).toFixed(2)}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Precision:</span>{' '}
                <span className="font-semibold">
                  {(task.result.precision * 100).toFixed(2)}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Recall:</span>{' '}
                <span className="font-semibold">
                  {(task.result.recall * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          )}

          {/* Error message */}
          {task.status === 'failed' && task.error && (
            <div className="mt-3 rounded-md bg-red-50 p-3 text-sm text-red-900">
              <span className="font-semibold">Error:</span> {task.error}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const runningTasks = tasks.filter((t) => t.status === 'running');
  const recentTasks = tasks.filter((t) => t.status !== 'running').slice(0, 5);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Evaluation Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <Button
              onClick={() => startEvaluation('standard')}
              disabled={isStarting || runningTasks.length >= 2}
            >
              <Play className="mr-2 h-4 w-4" />
              Standard Eval
            </Button>
            <Button
              onClick={() => startEvaluation('dual')}
              disabled={isStarting || runningTasks.length >= 2}
              variant="outline"
            >
              <Play className="mr-2 h-4 w-4" />
              Dual Eval
            </Button>
            <Button onClick={fetchTasks} variant="outline" size="icon">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button onClick={cleanupTasks} variant="outline" size="icon">
              <Trash2 className="h-4 w-4" />
            </Button>
            {!isConnected && (
              <Badge variant="destructive" className="ml-auto">
                WebSocket Disconnected
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Running tasks */}
      {runningTasks.length > 0 && (
        <div>
          <h3 className="mb-2 text-lg font-semibold">Running Evaluations</h3>
          {runningTasks.map(renderTask)}
        </div>
      )}

      {/* Recent tasks */}
      {recentTasks.length > 0 && (
        <div>
          <h3 className="mb-2 text-lg font-semibold">Recent Evaluations</h3>
          {recentTasks.map(renderTask)}
        </div>
      )}

      {/* No tasks */}
      {tasks.length === 0 && (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No evaluations yet. Start one to see real-time progress!
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default EvaluationMonitor;
