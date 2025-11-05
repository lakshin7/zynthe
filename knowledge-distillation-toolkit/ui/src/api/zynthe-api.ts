/**
 * Zynthe API Client
 * Connects to the FastAPI backend
 */

const API_BASE = 'http://localhost:8765';

export interface ExperimentInfo {
  id: string;
  title: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  created: string;
  last_updated: string;
  stages: any[];
  cpu_usage: number;
  gpu_usage: number;
}

export async function getExperiments() {
  try {
    const response = await fetch(`${API_BASE}/api/experiments`);
    const data = await response.json();
    return data.experiments || [];
  } catch (error) {
    console.error('API Error:', error);
    return [];
  }
}

export async function getMetrics() {
  try {
    const response = await fetch(`${API_BASE}/api/metrics`);
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    return {
      compression_rate: 0,
      accuracy_retention: 0,
      inference_speed: 0,
      compression_data: [],
      accuracy_data: [],
      speed_data: []
    };
  }
}

export async function startTraining(config: any) {
  const response = await fetch(`${API_BASE}/api/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  return await response.json();
}

export function connectWebSocket(onMessage: (data: any) => void) {
  const ws = new WebSocket(`ws://localhost:8765/ws`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  return ws;
}
