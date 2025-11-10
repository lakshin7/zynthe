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

// ==================== NEW ARTIFACT/LIVE TRAINING HELPERS ====================

export async function fetchArtifacts(expId: string) {
  const r = await fetch(`${API_BASE}/api/experiments/${expId}/artifacts`);
  if (!r.ok) throw new Error('Failed to fetch artifacts');
  return r.json();
}

export async function fetchConfusion(expId: string, role: 'teacher' | 'student') {
  const r = await fetch(`${API_BASE}/api/experiments/${expId}/confusion/${role}`);
  if (!r.ok) throw new Error('Confusion matrix not available');
  return r.json();
}

export async function fetchBatchLog(expId: string) {
  const r = await fetch(`${API_BASE}/api/experiments/${expId}/batch-log`);
  if (!r.ok) throw new Error('Batch log not found');
  return r.json();
}

export async function fetchMicroSeries(expId: string, role: 'teacher' | 'student', epoch: number) {
  const r = await fetch(`${API_BASE}/api/experiments/${expId}/micro/${role}/${epoch}`);
  if (!r.ok) throw new Error('Micro-series not found');
  return r.json();
}

export function parseCsv(text: string) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const cols = line.split(',');
    const row: Record<string, string> = {};
    headers.forEach((h, i) => { row[h] = (cols[i] || '').trim(); });
    return row;
  });
}

export interface LiveBatchEvent {
  type: string;
  experiment_id?: string;
  role?: 'teacher' | 'student';
  phase?: 'train' | 'eval' | string;
  batch_idx?: number;
  loss?: number;
  grad_norm?: number;
  lr?: number;
  accuracy?: number;
  epoch?: number;
}

export function connectLiveTraining(onEvent: (ev: LiveBatchEvent) => void) {
  const ws = new WebSocket(`ws://localhost:8765/ws`);
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch (e) {
      console.warn('Live WS parse error', e);
    }
  };
  return ws;
}
