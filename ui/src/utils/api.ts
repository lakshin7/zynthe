/**
 * API utilities for Zynthe backend
 */

const API_BASE = 'http://localhost:8765/api';

export const api = {
  // Experiments
  async getExperiments() {
    const response = await fetch(`${API_BASE}/experiments`);
    if (!response.ok) throw new Error('Failed to fetch experiments');
    return response.json();
  },

  async getExperiment(expId: string) {
    const response = await fetch(`${API_BASE}/experiments/${expId}`);
    if (!response.ok) throw new Error('Failed to fetch experiment');
    return response.json();
  },

  // Training
  async createTraining(config: any) {
    const response = await fetch(`${API_BASE}/training/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error('Failed to create training');
    return response.json();
  },

  async getActiveTraining() {
    const response = await fetch(`${API_BASE}/training/active`);
    if (!response.ok) throw new Error('Failed to fetch active training');
    return response.json();
  },

  async getTrainingMetrics(expId: string) {
    const response = await fetch(`${API_BASE}/training/${expId}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch metrics');
    return response.json();
  },

  async pauseTraining(expId: string) {
    const response = await fetch(`${API_BASE}/training/${expId}/pause`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to pause training');
    return response.json();
  },

  async resumeTraining(expId: string) {
    const response = await fetch(`${API_BASE}/training/${expId}/resume`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to resume training');
    return response.json();
  },

  async stopTraining(expId: string) {
    const response = await fetch(`${API_BASE}/training/${expId}/stop`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to stop training');
    return response.json();
  },

  async saveCheckpoint(expId: string) {
    const response = await fetch(`${API_BASE}/training/${expId}/checkpoint`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to save checkpoint');
    return response.json();
  },

  // Datasets
  async getDatasets() {
    const response = await fetch(`${API_BASE}/datasets`);
    if (!response.ok) throw new Error('Failed to fetch datasets');
    return response.json();
  },

  async uploadDataset(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/dataset/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Failed to upload dataset');
    return response.json();
  },

  async deleteDataset(datasetId: string) {
    const response = await fetch(`${API_BASE}/dataset/${datasetId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete dataset');
    return response.json();
  },

  // Models
  async getModels() {
    const response = await fetch(`${API_BASE}/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    return response.json();
  },

  async compareModels(modelIds: string[]) {
    const response = await fetch(`${API_BASE}/models/compare?model_ids=${modelIds.join(',')}`);
    if (!response.ok) throw new Error('Failed to compare models');
    return response.json();
  },

  // Metrics
  async getMetrics() {
    const response = await fetch(`${API_BASE}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch metrics');
    return response.json();
  },

  // Downloads
  async downloadFile(expId: string, filename: string) {
    const response = await fetch(`${API_BASE}/download/${expId}/${filename}`);
    if (!response.ok) throw new Error('Failed to download file');
    const blob = await response.blob();
    
    // Trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },
};

// WebSocket helper
export function createWebSocket(onMessage: (data: any) => void) {
  const ws = new WebSocket('ws://localhost:8765/ws');
  
  ws.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };
  
  return ws;
}
