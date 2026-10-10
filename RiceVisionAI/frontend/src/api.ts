const API_URL = import.meta.env.VITE_API_URL || '';
const BASE = API_URL ? `${API_URL.replace(/\/+$/, '')}/api` : '/api';

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Backend unavailable');
  return res.json();
}

export async function fetchClasses() {
  const res = await fetch(`${BASE}/classes`);
  if (!res.ok) throw new Error('Failed to load classes');
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error('Failed to load model data');
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to load metrics');
  return res.json();
}

export async function predictImage(file: File, model: string) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/predict?model=${model}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

