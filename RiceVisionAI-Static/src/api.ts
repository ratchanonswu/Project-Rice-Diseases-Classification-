import { STATIC_DATA } from './staticData';
import type { HealthResponse, ModelsResponse, MetricsResponse, ClassInfo } from './types';

export async function fetchHealth(): Promise<HealthResponse> {
  return {
    status: 'ok',
    resnet50_available: true,
    mobilenetv2_available: true,
    resnet50_loaded: false,
    mobilenetv2_loaded: false,
  };
}

export async function fetchClasses(): Promise<ClassInfo[]> {
  return STATIC_DATA.classes as unknown as ClassInfo[];
}

export async function fetchModels(): Promise<ModelsResponse> {
  return STATIC_DATA.models as unknown as ModelsResponse;
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return STATIC_DATA.metrics as unknown as MetricsResponse;
}

export async function predictImage(_file: File, _model: string) {
  throw new Error('Live model inference is disabled in this static research showcase. View the project locally to run real-time predictions.');
}
