// API types shared across the app

export interface HealthResponse {
  status: string;
  resnet50_available: boolean;
  mobilenetv2_available: boolean;
  resnet50_loaded: boolean;
  mobilenetv2_loaded: boolean;
}

export interface ClassInfo {
  name: string;
  index: number;
  description: string;
  image: string | null;
}

export interface ModelMetrics {
  model: string;
  test_sample_count: number;
  accuracy: number;
  precision_macro: number;
  precision_weighted: number;
  recall_macro: number;
  recall_weighted: number;
  f1_score_macro: number;
  f1_score_weighted: number;
  roc_auc_macro: number;
  roc_auc_weighted: number;
}

export interface InferenceLatency {
  profiled_samples: number;
  average_inference_time_ms: number;
  median_inference_time_ms: number;
  std_inference_time_ms: number;
  throughput_fps: number;
}

export interface ModelEfficiency {
  model_architecture: string;
  total_parameters: number;
  trainable_parameters: number;
  model_file_size_mb: number;
  training_time_seconds: number;
  training_time_minutes: number;
  inference_latency: InferenceLatency;
}

export interface ModelData {
  available: boolean;
  metrics: ModelMetrics;
  efficiency: ModelEfficiency;
}

export interface ModelsResponse {
  resnet50: ModelData;
  mobilenetv2: ModelData;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface MetricsResponse {
  resnet50: {
    per_class: PerClassMetric[];
    confusion_matrix: Record<string, Record<string, number>>;
  };
  mobilenetv2: {
    per_class: PerClassMetric[];
    confusion_matrix: Record<string, Record<string, number>>;
  };
  class_names: string[];
}

export interface PredictResponse {
  model: string;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  scores: Record<string, number>;
  inference_ms: number;
  disclaimer: string;
}

