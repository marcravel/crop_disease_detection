export interface PredictionItem {
  class_idx: number;
  class_name: string;
  confidence: number;
}

export interface DiseaseDetail {
  disease_id: string;
  name_tr: string;
  name_en: string;
  crop_type: string;
  is_healthy: boolean;
  severity: string;
  description: string;
  symptoms: string[];
  organic_treatment: string[];
  chemical_treatment: string[];
  prevention: string[];
}

export interface SinglePredictionResponse {
  status: string;
  filename: string;
  top_prediction: PredictionItem;
  top_k_predictions: PredictionItem[];
  disease_info?: DiseaseDetail;
  latency_ms: number;
}

export interface BatchPredictionResponse {
  status: string;
  total_images: number;
  predictions: SinglePredictionResponse[];
  total_latency_ms: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  model_loaded: boolean;
  model_path: string;
  device: string;
}

export interface SavedHistoryItem extends SinglePredictionResponse {
  id: string;
  timestamp: string;
  imageUrl?: string;
}
