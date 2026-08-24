import axios from 'axios';
import { SinglePredictionResponse, BatchPredictionResponse, HealthStatus, DiseaseDetail } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Accept': 'application/json',
  },
});

export const getHealthStatus = async (): Promise<HealthStatus> => {
  const response = await apiClient.get<HealthStatus>('/health');
  return response.data;
};

export const predictSingleImage = async (file: File, topK: number = 3): Promise<SinglePredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<SinglePredictionResponse>(`/predict?top_k=${topK}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const predictBatchImages = async (files: File[], topK: number = 3): Promise<BatchPredictionResponse> => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await apiClient.post<BatchPredictionResponse>(`/predict-batch?top_k=${topK}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getDiseaseDetail = async (className: string): Promise<DiseaseDetail> => {
  const response = await apiClient.get<DiseaseDetail>(`/disease/${className}`);
  return response.data;
};
