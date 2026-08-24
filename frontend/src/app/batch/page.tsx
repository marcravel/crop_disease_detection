'use client';

import React, { useState } from 'react';
import { predictBatchImages } from '@/services/apiService';
import { BatchPredictionResponse } from '@/types';
import { Layers, UploadCloud, CheckCircle2, AlertTriangle, Clock } from 'lucide-react';

export default function BatchPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files).filter((f) => f.type.startsWith('image/'));
      setSelectedFiles(filesArray);
      setBatchResult(null);
      setError(null);
    }
  };

  const handleBatchAnalyze = async () => {
    if (selectedFiles.length === 0) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictBatchImages(selectedFiles, 3);
      setBatchResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Toplu analiz gerçekleştirilirken hata oluştu.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      
      <div className="text-center space-y-3 pt-4">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/30 text-brand-300 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>TOPLU BİTKİ YAPRAĞI ANALİZİ</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white">Çoklu Yaprak Analiz Portalı</h1>
        <p className="text-sm text-slate-300 max-w-2xl mx-auto">
          Tarlanızdan veya seranızdan topladığınız birden fazla yaprak fotoğrafını aynı anda yükleyin ve ONNX batch çıkarım yanıtlarını saniyeler içinde inceleyin.
        </p>
      </div>

      <div className="glass-card p-8 rounded-2xl space-y-6 text-center">
        <label className="block cursor-pointer">
          <input
            type="file"
            multiple
            accept="image/jpeg,image/png,image/jpg"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="p-8 border-2 border-dashed border-slate-700 hover:border-brand-400 rounded-xl transition-colors flex flex-col items-center justify-center space-y-3">
            <UploadCloud className="w-10 h-10 text-brand-400" />
            <span className="text-base font-semibold text-white">
              Birden Fazla Fotoğraf Seçin ({selectedFiles.length} Görsel Seçildi)
            </span>
            <span className="text-xs text-slate-400">
              CTRL veya SHIFT tuşunu basılı tutarak çoklu seçim yapabilirsiniz.
            </span>
          </div>
        </label>

        {selectedFiles.length > 0 && (
          <button
            onClick={handleBatchAnalyze}
            disabled={isLoading}
            className="px-8 py-3 rounded-xl bg-brand-500 hover:bg-brand-400 text-white font-semibold shadow-lg hover:shadow-brand-500/20 transition-all disabled:opacity-50"
          >
            {isLoading ? 'Toplu Çıkarım Yapılıyor...' : `${selectedFiles.length} Adet Yaprağı Toplu Analiz Et`}
          </button>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm text-center">
          {error}
        </div>
      )}

      {batchResult && (
        <div className="space-y-6 pt-4">
          <div className="flex items-center justify-between glass-card p-4 rounded-xl">
            <span className="text-sm font-semibold text-white">
              Toplu Analiz Tamamlandı ({batchResult.total_images} Görsel)
            </span>
            <span className="flex items-center space-x-1.5 text-xs text-brand-300 font-mono">
              <Clock className="w-3.5 h-3.5" />
              <span>Toplam Süre: {batchResult.total_latency_ms} ms</span>
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {batchResult.predictions.map((pred, idx) => {
              const isHealthy = pred.disease_info?.is_healthy ?? pred.top_prediction.class_name.toLowerCase().includes('healthy');
              return (
                <div key={idx} className="glass-card p-5 rounded-xl space-y-3">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="font-mono truncate max-w-[200px]">{pred.filename}</span>
                    <span className="font-mono text-brand-400">{pred.latency_ms} ms</span>
                  </div>

                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${isHealthy ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                      {isHealthy ? <CheckCircle2 className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                    </div>
                    <div>
                      <h4 className="text-base font-bold text-white">
                        {pred.disease_info?.name_tr || pred.top_prediction.class_name}
                      </h4>
                      <span className="text-xs font-mono font-semibold text-brand-300">
                        Güven: %{pred.top_prediction.confidence.toFixed(1)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
