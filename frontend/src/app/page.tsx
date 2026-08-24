'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ImageUploader } from '@/components/ImageUploader';
import { PredictionResult } from '@/components/PredictionResult';
import { DiseaseDetailCard } from '@/components/DiseaseDetailCard';
import { predictSingleImage } from '@/services/apiService';
import { SinglePredictionResponse, SavedHistoryItem } from '@/types';
import { Sparkles, ShieldCheck, Zap, Layers } from 'lucide-react';

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SinglePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageAnalysis = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await predictSingleImage(file, 3);
      setResult(data);

      // Save to localStorage history
      const savedHistory = localStorage.getItem('agrivision_history');
      const historyList: SavedHistoryItem[] = savedHistory ? JSON.parse(savedHistory) : [];
      const newHistoryItem: SavedHistoryItem = {
        ...data,
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
      };
      historyList.unshift(newHistoryItem);
      localStorage.setItem('agrivision_history', JSON.stringify(historyList.slice(0, 20)));
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Hastalık analizi sırasında bir bağlantı hatası oluştu.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-10">
      
      {/* Hero Banner Header */}
      <div className="text-center space-y-4 max-w-3xl mx-auto pt-4">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/30 text-brand-300 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-brand-400" />
          <span>ONNX RUNTIME ÇIKARIM MOTORU İLE DESTEKLENMEKTEDİR</span>
        </div>
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
          Yapay Zeka ile <span className="bg-gradient-to-r from-brand-400 via-emerald-300 to-teal-200 bg-clip-text text-transparent">Bitki Hastalık Teşhisi</span>
        </h1>
        <p className="text-sm sm:text-base text-slate-300">
          Domates, Patates ve Biber yaprak fotoğraflarınızı yükleyin; anında yüksek doğruluklu teşhis, top-k olasılık dağılımı ve uzman ziraat tedavi rehberine ulaşın.
        </p>
      </div>

      {/* Stats Feature Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-4xl mx-auto">
        <div className="glass-card p-4 rounded-xl flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-brand-500/20 text-brand-400 flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-bold text-white">&lt;50 ms Çıkarım</div>
            <div className="text-[11px] text-slate-400">ONNX Hızlı Tahmin</div>
          </div>
        </div>

        <Link href="/metrics" className="glass-card p-4 rounded-xl flex items-center space-x-3 hover:border-brand-500/50 transition-all cursor-pointer group">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0 group-hover:scale-105 transition-transform">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-bold text-white flex items-center space-x-1">
              <span>%99.27 Lab / %22.55 Saha</span>
            </div>
            <div className="text-[11px] text-brand-300">Model Metriklerini İncele →</div>
          </div>
        </Link>

        <div className="glass-card p-4 rounded-xl flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center flex-shrink-0">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-bold text-white">15 Tarımsal Sınıf</div>
            <div className="text-[11px] text-slate-400">Domates, Patates, Biber</div>
          </div>
        </div>
      </div>

      {/* Main Upload Zone */}
      <div className="max-w-3xl mx-auto">
        <ImageUploader onImageSelected={handleImageAnalysis} isLoading={isLoading} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="max-w-3xl mx-auto p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm text-center">
          {error}
        </div>
      )}

      {/* Results & Treatment Layout */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-4">
          <PredictionResult result={result} />
          {result.disease_info && <DiseaseDetailCard info={result.disease_info} />}
        </div>
      )}

    </div>
  );
}
