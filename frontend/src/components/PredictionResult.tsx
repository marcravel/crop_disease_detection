'use client';

import React from 'react';
import { SinglePredictionResponse } from '@/types';
import { CheckCircle2, AlertTriangle, Clock, BarChart3, Award } from 'lucide-react';

interface PredictionResultProps {
  result: SinglePredictionResponse;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
  const { top_prediction, top_k_predictions, disease_info, latency_ms } = result;
  const isHealthy = disease_info?.is_healthy ?? top_prediction.class_name.toLowerCase().includes('healthy');

  return (
    <div className="glass-card p-6 md:p-8 rounded-2xl space-y-6">
      
      {/* Top Prediction Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center space-x-3">
            <span
              className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center space-x-1.5 ${
                isHealthy
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
              }`}
            >
              {isHealthy ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>SAĞLIKLI BİTKİ</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>HASTALIK TESPİT EDİLDİ ({disease_info?.severity || 'Dikkat'})</span>
                </>
              )}
            </span>

            <span className="flex items-center space-x-1 text-xs text-slate-400 bg-slate-900 px-2.5 py-1 rounded-full border border-slate-800 font-mono">
              <Clock className="w-3 h-3 text-brand-400" />
              <span>{latency_ms} ms</span>
            </span>
          </div>

          <h2 className="text-2xl font-bold text-white pt-2">
            {disease_info?.name_tr || top_prediction.class_name.replace(/_/g, ' ')}
          </h2>
          <p className="text-sm text-slate-400 italic">
            {disease_info?.name_en || top_prediction.class_name}
          </p>
        </div>

        {/* Confidence Percentage Badge */}
        <div className="flex flex-col items-center justify-center p-4 rounded-xl bg-slate-900/80 border border-slate-800 min-w-[140px]">
          <span className="text-3xl font-extrabold text-brand-400 font-mono">
            %{top_prediction.confidence.toFixed(1)}
          </span>
          <span className="text-[11px] text-slate-400 uppercase tracking-wider font-medium mt-0.5">
            Model Güveni
          </span>
        </div>
      </div>

      {/* Top-K Probability Breakdown Progress Bars */}
      <div className="space-y-4 pt-2">
        <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
          <BarChart3 className="w-4 h-4 text-brand-400" />
          <span>Olasılık Dağılımı (Top-K Tahminler)</span>
        </div>

        <div className="space-y-3">
          {top_k_predictions.map((item, index) => {
            const isTop1 = index === 0;
            return (
              <div key={item.class_idx} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className={`font-medium ${isTop1 ? 'text-brand-300' : 'text-slate-400'}`}>
                    {index + 1}. {item.class_name.replace(/_/g, ' ')}
                  </span>
                  <span className="font-mono text-slate-300 font-semibold">
                    %{item.confidence.toFixed(2)}
                  </span>
                </div>
                <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800/60">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isTop1
                        ? 'bg-gradient-to-r from-brand-600 to-brand-400 shadow-[0_0_12px_#34d399]'
                        : 'bg-slate-700'
                    }`}
                    style={{ width: `${Math.max(item.confidence, 2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
};
