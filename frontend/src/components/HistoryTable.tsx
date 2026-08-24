'use client';

import React, { useEffect, useState } from 'react';
import { SavedHistoryItem } from '@/types';
import { History, Trash2, Clock, CheckCircle2, AlertTriangle, ExternalLink } from 'lucide-react';

export const HistoryTable: React.FC = () => {
  const [history, setHistory] = useState<SavedHistoryItem[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('agrivision_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to parse history from localStorage', e);
      }
    }
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('agrivision_history');
    setHistory([]);
  };

  if (history.length === 0) {
    return (
      <div className="glass-card p-12 rounded-2xl text-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center mx-auto text-slate-500">
          <History className="w-8 h-8" />
        </div>
        <h3 className="text-lg font-bold text-white">Henüz Geçmiş Analiz Kaydı Bulunmuyor</h3>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Bitki yaprağı fotoğrafları yükleyip analiz yaptığınızda geçmiş sonuçlarınız burada görüntülenecektir.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 rounded-2xl space-y-6">
      
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <History className="w-5 h-5 text-brand-400" />
          <h3 className="text-lg font-bold text-white">Geçmiş Teşhis Kayıtları ({history.length})</h3>
        </div>
        <button
          onClick={clearHistory}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 hover:bg-rose-500/20 text-xs font-semibold transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Geçmişi Temizle</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-900/80 text-slate-400 uppercase font-mono border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Tarih</th>
              <th className="py-3 px-4">Dosya Adı</th>
              <th className="py-3 px-4">Teşhis</th>
              <th className="py-3 px-4 text-center">Güven</th>
              <th className="py-3 px-4 text-right">Gecikme</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {history.map((item) => {
              const isHealthy = item.disease_info?.is_healthy ?? item.top_prediction.class_name.toLowerCase().includes('healthy');
              return (
                <tr key={item.id} className="hover:bg-slate-850/50 transition-colors">
                  <td className="py-3.5 px-4 font-mono text-slate-400 whitespace-nowrap">
                    {item.timestamp}
                  </td>
                  <td className="py-3.5 px-4 font-medium text-white max-w-[150px] truncate">
                    {item.filename}
                  </td>
                  <td className="py-3.5 px-4">
                    <span className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full font-medium text-[11px] ${
                      isHealthy
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}>
                      {isHealthy ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                      <span>{item.disease_info?.name_tr || item.top_prediction.class_name}</span>
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-center font-mono font-bold text-brand-300">
                    %{item.top_prediction.confidence.toFixed(1)}
                  </td>
                  <td className="py-3.5 px-4 text-right font-mono text-slate-400 whitespace-nowrap">
                    {item.latency_ms} ms
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

    </div>
  );
};
