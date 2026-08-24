'use client';

import React from 'react';
import { HistoryTable } from '@/components/HistoryTable';

export default function HistoryPage() {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pt-4">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-extrabold text-white">Geçmiş Analiz İnceleme Portalı</h1>
        <p className="text-sm text-slate-400">
          Bu cihaz üzerinde gerçekleştirdiğiniz geçmiş bitki hastalığı tahmin sonuçlarını inceleyin.
        </p>
      </div>

      <HistoryTable />
    </div>
  );
}
