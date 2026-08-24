'use client';

import React, { useState } from 'react';
import { BarChart3, ShieldCheck, Zap, ArrowUpRight, TrendingUp, Layers, CheckCircle2, FileText, Image as ImageIcon } from 'lucide-react';

const PER_CLASS_METRICS = [
  { class_name: 'Pepper__bell___Bacterial_spot', precision: 1.00, recall: 1.00, f1: 1.00, support: 84 },
  { class_name: 'Pepper__bell___healthy', precision: 1.00, recall: 1.00, f1: 1.00, support: 158 },
  { class_name: 'Potato___Early_blight', precision: 1.00, recall: 0.98, f1: 0.99, support: 106 },
  { class_name: 'Potato___Late_blight', precision: 1.00, recall: 0.97, f1: 0.99, support: 118 },
  { class_name: 'Potato___healthy', precision: 1.00, recall: 1.00, f1: 1.00, support: 18 },
  { class_name: 'Tomato_Bacterial_spot', precision: 1.00, recall: 0.99, f1: 1.00, support: 207 },
  { class_name: 'Tomato_Early_blight', precision: 0.97, recall: 0.96, f1: 0.97, support: 109 },
  { class_name: 'Tomato_Late_blight', precision: 0.97, recall: 0.99, f1: 0.98, support: 186 },
  { class_name: 'Tomato_Leaf_Mold', precision: 1.00, recall: 1.00, f1: 1.00, support: 93 },
  { class_name: 'Tomato_Septoria_leaf_spot', precision: 0.99, recall: 1.00, f1: 1.00, support: 177 },
  { class_name: 'Tomato_Spider_mites_Two_spotted_spider_mite', precision: 1.00, recall: 0.99, f1: 1.00, support: 174 },
  { class_name: 'Tomato__Target_Spot', precision: 0.98, recall: 1.00, f1: 0.99, support: 132 },
  { class_name: 'Tomato__Tomato_YellowLeaf__Curl_Virus', precision: 0.99, recall: 1.00, f1: 1.00, support: 331 },
  { class_name: 'Tomato__Tomato_mosaic_virus', precision: 0.97, recall: 1.00, f1: 0.99, support: 33 },
  { class_name: 'Tomato_healthy', precision: 1.00, recall: 1.00, f1: 1.00, support: 139 },
];

export default function MetricsPage() {
  const [activeTab, setActiveTab] = useState<'table' | 'charts'>('table');

  return (
    <div className="space-y-10 max-w-6xl mx-auto pt-4">
      
      {/* Header Banner */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/30 text-brand-300 text-xs font-semibold">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>DENEYSEL SONUÇLAR VE PERFORMANS METRİKLERİ</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">
          Model Başarım ve Saha Adaptasyon Analizi
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl mx-auto leading-relaxed">
          Staj-I kapsamında eğitilen ResNet-18 modelinin PlantVillage laboratuvar başarımları ve PlantDoc gerçek saha verileri üzerindeki etki değerlendirmesi.
        </p>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="glass-card p-5 rounded-2xl space-y-2 border-emerald-500/30">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>PlantVillage Doğruluğu</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold text-white font-mono">%96.13</div>
          <div className="text-[11px] text-emerald-400 font-medium">Saha Simülasyonlu Model (2,065 İmaj)</div>
        </div>

        <div className="glass-card p-5 rounded-2xl space-y-2 border-amber-500/30">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Eski Sıfır-Vuruş</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-extrabold text-slate-400 font-mono">%15.69</div>
          <div className="text-[11px] text-slate-400">Çoğullamasız İlk Taban Başarım</div>
        </div>

        <div className="glass-card p-5 rounded-2xl space-y-2 border-brand-500/30">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Yeni Sıfır-Vuruş (Saha)</span>
            <TrendingUp className="w-4 h-4 text-brand-400" />
          </div>
          <div className="text-3xl font-extrabold text-brand-300 font-mono">%26.47</div>
          <div className="text-[11px] text-brand-400 font-medium">Saha Çoğullamalı Sıfır-Vuruş</div>
        </div>

        <div className="glass-card p-5 rounded-2xl space-y-2 border-teal-500/30">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Saha Genelletirme Artışı</span>
            <ArrowUpRight className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-3xl font-extrabold text-teal-300 font-mono">+%10.78</div>
          <div className="text-[11px] text-teal-400 font-medium">Net Sıfır-Vuruş Artışı (Delta)</div>
        </div>

      </div>

      {/* Domain Shift Explanation Card */}
      <div className="glass-card p-6 md:p-8 rounded-2xl space-y-6">
        <div className="flex items-start space-x-3">
          <div className="p-3 rounded-xl bg-brand-500/20 text-brand-400 flex-shrink-0">
            <Layers className="w-6 h-6" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-bold text-white">Laboratuvardan Sahaya Genelletirme Engeli (Domain Shift)</h3>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
              Laboratuvar ortamında toplanan <b>PlantVillage</b> veri setinde model yüksek doğruluk elde ederken, gerçek tarla ortamında çekilen <b>PlantDoc</b> veri setinde başlangıç başarımı <b>%15.69</b> seviyesinde kalmıştır. 
              Saha şartlarını simüle eden veri çoğullama boru hattı (ColorJitter, RandomCrop, Rotation, Cutout), Erken Durdurma (Early Stopping - val_loss), L2 Weight Decay (1e-4) ve Dropout (p=0.3) uygulanması sonucunda modelin laboratuvar stüdyo arka planı ezberlemesi engellenmiş ve <b>PlantDoc sıfır-vuruş (zero-shot) başarımı %15.69'dan %26.47'ye (+%10.78 net artış) yükseltilmiştir</b>.
            </p>
          </div>
        </div>

        {/* Progress Bar Comparison */}
        <div className="space-y-4 pt-2 border-t border-slate-800">
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-semibold text-slate-300">
              <span>PlantVillage Test Doğruluğu (Laboratuvar + Saha Çoğullamalı)</span>
              <span className="font-mono text-emerald-400">%96.13</span>
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div className="h-full bg-emerald-500 rounded-full" style={{ width: '96.13%' }} />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-semibold text-slate-300">
              <span>Eski PlantDoc Sıfır-Vuruş Başarımı (Çoğullamasız)</span>
              <span className="font-mono text-amber-400">%15.69</span>
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div className="h-full bg-amber-500 rounded-full" style={{ width: '15.69%' }} />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-semibold text-slate-300">
              <span>Yeni PlantDoc Sıfır-Vuruş Başarımı (Saha Çoğullamalı + Erken Durdurmalı Model)</span>
              <span className="font-mono text-brand-300">%26.47</span>
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div className="h-full bg-gradient-to-r from-brand-600 to-brand-400 rounded-full" style={{ width: '26.47%' }} />
            </div>
          </div>
        </div>
      </div>

      {/* Tabs View Selector */}
      <div className="space-y-6">
        <div className="flex space-x-3 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('table')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
              activeTab === 'table'
                ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/20'
                : 'bg-slate-900/80 text-slate-400 hover:text-white'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Sınıf Bazlı Metrik Tablosu (PlantVillage)</span>
          </button>

          <button
            onClick={() => setActiveTab('charts')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
              activeTab === 'charts'
                ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/20'
                : 'bg-slate-900/80 text-slate-400 hover:text-white'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            <span>Karmaşıklık Matrisi ve Eğitim Eğrileri</span>
          </button>
        </div>

        {activeTab === 'table' ? (
          <div className="glass-card p-6 rounded-2xl overflow-x-auto space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white">15 Sınıf İçin Precision, Recall ve F1 Skorları</h3>
              <span className="text-xs font-mono text-slate-400">Toplam Test Örneği: 2,065</span>
            </div>

            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/80 text-slate-400 uppercase font-mono border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Sınıf Adı</th>
                  <th className="py-3 px-4 text-center">Precision (Hassasiyet)</th>
                  <th className="py-3 px-4 text-center">Recall (Duyarlılık)</th>
                  <th className="py-3 px-4 text-center">F1-Score</th>
                  <th className="py-3 px-4 text-right">Destek (Support)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {PER_CLASS_METRICS.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-850/50 transition-colors">
                    <td className="py-3 px-4 font-sans font-medium text-white">{row.class_name.replace(/_/g, ' ')}</td>
                    <td className="py-3 px-4 text-center text-emerald-400 font-bold">{(row.precision * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-center text-emerald-400 font-bold">{(row.recall * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-center text-brand-300 font-bold">{(row.f1 * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-right text-slate-400">{row.support}</td>
                  </tr>
                ))}
                <tr className="bg-slate-900/90 font-bold text-white">
                  <td className="py-3.5 px-4 font-sans">Ağırlıklı Ortalama (Weighted Average)</td>
                  <td className="py-3.5 px-4 text-center text-emerald-400">99.3%</td>
                  <td className="py-3.5 px-4 text-center text-emerald-400">99.3%</td>
                  <td className="py-3.5 px-4 text-center text-brand-300">99.3%</td>
                  <td className="py-3.5 px-4 text-right text-slate-300">2,065</td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-6 rounded-2xl space-y-4 text-center">
              <h3 className="text-base font-bold text-white">15 Sınıflı Karmaşıklık Matrisi (Confusion Matrix)</h3>
              <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 p-2">
                <img
                  src="/results/confusion_matrix.png"
                  alt="Confusion Matrix"
                  className="w-full h-auto object-contain rounded-lg"
                />
              </div>
              <p className="text-xs text-slate-400">
                ResNet-18 modelinin 15 hastalık sınıfı arasındaki doğru ve yanlış tahmin dağılımı.
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl space-y-4 text-center">
              <h3 className="text-base font-bold text-white">Eğitim & Doğrulama Başarım Eğrileri</h3>
              <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 p-2">
                <img
                  src="/results/learning_curves.png"
                  alt="Learning Curves"
                  className="w-full h-auto object-contain rounded-lg"
                />
              </div>
              <p className="text-xs text-slate-400">
                15 epoch boyunca Eğitim Kaybı (Train Loss) ve Doğrulama Başarımı (Validation Accuracy) gelişimi.
              </p>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
