'use client';

import React, { useState } from 'react';
import { DiseaseDetail } from '@/types';
import { ShieldAlert, Sprout, FlaskConical, ShieldCheck, FileText } from 'lucide-react';

interface DiseaseDetailCardProps {
  info: DiseaseDetail;
}

export const DiseaseDetailCard: React.FC<DiseaseDetailCardProps> = ({ info }) => {
  const [activeTab, setActiveTab] = useState<'symptoms' | 'organic' | 'chemical' | 'prevention'>('symptoms');

  if (info.is_healthy) {
    return (
      <div className="glass-card p-6 md:p-8 rounded-2xl space-y-4 border-emerald-500/30">
        <div className="flex items-center space-x-3 text-emerald-400">
          <ShieldCheck className="w-7 h-7" />
          <h3 className="text-xl font-bold text-white">Sağlıklı Bitki Bakımı Tavsiyeleri</h3>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed">
          {info.description}
        </p>
        <div className="pt-2">
          <h4 className="text-sm font-semibold text-brand-300 mb-2">Önerilen Koruyucu Tedbirler:</h4>
          <ul className="list-disc list-inside text-xs text-slate-300 space-y-1.5">
            {info.prevention.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'symptoms', label: 'Belirtiler', icon: FileText, count: info.symptoms.length },
    { id: 'organic', label: 'Organik Tedavi', icon: Sprout, count: info.organic_treatment.length },
    { id: 'chemical', label: 'Kimyasal İlaç', icon: FlaskConical, count: info.chemical_treatment.length },
    { id: 'prevention', label: 'Önlemler', icon: ShieldAlert, count: info.prevention.length },
  ] as const;

  return (
    <div className="glass-card p-6 md:p-8 rounded-2xl space-y-6">
      
      <div>
        <h3 className="text-xl font-bold text-white mb-1">Tarımsal Mücadele & Tedavi Rehberi</h3>
        <p className="text-xs text-slate-400">
          {info.name_tr} için uzman ziraat önerileri ve müdahale yöntemleri.
        </p>
      </div>

      {/* Tabs Header */}
      <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200 ${
                isActive
                  ? 'bg-brand-500 text-white shadow-lg shadow-brand-500/20'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
              {tab.count > 0 && (
                <span className={`px-1.5 py-0.5 rounded-full text-[10px] ${
                  isActive ? 'bg-brand-600 text-white' : 'bg-slate-800 text-slate-400'
                }`}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="pt-2">
        {activeTab === 'symptoms' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400">{info.description}</p>
            <ul className="space-y-2">
              {info.symptoms.map((item, idx) => (
                <li key={idx} className="flex items-start space-x-3 text-sm text-slate-200 bg-slate-900/50 p-3 rounded-xl border border-slate-800/60">
                  <span className="w-2 h-2 rounded-full bg-brand-400 mt-1.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {activeTab === 'organic' && (
          <div className="space-y-3">
            <ul className="space-y-2">
              {info.organic_treatment.map((item, idx) => (
                <li key={idx} className="flex items-start space-x-3 text-sm text-emerald-300 bg-emerald-950/30 p-3 rounded-xl border border-emerald-900/40">
                  <Sprout className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {activeTab === 'chemical' && (
          <div className="space-y-3">
            <ul className="space-y-2">
              {info.chemical_treatment.map((item, idx) => (
                <li key={idx} className="flex items-start space-x-3 text-sm text-amber-300 bg-amber-950/30 p-3 rounded-xl border border-amber-900/40">
                  <FlaskConical className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {activeTab === 'prevention' && (
          <div className="space-y-3">
            <ul className="space-y-2">
              {info.prevention.map((item, idx) => (
                <li key={idx} className="flex items-start space-x-3 text-sm text-slate-200 bg-slate-900/50 p-3 rounded-xl border border-slate-800/60">
                  <ShieldCheck className="w-4 h-4 text-brand-400 mt-0.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

    </div>
  );
};
