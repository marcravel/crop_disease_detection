'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Leaf, Cpu, Layers, History, Activity, Github } from 'lucide-react';
import { getHealthStatus } from '@/services/apiService';

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [device, setDevice] = useState<string>('CPU');

  useEffect(() => {
    getHealthStatus()
      .then((res) => {
        setIsHealthy(res.model_loaded);
        setDevice(res.device);
      })
      .catch(() => setIsHealthy(false));
  }, []);

  const navLinks = [
    { href: '/', label: 'Tekli Teşhis', icon: Leaf },
    { href: '/batch', label: 'Toplu Analiz', icon: Layers },
    { href: '/history', label: 'Geçmiş Analizler', icon: History },
  ];

  return (
    <header className="sticky top-0 z-50 glass-nav">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <Link href="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-brand-400 flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform duration-200">
              <Leaf className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-xl font-bold bg-gradient-to-r from-white via-slate-200 to-brand-300 bg-clip-text text-transparent">
                AgriVision AI
              </span>
              <span className="block text-[10px] text-brand-400 font-mono tracking-wider">
                STAJ-II • ONNX RUNTIME
              </span>
            </div>
          </Link>

          {/* Nav Navigation Items */}
          <nav className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-brand-500/20 text-brand-300 border border-brand-500/30'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-800 text-xs">
              <Activity className="w-3.5 h-3.5 text-brand-400 animate-pulse" />
              <span className="text-slate-400">ONNX:</span>
              <span className="font-mono text-slate-200">{device}</span>
              <span
                className={`w-2 h-2 rounded-full ${
                  isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'
                }`}
                title={isHealthy ? 'Model Active' : 'Model Disconnected'}
              />
            </div>
            
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="hidden lg:flex items-center space-x-1 text-xs text-slate-400 hover:text-brand-300 transition-colors"
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>API Docs</span>
            </a>

            <a
              href="https://github.com/marcravel/crop_disease_detection"
              target="_blank"
              rel="noreferrer"
              className="hidden sm:flex items-center space-x-1 text-xs text-slate-400 hover:text-white transition-colors pl-1"
              title="GitHub Reposu"
            >
              <Github className="w-4 h-4 text-slate-300" />
            </a>
          </div>

        </div>
      </div>
    </header>
  );
};
