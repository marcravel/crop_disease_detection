import React from 'react';
import { Leaf, Code, ExternalLink } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-20 border-t border-slate-800/80 bg-slate-950/60 py-8 text-slate-400 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        
        <div className="flex items-center space-x-2">
          <Leaf className="w-4 h-4 text-brand-400" />
          <span className="font-semibold text-slate-200">Crop Disease Detector</span>
          <span>— Staj-II Model Deployment & Web Platform</span>
        </div>

        <div className="flex items-center space-x-6 text-slate-400">
          <span className="flex items-center space-x-1">
            <Code className="w-3.5 h-3.5 text-brand-400" />
            <span>FastAPI + Next.js + ONNX Runtime</span>
          </span>
          <a
            href="https://github.com/marcravel/crop_disease_detection"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1 hover:text-brand-300 transition-colors"
          >
            <span>GitHub Reposu</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>

      </div>
    </footer>
  );
};
