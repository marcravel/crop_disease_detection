'use client';

import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, X, AlertCircle, Sparkles } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  isLoading?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, isLoading = false }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];

    if (!file.type.startsWith('image/')) {
      setErrorMessage('Lütfen geçerli bir görsel dosyası seçin (JPG, PNG, JPEG).');
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setErrorMessage('Görsel boyutu 15MB\'tan küçük olmalıdır.');
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onImageSelected(selectedFile);
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {!previewUrl ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`glass-card p-10 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer text-center flex flex-col items-center justify-center space-y-4 ${
            dragActive
              ? 'border-brand-400 bg-brand-500/10 glow-emerald scale-[1.01]'
              : 'border-slate-700/80 hover:border-brand-500/50 hover:bg-slate-850/50'
          }`}
        >
          <div className="w-16 h-16 rounded-2xl bg-slate-800/80 flex items-center justify-center border border-slate-700 group-hover:scale-110 transition-transform">
            <UploadCloud className="w-8 h-8 text-brand-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">
              Bitki Yaprağı Fotoğrafı Yükleyin veya Sürükleyin
            </h3>
            <p className="text-sm text-slate-400 mt-1">
              Desteklenen Formatlar: JPG, JPEG, PNG (Maks 15MB)
            </p>
          </div>
        </div>
      ) : (
        <div className="glass-card p-6 rounded-2xl space-y-6">
          <div className="relative rounded-xl overflow-hidden bg-slate-950 max-h-80 flex items-center justify-center border border-slate-800">
            <img
              src={previewUrl}
              alt="Uploaded Leaf Preview"
              className="max-h-80 w-auto object-contain rounded-lg"
            />
            <button
              onClick={clearSelection}
              className="absolute top-3 right-3 p-2 rounded-full bg-slate-900/80 text-slate-300 hover:text-white hover:bg-rose-600 transition-colors"
              title="Görseli Kaldır"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-2 text-sm text-slate-300">
              <ImageIcon className="w-4 h-4 text-brand-400" />
              <span className="font-mono truncate max-w-xs">{selectedFile?.name}</span>
              <span className="text-xs text-slate-500">
                ({((selectedFile?.size || 0) / 1024 / 1024).toFixed(2)} MB)
              </span>
            </div>

            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white font-medium shadow-lg hover:shadow-brand-500/20 disabled:opacity-50 transition-all duration-200"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>ONNX Çıkarımı Yapılıyor...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Yaprağı Analiz Et</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="mt-4 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center space-x-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
