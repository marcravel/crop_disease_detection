# Crop Disease Detector — Full-Stack AI Platform (Staj-I & Staj-II)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17%2B-00599C?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14%2B-000000?style=for-the-badge&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3%2B-3178C6?style=for-the-badge&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38BDF8?style=for-the-badge&logo=tailwindcss)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker)

## 📌 Project Overview
This repository contains the complete end-to-end implementation for **Internship Part 1 (Staj-I: Model Engineering & Fine-Tuning)** and **Internship Part 2 (Staj-II: Web Application & Production Deployment)** of the **Crop Disease Detection Platform**.

The platform enables farmers and agricultural engineers to upload leaf imagery of **Tomato, Potato, and Pepper Bell** plants, delivering real-time deep learning inference (<50ms latency) via **ONNX Runtime**, top-k class probability distributions, and comprehensive agricultural treatment recommendations.

---

## 🏗️ System Architecture & Monorepo Layout

```
crop-disease-detector/
├── LOGBOOK.md                    # Staj-I Daily Logbook (Turkish)
├── LOGBOOK_STAJ2.md              # Staj-II Daily Logbook (Turkish)
├── PLAN.md                       # Staj-I Phase Plan
├── README.md                     # Main Repository Documentation
├── docker-compose.yml            # Unified Docker Deployment Manifest
├── checkpoints/                  # Staj-I Deep Learning Artifacts
│   ├── best_crop_model.pth       # PyTorch Checkpoint with Payload Metadata
│   ├── crop_disease_model.onnx   # Production Model served by ONNX Runtime
│   └── crop_disease_model.pt     # TorchScript Traced Export
├── backend/                      # Staj-II FastAPI REST API Service
│   ├── main.py                   # FastAPI Application Entrypoint & CORS setup
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile                # Backend Docker Container
│   ├── app/
│   │   ├── config.py             # Global configurations & paths
│   │   ├── schemas/              # Pydantic Request/Response Data Models
│   │   ├── services/
│   │   │   ├── onnx_service.py   # ONNX Runtime Inference Session & Preprocessing
│   │   │   └── disease_db.py     # 15-Class Agricultural Knowledge Base
│   │   └── api/v1/               # REST API Endpoints (/predict, /batch, /health, /disease)
│   └── tests/
│       └── test_predict_api.py   # Pytest Integration Suite
└── frontend/                     # Staj-II Next.js / TypeScript Web App
    ├── package.json
    ├── tailwind.config.js
    ├── Dockerfile                # Frontend Docker Container
    └── src/
        ├── app/                  # Next.js App Router Pages (Home, Batch, History)
        ├── components/           # UI Components (Navbar, Uploader, Results, Treatment, History)
        └── services/             # Axios API Client (`apiService.ts`)
```

---

## 🚀 Quick Start & Execution Guide

### Option 1: Docker Compose Deployment (Recommended)
Launch the entire system (FastAPI backend + Next.js frontend) with a single command:

```bash
# Build and run containerized stack
docker compose up --build
```
- **Frontend App:** `http://localhost:3000`
- **Backend API & Swagger Docs:** `http://localhost:8000/docs`

---

### Option 2: Local Development Setup

#### 1. Backend Setup (FastAPI + ONNX Runtime)
```bash
# Navigate to repository root
cd crop-disease-detector

# Activate Python virtual environment
source venv/bin/activate

# Run FastAPI backend via Uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Run Backend Pytest Integration Suite
```bash
PYTHONPATH=. pytest backend/tests/test_predict_api.py
```

#### 3. Frontend Setup (Next.js + TypeScript + Tailwind)
```bash
# Navigate to frontend folder
cd frontend

# Install npm dependencies
npm install

# Start Next.js development server
npm run dev
```

---

## 📊 Quantitative Benchmarks & Results

### 1. Lab Model Performance (PlantVillage Test Set)
- **Overall Accuracy:** `99.27%`
- **Macro Average F1-Score:** `0.99`
- **Weighted Average F1-Score:** `0.99`

### 2. Real-World Field Adaptation (PlantDoc Benchmark)
- **Zero-Shot Baseline Accuracy:** `15.69%`
- **Post Fine-Tuning Accuracy:** `22.55%`
- **Domain Adaptation Gain:** `+6.86%`

---

## 📝 Internship Daily Logbooks
- **Staj-I Logbook (Days 1–20):** [LOGBOOK.md](file:///home/anjaravel/Code/crop-disease-detector/LOGBOOK.md)
- **Staj-II Logbook (Days 1–20):** [LOGBOOK_STAJ2.md](file:///home/anjaravel/Code/crop-disease-detector/LOGBOOK_STAJ2.md)
