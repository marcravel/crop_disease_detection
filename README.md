# Crop Disease Detector — Full-Stack AI Platform (Staj-I & Staj-II)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17%2B-00599C?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14%2B-000000?style=for-the-badge&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3%2B-3178C6?style=for-the-badge&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38BDF8?style=for-the-badge&logo=tailwindcss)

## 📌 Project Overview
This repository contains the complete end-to-end implementation for **Internship Part 1 (Staj-I: Model Engineering & Fine-Tuning)** and **Internship Part 2 (Staj-II: Web Application & Production Deployment)** of the **Crop Disease Detection Platform**.

The platform enables farmers and agricultural engineers to upload leaf imagery of **Tomato, Potato, and Pepper Bell** plants, delivering real-time deep learning inference (<50ms latency) via **ONNX Runtime**, top-k class probability distributions, and comprehensive agricultural treatment recommendations.

---

## 🚀 Quick Start & Execution Guide

### Option 1: Native Execution (Recommended — Fast & No Sudo Required)

#### Terminal 1: Launch FastAPI Backend (ONNX Runtime)
```bash
# From repository root directory
source venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API & Swagger Docs:** `http://localhost:8000/docs`

#### Terminal 2: Launch Next.js Frontend App
```bash
# Navigate to frontend folder
cd frontend
npm run dev
```
- **Web App Dashboard:** `http://localhost:3000`

---

### Option 2: Docker Execution

If running Docker on Linux, ensure user permissions or use `sudo`:

```bash
# 1. Build and run Backend container
sudo docker build -t crop_disease_backend -f backend/Dockerfile .
sudo docker run -d -p 8000:8000 --name crop_backend crop_disease_backend

# 2. Build and run Frontend container
sudo docker build -t crop_disease_frontend -f frontend/Dockerfile .
sudo docker run -d -p 3000:3000 --name crop_frontend crop_disease_frontend
```

*Note: To install `docker compose` plugin on Ubuntu:*
```bash
sudo apt-get update && sudo apt-get install -y docker-compose-v2
```

---

## 🧪 Run Backend Automated Tests
```bash
PYTHONPATH=. pytest backend/tests/test_predict_api.py
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
