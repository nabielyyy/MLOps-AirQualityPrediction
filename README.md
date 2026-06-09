# Air Quality Prediction MLOps Project

## Overview
Project ini menyediakan pipeline MLOps untuk prediksi kualitas udara, termasuk preprocessing data, training model, pelacakan eksperimen dengan MLflow, serta deployment inference lewat FastAPI dan Docker Compose.

## Struktur Project

```bash
.
├── config/
├── data/
├── mlruns/
├── src/
├── tests/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Teknologi yang Digunakan
- Python
- Scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Docker Compose
- GitHub Actions

## Cara Build Image
Jalankan perintah berikut di root repository:

```bash
docker compose build
```

## Cara Menjalankan Docker Compose
```bash
docker compose up -d
```

## Cara Melihat Container
```bash
docker compose ps
```

## Cara Mengakses MLflow UI
Buka browser ke:

```bash
http://localhost:5000
```

## Cara Mengakses API
API inference tersedia di:

```bash
http://localhost:8000
```

## Endpoint API
- GET /
- GET /health
- POST /predict

## Contoh Request POST /predict
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pm2_5": 25.5,
    "pm10": 40.0,
    "co": 0.8,
    "no2": 20.0,
    "o3": 35.0,
    "so2": 10.0
  }'
```

## Cara Menjalankan di GitHub Codespaces
1. Buka repository di GitHub.
2. Pilih Code lalu Create Codespace.
3. Setelah container siap, jalankan perintah Docker Compose di terminal.
4. Gunakan port forwarding untuk mengakses 5000 dan 8000.

## Workflow Inti
1. Data Ingestion
2. Data Preprocessing
3. Model Training
4. Experiment Tracking dengan MLflow
5. Model Registry
6. Inference Service via FastAPI