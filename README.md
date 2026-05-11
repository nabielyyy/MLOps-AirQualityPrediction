# MLOps-AirQualityPrediction

## Air Quality Prediction Based on Air Pollution Parameters Using a Batch-Based MLOps Approach

This project implements an end-to-end MLOps pipeline for predicting air quality levels based on air pollution parameters using a batch-based approach. The system performs automated data ingestion from the OpenWeather Air Pollution API, preprocessing, model training, experiment tracking with MLflow, data versioning with DVC, and automation using GitHub Actions.

---

# Project Objectives

- Build an automated air quality prediction pipeline
- Implement batch-based data ingestion
- Apply preprocessing to air pollution datasets
- Train and evaluate a Machine Learning model
- Track experiments and models using MLflow
- Manage dataset versioning using DVC
- Automate the pipeline using GitHub Actions

---

# Technologies Used

- Python
- Pandas
- Scikit-Learn
- MLflow
- DVC (Data Version Control)
- GitHub Actions
- OpenWeather Air Pollution API

---

# Project Structure

```text
MLOps-AirQualityPrediction/
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── raw.dvc
│
├── mlruns/
│
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   └── train.py
│
├── .dvc/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Workflow Pipeline

```text
Air Pollution API
        ↓
Data Ingestion
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
MLflow Experiment Tracking
        ↓
Model Registry
        ↓
CI/CD Automation
```

---

# Data Ingestion

The ingestion pipeline collects real-time air pollution data from the OpenWeather Air Pollution API. The collected data includes:

- PM2.5
- PM10
- CO
- NO2
- O3
- SO2
- AQI

Each ingestion process generates a timestamp-based CSV file stored in:

```text
data/raw/
```

Run ingestion:

```bash
python src/ingest_data.py
```

---

# Data Preprocessing

The preprocessing stage performs:

- Duplicate removal
- Missing value handling
- AQI label transformation
- Dataset preparation for training

Processed datasets are stored in:

```text
data/processed/
```

Run preprocessing:

```bash
python src/preprocess.py
```

---

# Model Training

This project uses the Random Forest Classifier algorithm to classify air quality categories based on pollution parameters.

Evaluation metrics:
- Accuracy
- F1-Score

Run training:

```bash
python src/train.py
```

---

# MLflow Experiment Tracking

MLflow is used to:
- Track experiments
- Store parameters and metrics
- Save trained models
- Manage model versions

Run MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

# DVC Integration

DVC is used for dataset versioning and tracking dataset changes during continual learning simulation.

Initialize DVC:

```bash
dvc init
```

Track dataset:

```bash
dvc add data/raw
```

---

# GitHub Actions Automation

This project implements CI/CD automation using GitHub Actions. The workflow automatically performs:

- Data ingestion
- Data preprocessing
- Model training

Workflow file:

```text
.github/workflows/mlops_pipeline.yml
```

---

# Continual Learning Simulation

The project simulates continual learning by:
- periodically collecting new air quality data,
- updating datasets,
- retraining the model automatically,
- and generating new model versions using MLflow Model Registry.

---

# Model Registry

MLflow Model Registry is used to manage:
- model versioning,
- lifecycle management,
- staging,
- and production-ready models.

Registered model name:

```text
AirQualityRandomForestModel
```

---

# Installation

Clone repository:

```bash
git clone <repository-url>
cd MLOps-AirQualityPrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
pandas
numpy
requests
scikit-learn
mlflow
dvc
pytest
```

---

# Results

The implemented MLOps pipeline successfully supports:
- automated batch ingestion,
- preprocessing,
- experiment tracking,
- model registry,
- dataset versioning,
- and CI/CD automation.

---

# Author

Nabiel Tatra Edy Firdaus  
Department of Informatics Engineering  
Faculty of Computer Science  
Universitas Brawijaya
