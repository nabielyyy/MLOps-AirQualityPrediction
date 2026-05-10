# Air Quality Prediction MLOps Project

## Project Overview
This project aims to build a Machine Learning system for air quality prediction using an MLOps pipeline. The system supports automated data ingestion, preprocessing, model training, versioning, and deployment preparation.

## Project Structure

```bash
data/
├── raw/
└── processed/

models/
notebooks/
src/
tests/
config/
.github/workflows/
```

## Technologies Used
- Python
- Scikit-learn
- MLflow
- DVC
- GitHub Actions

## How to Run Codespaces
1. Open repository on GitHub
2. Click Code
3. Open Codespaces
4. Create Codespace on Main

## Workflow
1. Data Ingestion
2. Data Preprocessing
3. Model Training
4. Experiment Tracking
5. Model Registry
6. Automation Pipeline

## Data Ingestion

Run the ingestion script:

```bash
python src/ingest_data.py
```

## Data Preprocessing

Run preprocessing:

```bash
python src/preprocess.py
```

## Output
- Raw data stored in `data/raw/`
- Processed data stored in `data/processed/`