"""
Configuration untuk MLOps Pipeline.
Berisi threshold untuk validasi model dan setting pipeline lainnya.
"""

# Threshold validasi metrik model
METRICS_THRESHOLD = {
    "accuracy": 0.85,      # Accuracy minimum 85%
    "f1_score": 0.80,      # F1 Score minimum 80%
}

# Model Registry configuration
MODEL_REGISTRY = {
    "name": "AirQualityRandomForestModel",
    "tracking_uri": "file:./mlruns",
    "experiment_name": "AirQualityPrediction",
}

# Training configuration
TRAINING = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": [50, 100, 200],  # Parameter yang akan di-test
}

# Data configuration
DATA = {
    "raw_path": "data/raw/",
    "processed_path": "data/processed/processed_air_quality.csv",
    "features": ["pm2_5", "pm10", "co", "no2", "o3", "so2"],
    "target": "air_quality",
}

# Model configuration
MODEL = {
    "type": "RandomForestClassifier",
    "artifact_path": "random_forest_model",
}
