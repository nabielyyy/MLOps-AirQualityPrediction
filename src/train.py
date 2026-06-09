import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_air_quality.csv"
MODEL_NAME = os.getenv("MODEL_NAME", "AirQualityRandomForestModel")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("AirQualityPrediction")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Hapus baris target kosong jika ada
missing_target = df['air_quality'].isna().sum()
if missing_target > 0:
    print(f"Found {missing_target} rows with missing air_quality, dropping them")
    df = df.dropna(subset=['air_quality'])


X = df[["pm2_5", "pm10", "co", "no2", "o3", "so2"]]
y = df["air_quality"]

# Split dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params_to_test = [10, 100, 1000]
results = []
client = mlflow.MlflowClient()
best_run_id = None
best_result = None

for n_estimators in params_to_test:
    with mlflow.start_run(run_name=f"rf_n{n_estimators}") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
        )

        result = {
            "n_estimators": n_estimators,
            "accuracy": accuracy,
            "f1_score": f1,
            "run_id": run.info.run_id,
        }
        results.append(result)

        if best_result is None or accuracy > best_result["accuracy"]:
            best_result = result
            best_run_id = run.info.run_id

        print(f"n_estimators={n_estimators} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

# Cetak ringkasan
print("\n=== Perbandingan Model ===")
for result in results:
    print(f"n_estimators={result['n_estimators']}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")

if best_run_id and best_result:
    model_uri = f"runs:/{best_run_id}/random_forest_model"
    registered_model = mlflow.register_model(model_uri, MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(MODEL_NAME, "Production", registered_model.version)
    print(f"\n✓ Best model: n_estimators={best_result['n_estimators']} dengan Accuracy={best_result['accuracy']:.4f}")
    print(f"✓ Model {MODEL_NAME} versi {registered_model.version} ditetapkan sebagai Production")