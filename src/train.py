import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("file:./mlruns")

# Load dataset
df = pd.read_csv("data/processed/processed_air_quality.csv")

# Hapus baris target kosong jika ada
missing_target = df['air_quality'].isna().sum()
if missing_target > 0:
    print(f"Found {missing_target} rows with missing air_quality, dropping them")
    df = df.dropna(subset=['air_quality'])


X = df[["pm2_5", "pm10", "co", "no2", "o3", "so2"]]
y = df["air_quality"]

# Split dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment
mlflow.set_experiment("AirQualityPrediction")

params_to_test = [10, 100, 1000]

results = []

for n_estimators in params_to_test:
    with mlflow.start_run():
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
            registered_model_name="AirQualityRandomForestModel"
        )

        results.append({
            "n_estimators": n_estimators,
            "accuracy": accuracy,
            "f1_score": f1
        })

        print(f"n_estimators={n_estimators} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

# Cetak ringkasan
print("\n=== Perbandingan Model ===")
for result in results:
    print(f"n_estimators={result['n_estimators']}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")

best = max(results, key=lambda x: x['accuracy'])
print(f"\n✓ Best model: n_estimators={best['n_estimators']} dengan Accuracy={best['accuracy']:.4f}")