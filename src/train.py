import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/processed/processed_air_quality.csv")

# Features dan target
X = df[["pm2_5", "pm10", "co", "no2", "o3", "so2"]]
y = df["air_quality"]

# Karena dataset kecil
X_train = X
X_test = X
y_train = y
y_test = y

# MLflow Experiment
mlflow.set_experiment("AirQualityPrediction")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name="AirQualityRandomForestModel"
    )

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")