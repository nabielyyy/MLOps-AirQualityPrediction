import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/processed/processed_air_quality.csv")

# Feature dan target
X = df[["pm2_5", "pm10", "co", "no2", "o3", "so2"]]
y = df["air_quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Start MLflow experiment
mlflow.set_experiment("AirQualityPrediction")

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Logging
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")