import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "AirQualityRandomForestModel"

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_PATH = (
    "mlruns/671342552043566732/"
    "models/m-593ccbf6c6af432da476c8fa86536516/artifacts"
)

print("Loading model artifact...")

model = mlflow.sklearn.load_model(MODEL_PATH)

print("Creating fresh MLflow run...")

with mlflow.start_run() as run:

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    model_uri = f"runs:/{run.info.run_id}/random_forest_model"

    print("Registering model...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    version = result.version

    client = MlflowClient()

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    print("\nSUCCESS")
    print("Version:", version)