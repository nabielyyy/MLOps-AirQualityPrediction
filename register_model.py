import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_PATH = "mlruns/671342552043566732/models/m-593ccbf6c6af432da476c8fa86536516/artifacts"

result = mlflow.register_model(
    model_uri=MODEL_PATH,
    name="AirQualityRandomForestModel"
)

print("Registered version:", result.version)