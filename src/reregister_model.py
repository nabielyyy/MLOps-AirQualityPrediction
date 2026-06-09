import os
import time
from datetime import datetime

import mlflow
import yaml
from mlflow.tracking import MlflowClient

MODEL_NAME = "AirQualityRandomForestModel"
MODEL_ARTIFACT_PATH = "random_forest_model"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLRUNS_DIR = os.path.join(ROOT_DIR, "mlruns")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_REGISTRY_FILE = os.path.join(ROOT_DIR, "model_registry.yaml")


def load_local_model_meta():
    meta_path = os.path.join(
        MLRUNS_DIR,
        "models",
        MODEL_NAME,
        "version-1",
        "meta.yaml",
    )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Tidak dapat menemukan metadata model lokal di {meta_path}."
        )

    with open(meta_path, "r") as f:
        return yaml.safe_load(f)


def ensure_registered_model(client: MlflowClient):
    try:
        client.get_registered_model(MODEL_NAME)
        return
    except Exception:
        client.create_registered_model(MODEL_NAME)


def get_production_version(client: MlflowClient):
    try:
        versions = client.get_latest_versions(MODEL_NAME)
    except Exception:
        return None

    for v in versions:
        if v.current_stage == "Production":
            return v.version
    return None


def create_model_version(client: MlflowClient, run_id: str, storage_location: str | None):
    if storage_location and storage_location.startswith("file://"):
        model_source = storage_location
    else:
        model_source = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"

    print(f"🔧 Creating model version from source: {model_source}")
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_source,
    )
    return model_version.version


def wait_for_version_ready(client: MlflowClient, version: str, timeout_seconds: int = 30):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        mv = client.get_model_version(MODEL_NAME, version)
        if mv.status == "READY":
            return mv
        time.sleep(1)
    raise RuntimeError(f"Model version {version} tidak siap dalam {timeout_seconds} detik")


def transition_to_production(client: MlflowClient, version: str):
    print(f"➡️ Transitioning model version {version} to Production")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    try:
        client.set_registered_model_alias(MODEL_NAME, "Production", version)
    except Exception:
        pass


def write_registry_yaml(version: str):
    payload = {
        "model_registry": {
            "name": MODEL_NAME,
            "production_version": str(version),
            "staging_version": str(version),
            "last_update": datetime.utcnow().isoformat() + "Z",
        }
    }
    with open(MODEL_REGISTRY_FILE, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"✅ Created/updated {MODEL_REGISTRY_FILE}")


def main():
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    print(f"Using MLflow tracking URI: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    meta = load_local_model_meta()
    run_id = meta.get("run_id")
    if not run_id:
        raise RuntimeError("run_id tidak ditemukan di metadata model lokal")

    storage_location = meta.get("storage_location")
    ensure_registered_model(client)
    production_version = get_production_version(client)
    if production_version:
        print(f"ℹ️ Model {MODEL_NAME} sudah memiliki Production version {production_version}")
    else:
        version = create_model_version(client, run_id, storage_location)
        mv = wait_for_version_ready(client, version)
        if mv.current_stage != "Production":
            transition_to_production(client, version)
        production_version = version

    write_registry_yaml(production_version)
    print("\n✅ Model registry berhasil dipulihkan dan Production tersedia")
    print(f"   Model name: {MODEL_NAME}")
    print(f"   Production version: {production_version}")
    print(f"   Registry file: {MODEL_REGISTRY_FILE}")


if __name__ == "__main__":
    main()
