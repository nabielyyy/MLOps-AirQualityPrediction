import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


def get_latest_model_version(model_name: str) -> str:
    """Mengambil nomor versi terbaru dari model di MLflow Model Registry."""
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"Tidak ada versi ditemukan untuk model '{model_name}' di registry.")
    
    latest_version = max(model_versions, key=lambda v: int(v.version))
    return latest_version.version


def _load_latest_local_artifact(model_name: str):
    """Fallback: memuat versi terbaru yang tersedia secara lokal di folder mlruns."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(base_dir, 'mlruns', 'models', model_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Folder model tidak ditemukan di {model_dir}")
    
    # Cari semua folder version-* dan ambil yang paling besar
    versions = []
    for entry in os.listdir(model_dir):
        if entry.startswith("version-"):
            try:
                versions.append(int(entry.split("-", 1)[1]))
            except ValueError:
                continue
    
    if not versions:
        raise FileNotFoundError(f"Tidak ada versi lokal ditemukan untuk model '{model_name}'")
    
    latest_version = max(versions)
    meta_path = os.path.join(model_dir, f"version-{latest_version}", "meta.yaml")
    
    import yaml
    with open(meta_path, 'r') as file:
        metadata = yaml.safe_load(file)
    
    storage_location = metadata.get('storage_location', '').replace('file://', '')
    
    artifact_candidates = [storage_location]
    if '/mlruns/' in storage_location:
        suffix = storage_location.split('/mlruns/', 1)[1]
        artifact_candidates.extend([f"/app/mlruns/{suffix}", f"/mlruns/{suffix}"])
    
    for candidate in artifact_candidates:
        if os.path.exists(candidate):
            return mlflow.sklearn.load_model(candidate)
    
    return mlflow.sklearn.load_model(storage_location)


def load_model(version=None):
    """Memuat model dari MLflow Model Registry atau fallback ke artefak lokal."""
    model_name = os.getenv("MODEL_NAME") or os.getenv("MLFLOW_MODEL_NAME") or "AirQualityRandomForestModel"

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    explicit_uri = os.getenv("MODEL_URI")
    if explicit_uri:
        return mlflow.sklearn.load_model(explicit_uri)

    candidate_uris = []
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        candidate_uris.append(tracking_uri)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for candidate in ("./mlruns", f"{base_dir}/mlruns", "/app/mlruns", "/mlruns"):
        candidate_uris.append(f"file:{candidate}")

    seen = set()
    last_error = None
    for uri in candidate_uris:
        if uri in seen:
            continue
        seen.add(uri)

        try:
            mlflow.set_tracking_uri(uri)
            try:
                mlflow.set_registry_uri(uri)
            except Exception:
                pass

            if version is None:
                try:
                    latest_version = get_latest_model_version(model_name)
                    return mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
                except Exception as reg_exc:
                    last_error = reg_exc
                    # Fallback: cari versi terbaru secara lokal
                    return _load_latest_local_artifact(model_name)

            return mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Model tidak bisa dimuat dari MLflow: {last_error}")


def load_production_model():
    """Memuat model versi terbaru (produksi)."""
    return load_model()


def load_staging_model():
    """Memuat model versi staging — gunakan env var MLFLOW_STAGING_VERSION."""
    staging_version = os.getenv("MLFLOW_STAGING_VERSION")
    if not staging_version:
        raise ValueError("MLFLOW_STAGING_VERSION environment variable harus di-set untuk load_staging_model()")
    return load_model(staging_version)