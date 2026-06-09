import os

import mlflow
import mlflow.sklearn
import yaml


def load_model_registry():
    """Memuat model registry dari file YAML."""
    registry_path = os.path.join(os.path.dirname(__file__), '..', 'model_registry.yaml')
    with open(registry_path, 'r') as file:
        registry = yaml.safe_load(file)
    return registry


def _load_model_from_local_artifact(model_name, version):
    """Memuat model langsung dari artefak lokal di folder mlruns jika registry URI tidak tersedia."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    registry_meta_path = os.path.join(base_dir, 'mlruns', 'models', model_name, f'version-{version}', 'meta.yaml')
    if not os.path.exists(registry_meta_path):
        raise FileNotFoundError(f"Metadata model tidak ditemukan di {registry_meta_path}")

    with open(registry_meta_path, 'r') as file:
        metadata = yaml.safe_load(file)

    storage_location = metadata.get('storage_location')
    if not storage_location:
        raise FileNotFoundError('storage_location tidak ditemukan pada metadata model')

    artifact_candidates = []
    raw_path = storage_location.replace('file://', '')
    artifact_candidates.append(raw_path)

    if '/mlruns/' in raw_path:
        suffix = raw_path.split('/mlruns/', 1)[1]
        artifact_candidates.extend([f"/app/mlruns/{suffix}", f"/mlruns/{suffix}"])

    if raw_path.startswith('/workspaces/') or raw_path.startswith('/home/'):
        artifact_candidates.append(raw_path.replace('/workspaces/MLOps-AirQualityPrediction', '/app'))

    for candidate in artifact_candidates:
        if os.path.exists(candidate):
            return mlflow.sklearn.load_model(candidate)

    return mlflow.sklearn.load_model(raw_path)


def load_model(version=None):
    """Memuat model dari MLflow Model Registry atau fallback ke artefak lokal."""
    registry = load_model_registry()
    model_name = os.getenv("MODEL_NAME") or os.getenv("MLFLOW_MODEL_NAME") or registry['model_registry']['name']

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
                alias = os.getenv("MLFLOW_MODEL_ALIAS", "Production")
                try:
                    return mlflow.sklearn.load_model(f"models:/{model_name}/{alias}")
                except Exception:
                    production_version = registry['model_registry']['production_version']
                    return _load_model_from_local_artifact(model_name, production_version)

            return mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        except Exception as exc:  # pragma: no cover - exercised during startup fallback
            last_error = exc

    raise RuntimeError(f"Model tidak bisa dimuat dari MLflow: {last_error}")


def load_production_model():
    """Memuat model versi produksi."""
    return load_model()


def load_staging_model():
    """Memuat model versi staging."""
    registry = load_model_registry()
    version = registry['model_registry']['staging_version']
    return load_model(version)