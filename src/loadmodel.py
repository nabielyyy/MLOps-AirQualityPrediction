import mlflow
import mlflow.sklearn
import yaml
import os

def load_model_registry():
    """Memuat model registry dari file YAML."""
    registry_path = os.path.join(os.path.dirname(__file__), '..', 'model_registry.yaml')
    with open(registry_path, 'r') as file:
        registry = yaml.safe_load(file)
    return registry

def load_model(version=None):
    """
    Memuat model dari MLflow berdasarkan versi.
    
    Args:
        version (str): Versi model. Jika None, gunakan production_version.
    
    Returns:
        model: Model yang dimuat.
    """
    registry = load_model_registry()
    model_name = registry['model_registry']['name']
    
    if version is None:
        version = registry['model_registry']['production_version']
    
    # Memuat model dari MLflow Model Registry
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

def load_production_model():
    """Memuat model versi produksi."""
    return load_model()

def load_staging_model():
    """Memuat model versi staging."""
    registry = load_model_registry()
    version = registry['model_registry']['staging_version']
    return load_model(version)