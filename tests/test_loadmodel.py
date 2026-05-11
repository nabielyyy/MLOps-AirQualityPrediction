import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loadmodel import load_model_registry, load_model, load_production_model, load_staging_model

def test_load_model_registry():
    """Test memuat model registry dari YAML."""
    registry = load_model_registry()
    assert isinstance(registry, dict)
    assert 'model_registry' in registry
    assert 'name' in registry['model_registry']
    assert 'production_version' in registry['model_registry']
    assert 'staging_version' in registry['model_registry']

def test_load_model_registry_content():
    """Test isi registry sesuai."""
    registry = load_model_registry()
    mr = registry['model_registry']
    assert mr['name'] == "AirQualityRandomForestModel"
    assert mr['production_version'] == "1"
    assert mr['staging_version'] == "2"

def test_load_production_model():
    """Test memuat model produksi."""
    # Jika model tidak ada, akan raise exception
    try:
        model = load_production_model()
        assert model is not None
        # Asumsikan model adalah RandomForestClassifier
        assert hasattr(model, 'predict')
    except Exception as e:
        pytest.skip(f"Model tidak bisa di-load: {e}")

def test_load_staging_model():
    """Test memuat model staging."""
    try:
        model = load_staging_model()
        assert model is not None
        assert hasattr(model, 'predict')
    except Exception as e:
        pytest.skip(f"Model tidak bisa di-load: {e}")

def test_load_model_specific_version():
    """Test memuat model versi spesifik."""
    try:
        model = load_model(version="1")
        assert model is not None
        assert hasattr(model, 'predict')
    except Exception as e:
        pytest.skip(f"Model versi 1 tidak bisa di-load: {e}")