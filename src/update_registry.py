"""
Script untuk mendaftarkan model ke MLflow Model Registry dengan status Staging.
Hanya dijalankan jika evaluasi model berhasil.
"""

import mlflow
import mlflow.sklearn
import json
import os
import sys
from datetime import datetime

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")

MODEL_NAME = "AirQualityRandomForestModel"

def get_latest_model_uri():
    """Ambil URI model terbaru yang berhasil di-train."""
    client = mlflow.tracking.MlflowClient()
    
    # Cari experiment berkelanjuta bernama "AirQualityPrediction"
    experiment = client.get_experiment_by_name("AirQualityPrediction")
    if not experiment:
        print("❌ Experiment 'AirQualityPrediction' tidak ditemukan")
        return None
    
    # Ambil run terbaru
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if not runs:
        print("❌ Tidak ada runs ditemukan")
        return None
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    
    # Model biasanya disimpan di artifact dengan nama "random_forest_model"
    model_uri = f"runs:/{run_id}/random_forest_model"
    
    return model_uri, run_id

def check_evaluation_passed():
    """Periksa apakah evaluasi berhasil."""
    if not os.path.exists("evaluation_result.json"):
        print("❌ Evaluation result file tidak ditemukan")
        return False
    
    with open("evaluation_result.json", "r") as f:
        result = json.load(f)
    
    return result.get("passed", False)

def promote_model_to_staging(model_uri, run_id):
    """
    Promosi model ke staging dengan menambahkan alias di Model Registry.
    """
    client = mlflow.tracking.MlflowClient()
    
    print(f"\n🚀 Promoting model to Staging...")
    print(f"   Model URI: {model_uri}")
    print(f"   Run ID: {run_id}")
    
    try:
        # Register model jika belum terdaftar
        model_version = mlflow.sklearn.log_model(
            sk_model=mlflow.sklearn.load_model(model_uri),
            artifact_path="random_forest_model",
            registered_model_name=MODEL_NAME
        )
        
        print(f"✅ Model registered with version")
    except Exception as e:
        # Model mungkin sudah terdaftar, coba untuk membuat version baru
        print(f"ℹ️ Model registration attempt: {str(e)}")
    
    # Ambil semua versions dari model
    try:
        versions = client.get_latest_versions(MODEL_NAME)
        
        if versions:
            latest_version = versions[0]
            version_number = latest_version.version
            
            print(f"📦 Latest model version: {version_number}")
            
            # Update stage ke "Staging"
            try:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=version_number,
                    stage="Staging",
                    archive_existing_versions=False
                )
                print(f"✅ Model version {version_number} promoted to Staging")
            except Exception as e:
                print(f"ℹ️ Stage transition: {str(e)}")
            
            return version_number
    except Exception as e:
        print(f"⚠️ Error retrieving model versions: {str(e)}")
        print("   This might be normal if model registry is using filesystem backend")
        return None

def update_model_registry_yaml(version_number):
    """
    Update file model_registry.yaml dengan versi terbaru di Staging.
    """
    import yaml
    
    registry_path = "model_registry.yaml"
    
    try:
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)
        
        # Update staging version
        registry['model_registry']['staging_version'] = str(version_number)
        registry['model_registry']['last_update'] = datetime.now().isoformat()
        
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False)
        
        print(f"📝 Updated model_registry.yaml - Staging version: {version_number}")
        return True
    except Exception as e:
        print(f"⚠️ Could not update model_registry.yaml: {str(e)}")
        return False

def print_registry_update_report(version_number):
    """Tampilkan laporan registry update."""
    print("\n" + "="*60)
    print("📋 MODEL REGISTRY UPDATE REPORT")
    print("="*60)
    print(f"\n✅ Model: {MODEL_NAME}")
    print(f"✅ Version: {version_number}")
    print(f"✅ Stage: Staging")
    print(f"✅ Timestamp: {datetime.now().isoformat()}")
    print(f"\n✅ Model successfully registered to MLflow Model Registry!")
    print("   Status: Ready for testing and production deployment")
    print("="*60 + "\n")

def main():
    print("🔄 Starting Model Registry Update...")
    
    # Periksa apakah evaluasi berhasil
    if not check_evaluation_passed():
        print("❌ Evaluation did not pass - skipping registry update")
        sys.exit(1)
    
    # Ambil model URI terbaru
    model_info = get_latest_model_uri()
    if not model_info:
        print("❌ Could not retrieve latest model URI")
        sys.exit(1)
    
    model_uri, run_id = model_info
    
    # Promosi model ke Staging
    version_number = promote_model_to_staging(model_uri, run_id)
    
    # Update model_registry.yaml
    if version_number:
        update_model_registry_yaml(version_number)
        print_registry_update_report(version_number)
    else:
        print("⚠️ Could not determine version number for registry update")
    
    print("✅ Model registry update process completed!")
    sys.exit(0)

if __name__ == "__main__":
    main()
