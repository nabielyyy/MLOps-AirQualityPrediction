

import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import sys
from datetime import datetime

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")

METRICS_THRESHOLD = {
    "accuracy": 0.85,      # Accuracy minimum 85%
    "f1_score": 0.80,      # F1 Score minimum 80%
}

def get_latest_run_metrics():
    """Ambil metrik dari run terakhir."""
    client = mlflow.tracking.MlflowClient()
    

    experiment = client.get_experiment_by_name("AirQualityPrediction")
    if not experiment:
        print("❌ Experiment 'AirQualityPrediction' tidak ditemukan")
        return None
    
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if not runs:
        print("❌ Tidak ada runs ditemukan")
        return None
    
    latest_run = runs[0]
    metrics = latest_run.data.metrics
    params = latest_run.data.params
    
    return {
        "run_id": latest_run.info.run_id,
        "metrics": metrics,
        "params": params,
        "timestamp": datetime.fromtimestamp(latest_run.info.start_time / 1000)
    }

def validate_metrics(metrics):
    """
    Validasi metrik model terhadap threshold yang ditentukan.
    
    Returns:
        dict: {
            "passed": bool,
            "results": list of validation results,
            "failed_metrics": list of failed metrics
        }
    """
    validation_results = {
        "passed": True,
        "results": [],
        "failed_metrics": []
    }
    
    for metric_name, threshold in METRICS_THRESHOLD.items():
        if metric_name not in metrics:
            validation_results["results"].append({
                "metric": metric_name,
                "status": "MISSING",
                "threshold": threshold,
                "actual": None
            })
            validation_results["passed"] = False
            validation_results["failed_metrics"].append(metric_name)
            continue
        
        actual_value = metrics[metric_name]
        passed = actual_value >= threshold
        
        validation_results["results"].append({
            "metric": metric_name,
            "status": "PASS" if passed else "FAIL",
            "threshold": threshold,
            "actual": actual_value,
            "difference": actual_value - threshold
        })
        
        if not passed:
            validation_results["passed"] = False
            validation_results["failed_metrics"].append(metric_name)
    
    return validation_results

def save_evaluation_report(run_info, validation_results):
    """Simpan laporan evaluasi."""
    report = {
        "timestamp": run_info["timestamp"].isoformat(),
        "run_id": run_info["run_id"],
        "parameters": run_info["params"],
        "metrics": run_info["metrics"],
        "validation": validation_results,
        "passed": validation_results["passed"]
    }
    
    os.makedirs("evaluation_reports", exist_ok=True)
    report_path = f"evaluation_reports/evaluation_{run_info['run_id']}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report_path, report

def print_evaluation_report(validation_results, run_info):
    """Tampilkan laporan evaluasi."""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\n🔍 Run ID: {run_info['run_id']}")
    print(f"⏰ Timestamp: {run_info['timestamp']}")
    print(f"\n📈 Model Metrics:")
    for metric_name, value in run_info['metrics'].items():
        print(f"  • {metric_name}: {value:.4f}")
    
    print(f"\n🎯 Validation Results:")
    print("-" * 60)
    for result in validation_results["results"]:
        status_icon = "✅" if result['status'] == "PASS" else "❌"
        print(f"{status_icon} {result['metric'].upper()}")
        print(f"   Threshold: {result['threshold']:.4f}")
        print(f"   Actual: {result['actual']:.4f}" if result['actual'] else f"   Actual: MISSING")
        if result.get('difference') is not None:
            print(f"   Difference: {result['difference']:+.4f}")
    
    print("-" * 60)
    if validation_results["passed"]:
        print("\n✅ VALIDATION PASSED - Model is ready for staging")
    else:
        print(f"\n❌ VALIDATION FAILED - Failed metrics: {', '.join(validation_results['failed_metrics'])}")
    print("="*60 + "\n")

def main():
    print("🔄 Starting Model Evaluation...")
    
    # Ambil metrik dari run terakhir
    run_info = get_latest_run_metrics()
    if not run_info:
        print("❌ Evaluation failed: Could not retrieve latest run metrics")
        sys.exit(1)
    
    # Validasi metrik
    validation_results = validate_metrics(run_info["metrics"])
    
    # Simpan laporan
    report_path, report = save_evaluation_report(run_info, validation_results)
    print(f"📄 Evaluation report saved to: {report_path}")
    
    # Tampilkan laporan
    print_evaluation_report(validation_results, run_info)
    
    # Simpan informasi untuk stage berikutnya
    with open("evaluation_result.json", "w") as f:
        json.dump({
            "passed": validation_results["passed"],
            "run_id": run_info["run_id"]
        }, f)
    
    # Exit dengan status sesuai hasil evaluasi
    if validation_results["passed"]:
        print("✅ Model evaluation successful!")
        sys.exit(0)
    else:
        print("❌ Model evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
