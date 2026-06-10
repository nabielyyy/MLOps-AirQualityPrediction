import os
import time
from typing import Optional

import mlflow
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import fungsi tambahan untuk mengambil versi terbaru
from src.loadmodel import load_model, get_latest_model_version


# Model input format yang disesuaikan dengan fitur yang dipakai saat training.
class AirQualityInput(BaseModel):
    pm2_5: float
    pm10: float
    co: float
    no2: float
    o3: float
    so2: float


class PredictionResponse(BaseModel):
    prediction: str
    model_name: str
    model_version: Optional[str] = None
    message: str


# FastAPI app utama untuk inference service.
app = FastAPI(title="Air Quality Prediction API", version="1.0.0")


def _build_mlflow_payload(payload_dict: dict) -> dict:
    """Membentuk payload yang kompatibel dengan endpoint invocations MLflow model serving."""
    feature_order = ["pm2_5", "pm10", "co", "no2", "o3", "so2"]
    values = [payload_dict[name] for name in feature_order]
    return {
        "dataframe_split": {
            "columns": feature_order,
            "data": [values],
        }
    }


def _predict_via_mlflow_server(payload_dict: dict) -> str:
    """Mengirim prediksi ke MLflow model serving yang berjalan sebagai service terpisah."""
    model_server_url = os.getenv("MODEL_SERVER_URL", "http://mlflow-model-serving:5001")
    response = requests.post(
        f"{model_server_url}/invocations",
        json=_build_mlflow_payload(payload_dict),
        timeout=10,
    )
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and "predictions" in body:
        return str(body["predictions"][0])
    if isinstance(body, list):
        return str(body[0])
    return str(body)


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Memuat model produksi dari MLflow Model Registry saat aplikasi start."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    model_name = os.getenv("MODEL_NAME") or os.getenv("MLFLOW_MODEL_NAME", "AirQualityRandomForestModel")

    # Memberi jeda singkat agar mlflow-server sudah siap sebelum API mencoba memuat model.
    last_error = None
    for attempt in range(8):
        try:
            model = load_model()
            app.state.model = model
            app.state.model_name = model_name
            
            # Ambil versi terbaru yang sebenarnya dari registry untuk dilaporkan di state
            try:
                app.state.model_version = get_latest_model_version(model_name)
            except Exception:
                # Fallback jika query ke registry gagal
                app.state.model_version = os.getenv("MLFLOW_MODEL_ALIAS", "Production")
                
            app.state.model_loaded = True
            app.state.use_mlflow_server = os.getenv("USE_MLFLOW_MODEL_SERVER", "true").lower() == "true"
            return
        except Exception as exc:  # pragma: no cover - startup path
            last_error = exc
            time.sleep(2)

    raise RuntimeError(f"Model tidak bisa dimuat dari MLflow: {last_error}")


@app.get("/")
def read_root() -> dict:
    """Endpoint sederhana untuk memastikan API berjalan."""
    return {"message": "Air Quality Prediction API is running"}


@app.get("/health")
def health_check() -> dict:
    """Endpoint health check untuk monitoring container."""
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model_loaded", False),
        "model_name": getattr(app.state, "model_name", os.getenv("MODEL_NAME") or os.getenv("MLFLOW_MODEL_NAME", "AirQualityRandomForestModel")),
        "model_version": getattr(app.state, "model_version", os.getenv("MLFLOW_MODEL_ALIAS", "Production")),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"),
        "model_server_url": os.getenv("MODEL_SERVER_URL", "http://mlflow-model-serving:5001"),
        "use_mlflow_server": getattr(app.state, "use_mlflow_server", False),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: AirQualityInput) -> PredictionResponse:
    """Menerima fitur numerik dan mengembalikan prediksi kualitas udara."""
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model belum siap")

    payload_dict = payload.dict() if hasattr(payload, "dict") else payload.model_dump()

    if getattr(app.state, "use_mlflow_server", False):
        try:
            prediction = _predict_via_mlflow_server(payload_dict)
            return PredictionResponse(
                prediction=str(prediction),
                model_name=app.state.model_name,
                model_version=app.state.model_version,
                message="Prediksi berhasil dibuat melalui MLflow Model Serving",
            )
        except Exception as exc:  # pragma: no cover - startup path
            raise HTTPException(status_code=502, detail=f"MLflow model server tidak tersedia: {exc}") from exc

    feature_order = ["pm2_5", "pm10", "co", "no2", "o3", "so2"]
    feature_frame = pd.DataFrame([payload_dict], columns=feature_order)
    prediction = app.state.model.predict(feature_frame)[0]

    return PredictionResponse(
        prediction=str(prediction),
        model_name=app.state.model_name,
        model_version=app.state.model_version,
        message="Prediksi berhasil dibuat menggunakan model lokal",
    )