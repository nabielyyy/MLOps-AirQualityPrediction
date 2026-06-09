# Dockerfile untuk API inference service.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true

# Install dependency yang dibutuhkan oleh API dan MLflow.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh project ke container.
COPY . /app

# Expose port aplikasi inference.
EXPOSE 8000

# Jalankan FastAPI melalui Uvicorn.
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
