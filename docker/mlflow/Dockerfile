# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install MLflow at build time to speed up container startup
RUN pip install --no-cache-dir mlflow==2.16.2

# Expose the MLflow server port
EXPOSE 5000

# Add the healthcheck directly in the Dockerfile
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()" || exit 1

# Command to create the directory and start the MLflow server
# We use the exec form ["sh", "-c", "..."] to ensure proper signal handling
CMD ["sh", "-c", "mkdir -p /workspaces/MLOps-AirQualityPrediction/mlruns && \
      mlflow server \
      --backend-store-uri sqlite:////workspaces/MLOps-AirQualityPrediction/mlruns/mlflow.db \
      --default-artifact-root /workspaces/MLOps-AirQualityPrediction/mlruns \
      --host 0.0.0.0 \
      --port 5000 \
      --gunicorn-opts \"--forwarded-allow-ips='*'\""]