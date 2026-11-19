# Optional separate backend image. Not a Vercel function and not yet built on Linux.
FROM python:3.13.7-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY web/requirements-runtime.txt /app/web/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /app/web/requirements-runtime.txt
RUN useradd --create-home --uid 1000 brent && mkdir /runtime && chown brent:brent /runtime
COPY src/oil_forecast /app/src/oil_forecast
COPY web/__init__.py web/backend.py web/production.py /app/web/
COPY configs/data.json configs/economic_events.json configs/serving_bundle.json configs/dashboard_bundle.json /app/configs/
COPY artifacts/derived/brent-academic-v1 /app/artifacts/derived/brent-academic-v1
# No private datasets, weights, bundles or tokens are included in the image.
# Authorized bundles must be provisioned separately, not copied into a public repository.
ENV BRENT_MODEL_SOURCE=hf BRENT_MODEL_CACHE_DIR=/runtime/models BRENT_LIVE_DIR=/runtime/live BRENT_ALLOW_PROVIDER_ACCESS=0 BRENT_BIND_HOST=0.0.0.0 PORT=8000
USER brent
EXPOSE 8000
CMD ["python", "-m", "web.production"]
