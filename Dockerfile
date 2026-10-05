FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config.yaml \
    GRADIO_ANALYTICS_ENABLED=False \
    TARIHHTR_DEVICE=cpu \
    GRADIO_PORT=7860 \
    API_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user \
    && mkdir -p /app \
    && chown user:user /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch, kraken; print('torch', torch.__version__, 'kraken', kraken.__version__)"

USER user

ENV HOME=/home/user

WORKDIR /app

COPY --chown=user:user . /app

RUN python scripts/download_models.py

EXPOSE 7860

CMD ["python", "-m", "app.serve"]
