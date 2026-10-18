FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/omniscript-venv/bin:${PATH}" \
    NLTK_DATA=/opt/nltk_data \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_MODULES_CACHE=/tmp/omniscript-hf-modules \
    OMNISCRIPT_MOCK=0 \
    OMNISCRIPT_PRELOAD=1 \
    OMNISCRIPT_DEVICE=cuda:0 \
    OMNISCRIPT_DTYPE=auto \
    TESSERACT_CMD=/usr/bin/tesseract \
    PORT=8000 \
    PORT_HEALTH=8000 \
    HEALTH_CHECK_PATH=/health

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Make the base image's PyTorch available to the project environment.
RUN python3 -m venv --system-site-packages /opt/omniscript-venv

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

# Bundle these small resources in the image.
RUN python -m nltk.downloader -e -d /opt/nltk_data punkt punkt_tab

COPY src/ ./src/
COPY scripts/runpod/ ./scripts/runpod/

EXPOSE 8000

ENTRYPOINT ["bash", "/app/scripts/runpod/start.sh"]

CMD ["python", "-m", "uvicorn", "src.api:api", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
