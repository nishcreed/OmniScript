FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    rubberband-cli \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]