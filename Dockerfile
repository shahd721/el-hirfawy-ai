
FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR with Arabic and English languages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements-render.txt .

RUN pip install --no-cache-dir -r requirements-render.txt

# Copy project files, including face recognition models
COPY . .

# Start FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
