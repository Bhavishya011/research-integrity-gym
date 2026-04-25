# GPU-enabled Docker image for HuggingFace Spaces (A10G)
FROM python:3.11

EXPOSE 7860

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install PyTorch with CUDA support first (large layer, cached)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (including peerguard_lora_final/)
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

# Longer start period to allow model download on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
