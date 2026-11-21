FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false \
    PORT=8501
WORKDIR /app

# CPU-only PyTorch first (the default Linux build includes multi-GB GPU libraries)
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY data ./data
COPY .streamlit ./.streamlit
COPY start.sh .

# Download the embedding + reranker models and pre-compute embeddings at build time,
# so the container starts fast and never downloads anything at runtime.
RUN python -c "from app.rag import HybridRetriever; HybridRetriever()"

RUN useradd -m -u 1000 appuser && chown -R appuser /app && chmod +x start.sh
USER appuser
EXPOSE 8501
CMD ["./start.sh"]