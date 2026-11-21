#!/bin/sh
# Start the FastAPI backend (internal only), wait until it is ready, then start Streamlit (public).
set -e
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
python - <<'PY'
import time, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2)
        break
    except Exception:
        time.sleep(1)
PY
exec python -m streamlit run frontend/streamlit_app.py \
     --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true