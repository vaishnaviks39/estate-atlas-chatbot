# Estate Atlas: AI property assistant for DarGlobal & Wasalt

A chatbot that answers questions about DarGlobal luxury projects and Wasalt property listings
in Saudi Arabia, in English or Arabic, using data scraped from both public websites.

## How it works
- **Scraping:** Wasalt listings via the schema.org JSON-LD embedded in its search pages (844 listings + FAQs,
  6 cities, sale and rent). DarGlobal uses Imperva bot protection, so rather than bypass it, pages were saved
  with a normal browser and parsed from their embedded `__NEXT_DATA__` JSON (38 projects, 171 FAQs, 14 guides).
- **Chunking:** structure-aware, token-based chunks (max 128 tokens) with contextual headers.
- **Retrieval:** hybrid search (BM25 with stemming + multilingual MiniLM embeddings) merged with Reciprocal
  Rank Fusion, de-duplicated, then reranked with a multilingual cross-encoder.
- **Query understanding:** the model rewrites each question into an English search query and extracts
  filters (city, sale/rent, type, price, bedrooms, sort), which are applied to listings as metadata filters.
- **LLM:** NVIDIA Nemotron 3.5 Super VL (free, via OpenRouter), chosen after benchmarking three free models on
  speed, JSON reliability and Arabic quality. Automatic fallback to another free model, then OpenAI, if the
  free quota is exhausted.
- **App:** FastAPI backend (`/api/chat` streaming, `/api/health`) + Streamlit frontend.
- **Deployment:** Azure VM (Ubuntu) with Caddy for HTTPS. A `Dockerfile` + `start.sh` run the backend and
  frontend in one container.

## Project structure
    app/            backend: chunker, hybrid retrieval, query analysis, listing filter, LLM client, pipeline, FastAPI
    frontend/       Streamlit chat UI
    scraper/        Wasalt and DarGlobal scrapers
    data/           scraped data (JSON)
    Dockerfile, start.sh

## Run locally
    python3.12 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # add OPENROUTER_API_KEY (and optionally OPENAI_API_KEY)
    python -m uvicorn app.main:app --port 8000          # terminal 1
    python -m streamlit run frontend/streamlit_app.py    # terminal 2

## Run with Docker
    docker build -t estate-atlas .
    docker run --env-file .env -p 8501:8501 estate-atlas

## Refresh the data
    python -m scraper.wasalt
    python -m scraper.darglobal      # reads pages saved in raw/darglobal/

## Limitations
- Data is a snapshot from November 2026; Wasalt rent listings don't state whether rent is monthly or yearly.
- DarGlobal does not publish prices, so the assistant says so rather than guessing.