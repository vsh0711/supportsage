FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install fastapi uvicorn langchain langchain-community langchain-groq \
    pinecone sentence-transformers rank-bm25 \
    python-dotenv pydantic pydantic-settings \
    langsmith

# Pre-download embedding model at build time
# Model cached in image — no download at runtime = no OOM spike
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY main.py ./
COPY app/ ./app/
COPY data/processed/bm25_index.pkl ./data/processed/bm25_index.pkl

EXPOSE 8000

# Just start the server — no ingestion
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]