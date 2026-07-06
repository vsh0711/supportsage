FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install fastapi uvicorn langchain langchain-community langchain-groq \
    chromadb sentence-transformers rank-bm25 \
    anthropic presidio-analyzer presidio-anonymizer \
    "spacy>=3.7.2,<3.8.0" \
    python-dotenv pydantic pydantic-settings \
    langsmith datasets

RUN python -m spacy download en_core_web_lg

COPY main.py ./
COPY app/ ./app/
COPY data/raw/support_dataset.jsonl ./data/raw/support_dataset.jsonl

RUN mkdir -p ./data/processed ./data/chroma_db

EXPOSE 8000

# Run ingestion first, then start the server
CMD ["sh", "-c", "PYTHONPATH=/app python app/rag/ingestor.py && uvicorn main:app --host 0.0.0.0 --port 8000"]