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
    langsmith datasets rank-bm25

RUN python -m spacy download en_core_web_lg

COPY main.py ./
COPY app/ ./app/

RUN mkdir -p ./data/processed ./data/chroma_db

COPY data/processed/bm25_index.pkl ./data/processed/bm25_index.pkl
COPY data/chroma_db/ ./data/chroma_db/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]