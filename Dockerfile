# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Install pip dependencies directly (no Poetry in container)
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn langchain langchain-community langchain-groq \
    chromadb sentence-transformers rank-bm25 \
    anthropic presidio-analyzer presidio-anonymizer \
    "spacy>=3.7.2,<3.8.0" \
    python-dotenv pydantic pydantic-settings \
    langsmith datasets

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY app/ ./app/
COPY main.py ./
COPY data/chroma_db/ ./data/chroma_db/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]