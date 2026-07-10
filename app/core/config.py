# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str

    # Vector DB
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "supportsage"

    # Embeddings via HF Inference API
    HF_TOKEN: str
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Models
    PRIMARY_MODEL: str = "llama-3.1-8b-instant"
    FALLBACK_MODEL: str = "llama-3.3-70b-versatile"

    # RAG
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3

    # Observability
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "supportsage"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()