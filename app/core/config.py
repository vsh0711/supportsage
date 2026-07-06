# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str
    
    # Observability
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "supportsage"
    
    # Models
    PRIMARY_MODEL: str = "llama-3.1-8b-instant"       # fast, cheap — simple queries
    FALLBACK_MODEL: str = "llama-3.3-70b-versatile"   # strong — complex queries
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"         # local, no API cost
    
    # RAG
    CHROMA_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "supportsage_kb"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 10      # retrieve 10, rerank to top 3
    TOP_K_RERANK: int = 3
    
    # Cache
    SIMILARITY_CACHE_THRESHOLD: float = 0.92  # 92% similar = cache hit
    
    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()