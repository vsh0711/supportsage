
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np

class Embedder:
    """
    Local embedding model — zero API cost, runs on M3 via MPS.
    all-MiniLM-L6-v2: 384-dim vectors, 80MB, fast enough for real-time.
    """
    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="mps"  # Apple Silicon GPU — faster than CPU
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True  # cosine sim = dot product after normalization
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            normalize_embeddings=True
        )

# Singleton — load model once, reuse across requests
embedder = Embedder()