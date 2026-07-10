# app/rag/embedder.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Embedder:
    """
    Lazy-loaded local embedder.

    Key design decision: model loads on FIRST REQUEST, not at import time.
    This means the container starts in <2 seconds with ~80MB RAM.
    First request takes ~3 seconds (model load) then stays warm.

    Why lazy loading matters for free tier:
    Render spins down idle containers after 15 minutes.
    On spin-up, if model loads at import → 30 second cold start → request timeout.
    Lazy load → fast spin-up → model loads on first real request → acceptable.

    Interview answer for "how did you handle cold starts?":
    "Lazy model loading + a /health endpoint that pre-warms the model
    by calling embed_query on startup ping."
    """
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (first request)...")
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )
            logger.info("Embedding model ready ✅")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

embedder = Embedder()