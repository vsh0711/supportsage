# app/rag/retriever.py
import pickle
import numpy as np
import logging
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.rag.embedder import embedder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval against Pinecone + local BM25.

    Dense search → Pinecone (managed, zero container RAM)
    Keyword search → BM25 (in-memory, ~10MB for 500 docs)
    Reranking → embedding dot product (no extra model)

    Why keep BM25 local and not in Pinecone?
    Pinecone supports dense search only. BM25 is stateless math
    on a pickled corpus — 10MB, negligible RAM, no API cost.
    Hybrid = best of both. This is the standard production pattern.
    """

    def __init__(self):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(settings.PINECONE_INDEX)

        with open("data/processed/bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi = bm25_data["bm25"]
        self.corpus: list[str] = bm25_data["corpus"]

        logger.info("HybridRetriever initialized ✅")

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        """Query Pinecone for semantic matches."""
        query_vec = embedder.embed_query(query).tolist()
        results = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"]

    def _bm25_search(self, query: str, top_k: int) -> dict[str, float]:
        """BM25 keyword search on local corpus."""
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        max_score = scores.max() if scores.max() > 0 else 1
        normalized = scores / max_score
        top_indices = np.argsort(normalized)[::-1][:top_k]
        return {f"doc_{i}": float(normalized[i]) for i in top_indices}

    def _hybrid_merge(
        self,
        dense_matches: list[dict],
        bm25_scores: dict[str, float],
        alpha: float = 0.7
    ) -> list[dict]:
        """
        Merge dense and BM25 scores.
        alpha=0.7: 70% semantic weight, 30% keyword weight.
        Tunable — lower alpha for product-code-heavy queries.
        """
        # Build dense score map
        dense_scores = {m["id"]: m["score"] for m in dense_matches}
        dense_meta = {m["id"]: m["metadata"] for m in dense_matches}

        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        merged = {}
        for doc_id in all_ids:
            dense = dense_scores.get(doc_id, 0.0)
            bm25 = bm25_scores.get(doc_id, 0.0)
            merged[doc_id] = alpha * dense + (1 - alpha) * bm25

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "id": doc_id,
                "score": score,
                "metadata": dense_meta.get(doc_id, {})
            }
            for doc_id, score in ranked
            if doc_id in dense_meta  # only return docs with full metadata
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        top_k_rerank: int = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_RETRIEVAL
        top_k_rerank = top_k_rerank or settings.TOP_K_RERANK

        # Stage 1: Hybrid search
        dense_matches = self._dense_search(query, top_k)
        bm25_scores = self._bm25_search(query, top_k)
        candidates = self._hybrid_merge(dense_matches, bm25_scores)[:top_k]

        if not candidates:
            return []

        # Stage 2: Rerank by embedding similarity
        query_vec = embedder.embed_query(query)
        reranked = []
        for candidate in candidates:
            text = candidate["metadata"].get("text", "")
            if not text:
                continue
            doc_vec = embedder.embed_query(text[:256])
            score = float(np.dot(query_vec, doc_vec))
            reranked.append({
                "document": text,
                "metadata": candidate["metadata"],
                "score": score
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k_rerank]

retriever = HybridRetriever()