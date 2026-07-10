# app/rag/ingestor.py
import json
import os
import pickle
import logging
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from app.core.config import settings
from app.rag.embedder import embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Ingestor:
    def __init__(self):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(settings.PINECONE_INDEX)
        logger.info("Connected to Pinecone ✅")

    def ingest(self, jsonl_path: str, sample_size: int = 500):
        logger.info(f"Loading {sample_size} rows...")
        rows = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                rows.append(json.loads(line))

        documents, metadatas, ids, bm25_corpus = [], [], [], []

        for i, row in enumerate(rows):
            context = (
                f"This support interaction is about "
                f"{row.get('category', 'general')} specifically addressing "
                f"{row.get('intent', 'customer query').replace('_', ' ')}."
            )
            enriched_doc = (
                f"{context}\n\nQ: {row['instruction']}\nA: {row['response']}"
            )
            documents.append(enriched_doc)
            metadatas.append({
                "category": row.get("category", ""),
                "intent": row.get("intent", ""),
                "original_question": row.get("instruction", ""),
                "original_answer": row.get("response", ""),
                "text": enriched_doc[:1000]
            })
            ids.append(f"doc_{i}")
            bm25_corpus.append(enriched_doc.lower().split())

        # Embed locally
        logger.info("Generating embeddings locally...")
        embeddings = embedder.embed(documents)

        # Upsert to Pinecone in batches
        logger.info("Upserting to Pinecone...")
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            vectors = [
                {
                    "id": ids[j],
                    "values": embeddings[j].tolist(),
                    "metadata": metadatas[j]
                }
                for j in range(i, min(i + batch_size, len(documents)))
            ]
            self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {min(i+batch_size, len(documents))}/{len(documents)}")

        # BM25 stays local
        logger.info("Building BM25 index...")
        bm25 = BM25Okapi(bm25_corpus)
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/bm25_index.pkl", "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": documents}, f)

        logger.info(f"✅ Done. {len(documents)} docs in Pinecone.")
        return len(documents)


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.ingest("data/raw/support_dataset.jsonl", sample_size=500)